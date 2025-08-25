import logging
import csv
import os
import numpy as np
import pandas as pd
import schedule
from datetime import datetime, time, timedelta
from collections import deque
import threading
import json
import requests
from typing import Dict, List, Optional, Tuple

from config import (ALPHAVANTAGE_API_KEY, EASTERN_TZ, MARKET_OPEN, MARKET_CLOSE,
                    FOUR_HOUR_TIMES)
from models import (MarketRegime, Order, OrderStatus, OrderType,
                    Position, PositionStatus)
from data_access import DataManager
from order_management import OrderManager
from ibkr_client import IBKRClient
from data_providers import AlphaVantageDataProvider
from brokers import IBKRBroker

class Indicators:
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series, multiplier: float = 3) -> Tuple[pd.Series, pd.Series]:
        # Basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=str)
        
        for i in range(1, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 'bullish'
            elif close.iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 'bearish'
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
                if supertrend.iloc[i] == upper_band.iloc[i-1] and close.iloc[i] > upper_band.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 'bullish'
                elif supertrend.iloc[i] == lower_band.iloc[i-1] and close.iloc[i] < lower_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = 'bearish'
        
        return supertrend, direction
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

class DataRollUp:
    def __init__(self):
        self.valid_4h_start_times = FOUR_HOUR_TIMES
        
    def is_valid_session_time(self, dt):
        """Check if datetime is during trading hours"""
        return (MARKET_OPEN <= dt.time() <= MARKET_CLOSE and 
                dt.weekday() < 5)  # Monday-Friday
    
    def rollup_1h_to_4h(self, hourly_data):
        """
        Convert 1H bars to 4H bars respecting exchange sessions
        """
        # Filter to only trading hours
        session_data = hourly_data[hourly_data.index.map(self.is_valid_session_time)]
        
        if len(session_data) == 0:
            return pd.DataFrame()
        
        # Create 4H groups based on fixed start times
        four_hour_bars = []
        current_group = []
        current_start = None
        
        for idx, row in session_data.iterrows():
            # Check if this is a start of a new 4H session
            if current_start is None or (idx.time() in self.valid_4h_start_times and 
                                       len(current_group) > 0):
                if current_group:
                    four_hour_bars.append(self._create_4h_bar(current_group))
                current_group = [row]
                current_start = idx
            else:
                current_group.append(row)
        
        # Add the last group
        if current_group:
            four_hour_bars.append(self._create_4h_bar(current_group))
        
        return pd.DataFrame(four_hour_bars)
    
    def _create_4h_bar(self, group_rows):
        """Create a single 4H bar from a group of 1H bars"""
        opens = [r['open'] for r in group_rows]
        highs = [r['high'] for r in group_rows]
        lows = [r['low'] for r in group_rows]
        closes = [r['close'] for r in group_rows]
        volumes = [r['volume'] for r in group_rows]
        
        return {
            'timestamp': group_rows[0].name,
            'open': opens[0],
            'high': max(highs),
            'low': min(lows),
            'close': closes[-1],
            'volume': sum(volumes),
            'hour_count': len(group_rows)
        }

class SupportLevelAnalyzer:
    def __init__(self):
        self.support_priority = [
            'validated_trendline',
            'anchored_vwap',
            'sma_50',
            'ema_20'
        ]
    
    def identify_support_level(self, symbol, daily_data, four_hour_data):
        """
        Identify the nearest support level based on priority
        """
        # Calculate all potential support levels
        all_levels = self._calculate_all_support_levels(daily_data, four_hour_data)
        
        # Filter to only levels below current price
        current_price = four_hour_data['close'].iloc[-1]
        valid_levels = {k: v for k, v in all_levels.items() 
                       if v is not None and v < current_price}
        
        if not valid_levels:
            return {
                'support_level': None,
                'support_price': None,
                'distance_pct': None,
                'distance_atr': None,
                'all_levels': all_levels
            }
        
        # Find the nearest support based on priority
        selected_level = None
        selected_price = None
        
        for level_type in self.support_priority:
            if level_type in valid_levels:
                selected_level = level_type
                selected_price = valid_levels[level_type]
                break
        
        # If none found in priority list, use the closest one
        if selected_level is None:
            closest_level = min(valid_levels.items(), key=lambda x: abs(current_price - x[1]))
            selected_level, selected_price = closest_level
        
        # Calculate distances
        atr = daily_data['atr_14'].iloc[-1] if 'atr_14' in daily_data else 0
        distance_pct = ((current_price - selected_price) / current_price) * 100
        distance_atr = (current_price - selected_price) / atr if atr > 0 else 0
        
        return {
            'support_level': selected_level,
            'support_price': selected_price,
            'distance_pct': distance_pct,
            'distance_atr': distance_atr,
            'all_levels': all_levels
        }
    
    def _calculate_all_support_levels(self, daily_data, four_hour_data):
        """Calculate all potential support levels"""
        levels = {}
        
        # 1. Validated Trendline
        levels['validated_trendline'] = self._calculate_trendline_support(daily_data)
        
        # 2. Anchored VWAP
        levels['anchored_vwap'] = self._calculate_anchored_vwap(four_hour_data)
        
        # 3. SMA 50
        levels['sma_50'] = daily_data['sma_50'].iloc[-1] if 'sma_50' in daily_data else None
        
        # 4. EMA 20
        levels['ema_20'] = daily_data['ema_20'].iloc[-1] if 'ema_20' in daily_data else None
        
        return levels
    
    def _calculate_trendline_support(self, daily_data):
        """Calculate trendline support"""
        if len(daily_data) < 20:
            return None
        
        # Simple implementation: use recent swing low
        recent_lows = daily_data['low'].rolling(5).min().dropna()
        if len(recent_lows) > 0:
            return recent_lows.iloc[-1]
        
        return None
    
    def _calculate_anchored_vwap(self, four_hour_data):
        """Calculate Anchored VWAP"""
        if len(four_hour_data) < 20:
            return None
        
        # Find the most recent significant low
        recent_low_idx = four_hour_data['low'].idxmin()
        vwap_data = four_hour_data.loc[recent_low_idx:]
        
        # Calculate VWAP
        typical_price = (vwap_data['high'] + vwap_data['low'] + vwap_data['close']) / 3
        vwap = (typical_price * vwap_data['volume']).cumsum() / vwap_data['volume'].cumsum()
        
        return vwap.iloc[-1] if len(vwap) > 0 else None
    
    def is_near_support(self, support_info, max_distance_atr=0.5, max_distance_pct=0.5):
        """
        Check if price is near support based on the rules
        """
        if support_info['support_price'] is None:
            return False
        
        return (support_info['distance_atr'] <= max_distance_atr or 
                support_info['distance_pct'] <= max_distance_pct)

class ExitConfirmation:
    def __init__(self):
        self.required_signals = ['supertrend_bearish', 'macd_bear_cross']
    
    def check_1h_confirmation(self, symbol, hourly_data):
        """
        Check if both confirmation signals are present on the most recent closed 1H candle
        """
        if len(hourly_data) < 2:
            return {'confirmed': False, 'signals_present': [], 'timestamp': None}
        
        # Get the most recent fully closed candle (previous candle)
        latest_candle = hourly_data.iloc[-2]  # -1 is current forming candle
        
        signals_present = []
        
        # Check Supertrend flip to bearish
        if self._is_supertrend_bearish(latest_candle, hourly_data, -3):
            signals_present.append('supertrend_bearish')
        
        # Check MACD bear cross
        if self._is_macd_bear_cross(latest_candle, hourly_data, -3):
            signals_present.append('macd_bear_cross')
        
        confirmed = all(signal in signals_present for signal in self.required_signals)
        
        return {
            'confirmed': confirmed,
            'signals_present': signals_present,
            'timestamp': latest_candle.name,
            'symbol': symbol
        }
    
    def _is_supertrend_bearish(self, candle, data, prev_index):
        """Check if Supertrend flipped bearish on this candle"""
        if prev_index >= len(data) or prev_index >= -1:
            return False
        
        prev_candle = data.iloc[prev_index]
        return (prev_candle.get('supertrend_direction') == 'bullish' and 
                candle.get('supertrend_direction') == 'bearish')
    
    def _is_macd_bear_cross(self, candle, data, prev_index):
        """Check if MACD had a bearish cross on this candle"""
        if prev_index >= len(data) or prev_index >= -1:
            return False
        
        prev_candle = data.iloc[prev_index]
        return (prev_candle.get('macd_line', 0) > prev_candle.get('macd_signal', 0) and 
                candle.get('macd_line', 0) < candle.get('macd_signal', 0))

class ScoringEngine:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ScoringEngine")
    
    def calculate_entry_score(self, symbol, daily_data, four_hour_data, regime):
        """
        Calculate entry score based on the scoring rules
        """
        score_components = {}
        
        # A) Trend - 45 points
        trend_score = self._calculate_trend_score(daily_data, four_hour_data)
        score_components['trend'] = trend_score
        
        # B) Momentum - 30 points
        momentum_score = self._calculate_momentum_score(daily_data, four_hour_data)
        score_components['momentum'] = momentum_score
        
        # C) Volume - 15 points
        volume_score = self._calculate_volume_score(daily_data, four_hour_data)
        score_components['volume'] = volume_score
        
        # D) Setup/Location - 10 points
        setup_score = self._calculate_setup_score(daily_data, four_hour_data)
        score_components['setup'] = setup_score
        
        # Apply regime multipliers
        regime_multipliers = self._get_regime_multipliers(regime)
        weighted_trend = trend_score * regime_multipliers['trend']
        weighted_momentum = momentum_score * regime_multipliers['momentum']
        weighted_volume = volume_score * regime_multipliers['volume']
        weighted_setup = setup_score * regime_multipliers['setup']

        raw_total = weighted_trend + weighted_momentum + weighted_volume + weighted_setup

        # Normalize score back to 0-100 range based on weighted maximum
        max_total = (
            45 * regime_multipliers['trend'] +
            30 * regime_multipliers['momentum'] +
            15 * regime_multipliers['volume'] +
            10 * regime_multipliers['setup']
        )
        total_score = (raw_total / max_total * 100) if max_total else 0
        
        # Apply penalties
        penalties = self._calculate_penalties(daily_data, four_hour_data)
        total_score += penalties
        score_components['penalties'] = penalties
        
        # Ensure score is within bounds
        total_score = max(0, min(100, total_score))
        
        return total_score, score_components
    
    def _calculate_trend_score(self, daily_data, four_hour_data):
        """Calculate trend component of entry score"""
        score = 0
        
        # Price > 200SMA (Daily): 12
        if 'sma_200' in daily_data and daily_data['close'].iloc[-1] > daily_data['sma_200'].iloc[-1]:
            score += 12
        
        # Price > 50SMA (Daily): 10
        if 'sma_50' in daily_data and daily_data['close'].iloc[-1] > daily_data['sma_50'].iloc[-1]:
            score += 10
        
        # Golden Cross (50>200): 10
        if ('sma_50' in daily_data and 'sma_200' in daily_data and 
            daily_data['sma_50'].iloc[-1] > daily_data['sma_200'].iloc[-1]):
            score += 10
        
        # Supertrend (Daily) bullish: 8
        if ('supertrend_direction' in daily_data and 
            daily_data['supertrend_direction'].iloc[-1] == 'bullish'):
            score += 8
        
        # Supertrend (4H) bullish: 5
        if ('supertrend_direction' in four_hour_data and 
            four_hour_data['supertrend_direction'].iloc[-1] == 'bullish'):
            score += 5
        
        return score
    
    def _calculate_momentum_score(self, daily_data, four_hour_data):
        """Calculate momentum component of entry score"""
        score = 0
        
        # Daily RSI 55-70: 10
        if 'rsi_14' in daily_data:
            rsi = daily_data['rsi_14'].iloc[-1]
            if 55 <= rsi <= 70:
                score += 10
            elif rsi > 75:  # Overheat penalty
                score -= 5
        
        # Daily MACD line > signal: 8
        if ('macd_line' in daily_data and 'macd_signal' in daily_data and 
            daily_data['macd_line'].iloc[-1] > daily_data['macd_signal'].iloc[-1]):
            score += 8
        
        # Daily MACD hist > 0 and expanding: 6
        if ('macd_hist' in daily_data and len(daily_data) >= 2 and 
            daily_data['macd_hist'].iloc[-1] > 0 and 
            daily_data['macd_hist'].iloc[-1] > daily_data['macd_hist'].iloc[-2]):
            score += 6
        
        # 4H RSI > 55 and rising: 6
        if ('rsi_14' in four_hour_data and len(four_hour_data) >= 2 and 
            four_hour_data['rsi_14'].iloc[-1] > 55 and 
            four_hour_data['rsi_14'].iloc[-1] > four_hour_data['rsi_14'].iloc[-2]):
            score += 6
        
        return score
    
    def _calculate_volume_score(self, daily_data, four_hour_data):
        """Calculate volume component of entry score"""
        score = 0
        
        # 20d Avg Vol ≥ 1M: 5
        if 'volume_20d_avg' in daily_data and daily_data['volume_20d_avg'].iloc[-1] >= 1000000:
            score += 5
        
        # Session vol ≥ 1.2× 20d avg: 6
        if ('volume_20d_avg' in daily_data and 'volume' in four_hour_data and 
            four_hour_data['volume'].iloc[-1] >= 1.2 * daily_data['volume_20d_avg'].iloc[-1]):
            score += 6
        
        # OBV(20) slope > 0: 4
        if 'obv_slope' in daily_data and daily_data['obv_slope'].iloc[-1] > 0:
            score += 4
        
        return score
    
    def _calculate_setup_score(self, daily_data, four_hour_data):
        """Calculate setup/location component of entry score"""
        score = 0
        
        # Pullback to value: 6
        # This would require more complex logic to determine pullbacks
        # Simplified implementation
        if ('ema_20' in daily_data and 'sma_50' in daily_data and 
            daily_data['close'].iloc[-1] <= daily_data['ema_20'].iloc[-1] * 1.02):
            score += 6
        
        # BB upper half (mid→upper): 4
        if ('bb_middle' in daily_data and 'bb_upper' in daily_data and 
            daily_data['close'].iloc[-1] > daily_data['bb_middle'].iloc[-1] and
            daily_data['close'].iloc[-1] < daily_data['bb_upper'].iloc[-1]):
            score += 4
        
        return score
    
    def _calculate_penalties(self, daily_data, four_hour_data):
        """Calculate penalties for entry score"""
        penalties = 0
        
        # Extended from 20EMA > +2.5×ATR(20): −6
        if ('ema_20' in daily_data and 'atr_14' in daily_data and 
            daily_data['close'].iloc[-1] > daily_data['ema_20'].iloc[-1] + 2.5 * daily_data['atr_14'].iloc[-1]):
            penalties -= 6
        
        # Gap-up > 2% at open: −3
        if len(daily_data) >= 2 and daily_data['open'].iloc[-1] > daily_data['close'].iloc[-2] * 1.02:
            penalties -= 3
        
        return penalties
    
    def _get_regime_multipliers(self, regime):
        """Get regime multipliers for different score components"""
        if regime == MarketRegime.TRENDING:
            return {'trend': 1.15, 'momentum': 1.10, 'volume': 1.0, 'setup': 0.9}
        elif regime == MarketRegime.RANGING:
            return {'trend': 0.85, 'momentum': 0.95, 'volume': 1.0, 'setup': 1.2}
        elif regime == MarketRegime.RISK_OFF:
            return {'trend': 0.9, 'momentum': 0.85, 'volume': 0.9, 'setup': 0.8}
        else:
            return {'trend': 1.0, 'momentum': 1.0, 'volume': 1.0, 'setup': 1.0}
    
    def calculate_exit_score(self, symbol, daily_data, four_hour_data, hourly_data):
        """
        Calculate exit score based on the scoring rules
        """
        score = 0
        
        # 4H/Daily signals
        if ('supertrend_direction' in four_hour_data and 
            four_hour_data['supertrend_direction'].iloc[-1] == 'bearish'):
            score += 3
        
        if ('macd_line' in four_hour_data and 'macd_signal' in four_hour_data and 
            four_hour_data['macd_line'].iloc[-1] < four_hour_data['macd_signal'].iloc[-1]):
            score += 2
        
        if ('sma_20' in four_hour_data and 
            four_hour_data['close'].iloc[-1] < four_hour_data['sma_20'].iloc[-1]):
            score += 1
        
        if ('rsi_14' in four_hour_data and len(four_hour_data) >= 2 and 
            four_hour_data['rsi_14'].iloc[-1] < 50 and 
            four_hour_data['rsi_14'].iloc[-2] >= 50):
            score += 1
        
        # Check for bearish engulfing or two lower closes
        if len(four_hour_data) >= 3:
            current = four_hour_data.iloc[-1]
            prev1 = four_hour_data.iloc[-2]
            prev2 = four_hour_data.iloc[-3]
            
            # Bearish engulfing
            if (current['open'] > prev1['close'] and current['close'] < prev1['open'] and
                current['high'] > prev1['high'] and current['low'] < prev1['low']):
                score += 1
            
            # Two lower closes
            if current['close'] < prev1['close'] and prev1['close'] < prev2['close']:
                score += 1
        
        if ('sma_50' in daily_data and 
            daily_data['close'].iloc[-1] < daily_data['sma_50'].iloc[-1]):
            score += 2
        
        if ('volume_20d_avg' in daily_data and 'volume' in daily_data and 
            daily_data['volume'].iloc[-1] > 1.5 * daily_data['volume_20d_avg'].iloc[-1] and
            daily_data['close'].iloc[-1] < daily_data['open'].iloc[-1]):
            score += 2
        
        # 1H Acceleration/Confirmation
        exit_confirmation = ExitConfirmation()
        confirmation = {
            'confirmed': False,
            'signals_present': [],
            'timestamp': None,
            'symbol': symbol
        }
        if hourly_data is not None:
            confirmation = exit_confirmation.check_1h_confirmation(symbol, hourly_data)
            if confirmation['confirmed']:
                score += 1

        return score, confirmation

class RegimeDetector:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RegimeDetector")
    
    def detect_regime(self, spy_data, vix_value):
        """
        Detect market regime based on SPY and VIX data
        """
        if len(spy_data) < 50:
            return MarketRegime.RISK_OFF
        
        # Calculate required indicators
        spy_data = spy_data.copy()
        spy_data['sma_50'] = Indicators.calculate_sma(spy_data['close'], 50)
        spy_data['sma_200'] = Indicators.calculate_sma(spy_data['close'], 200)

        high, low, close = spy_data['high'], spy_data['low'], spy_data['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        up_move = high.diff()
        down_move = low.shift() - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        current_close = close.iloc[-1]
        current_sma_50 = spy_data['sma_50'].iloc[-1]
        current_sma_200 = spy_data['sma_200'].iloc[-1]
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

        slope_50 = 0
        sma50 = spy_data['sma_50']
        if len(sma50) >= 21 and not pd.isna(sma50.iloc[-21]):
            slope_50 = (sma50.iloc[-1] - sma50.iloc[-21]) / 20 / sma50.iloc[-21]

        trending_conditions = (
            current_close > current_sma_50 > current_sma_200 and
            current_adx >= 20 and
            vix_value < 22
        )

        ranging_conditions = (
            abs(slope_50) < 0.0005 and
            current_adx < 20 and
            18 <= vix_value <= 26
        )

        risk_off_conditions = (
            current_close < current_sma_200 or
            vix_value >= 26
        )

        if trending_conditions:
            return MarketRegime.TRENDING
        if risk_off_conditions:
            return MarketRegime.RISK_OFF
        if ranging_conditions:
            return MarketRegime.RANGING
        return MarketRegime.RANGING

class SentimentAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SentimentAnalyzer")
    
    def get_fear_greed_index(self):
        """
        Get Fear & Greed Index from alternative source
        In a real implementation, this would use an API
        """
        try:
            # This is a placeholder - in reality you would use a proper API
            # response = requests.get('https://api.alternative.me/fng/', timeout=10)
            # data = response.json()
            # return int(data['data'][0]['value'])
            return 50  # Default neutral value
        except Exception as e:
            self.logger.warning(f"Failed to get Fear & Greed Index: {e}")
            return 50  # Default neutral value
    
    def get_news_sentiment(self, symbol):
        """
        Get news sentiment for a symbol
        In a real implementation, this would use a news API
        """
        try:
            # Placeholder for news sentiment API
            # This would typically return a score between -1 (very bearish) and 1 (very bullish)
            return 0  # Default neutral
        except Exception as e:
            self.logger.warning(f"Failed to get news sentiment for {symbol}: {e}")
            return 0  # Default neutral
    
    def apply_sentiment_gate(self, entry_score, symbol, regime):
        """
        Apply sentiment gate to entry score
        """
        fg_index = self.get_fear_greed_index()
        news_sentiment = self.get_news_sentiment(symbol)
        
        # Apply Fear & Greed penalties
        if fg_index < 25:
            # Hard block - return very low score
            return 0, {"fg_index": fg_index, "news_sentiment": news_sentiment, "action": "hard_block"}
        elif 25 <= fg_index <= 45:
            entry_score -= 5
        elif fg_index > 80:
            # Check if RSI > 70 for overheat penalty
            # This would require RSI data, simplified here
            entry_score -= 5
        
        # Apply news sentiment adjustments
        if news_sentiment > 0.7:  # Strong positive
            entry_score += 3
        elif news_sentiment < -0.7:  # Strong negative
            entry_score -= 5

        # Risk-off regime gating
        if regime == MarketRegime.RISK_OFF and (entry_score < 85 or news_sentiment < 0):
            return 0, {"fg_index": fg_index, "news_sentiment": news_sentiment, "action": "risk_off_block"}

        # Ensure score is within bounds
        entry_score = max(0, min(100, entry_score))

        return entry_score, {"fg_index": fg_index, "news_sentiment": news_sentiment, "action": "adjusted"}

class TradingBot:
    def __init__(self, broker: IBKRBroker | None = None, data_provider: AlphaVantageDataProvider | None = None):
        self.logger = logging.getLogger(f"{__name__}.TradingBot")

        self.data_manager = DataManager()
        self.broker = broker
        self.data_provider = data_provider
        self.scoring_engine = ScoringEngine()
        self.regime_detector = RegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.support_analyzer = SupportLevelAnalyzer()
        self.data_rollup = DataRollUp()
        self.exit_confirmation = ExitConfirmation()

        # Attempt IBKR connection using env configuration
        try:
            self.ibkr = IBKRClient()
            self.ibkr.connect_and_run()
            if self.broker is None:
                self.broker = IBKRBroker(self.ibkr)
        except Exception as e:  # pragma: no cover - network dependent
            self.logger.warning(f"IBKR client unavailable: {e}")
            self.ibkr = None

        if self.data_provider is None and ALPHAVANTAGE_API_KEY:
            self.data_provider = AlphaVantageDataProvider(ALPHAVANTAGE_API_KEY)

        self.order_manager = OrderManager(self.data_manager, broker=self.broker)
        # Load any persisted positions and display account state
        self.positions = self.data_manager.load_positions()
        self._print_account_status()

        self.universe = self._load_sp100_universe()
        self.current_regime = MarketRegime.RISK_OFF
    
    def _load_sp100_universe(self):
        """Load the S&P 100 universe from a CSV file specified by `SP100_CSV`."""
        csv_path = os.environ.get("SP100_CSV")
        if not csv_path or not os.path.exists(csv_path):
            raise RuntimeError("SP100_CSV environment variable must point to constituent CSV")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return [row["Symbol"].strip() for row in reader if row.get("Symbol")]

    def _print_account_status(self):
        """Log current account balance and any open positions."""
        if self.ibkr:
            try:
                summary = self.ibkr.get_account_summary()
                cash = summary.get("TotalCashValue")
                buying_power = summary.get("BuyingPower")
                self.logger.info(
                    f"Account cash: {cash}, buying power: {buying_power}"
                )
            except Exception as e:  # pragma: no cover - network dependent
                self.logger.warning(f"Unable to fetch account balance: {e}")
        else:
            self.logger.info("Account balance unavailable: IBKR client not connected")

        open_positions = [
            p for p in self.positions.values()
            if p.status != PositionStatus.EXITED
        ]
        if open_positions:
            for pos in open_positions:
                self.logger.info(
                    f"Position {pos.symbol}: qty={pos.quantity} entry={pos.entry_price}"
                )
        else:
            self.logger.info("No open positions loaded")
    
    def run_hourly(self):
        """Main hourly execution loop"""
        now = datetime.now(EASTERN_TZ)
        self.logger.info(f"Starting hourly run at {now}")
        
        # Skip if outside market hours
        if not (MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5):
            self.logger.info("Outside market hours, skipping")
            return
        
        # 1. Detect market regime
        self._detect_market_regime()
        
        # 2. Process each symbol in universe
        for symbol in self.universe:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
        
        # 3. Manage existing positions
        self._manage_positions()
        
        self.logger.info("Hourly run completed")
    
    def _detect_market_regime(self):
        """Detect current market regime"""
        # Fetch SPY data via data provider
        spy_data = self._fetch_symbol_data('SPY', 250, '1D')
        vix_value = self._fetch_vix_value(spy_data)
        
        self.current_regime = self.regime_detector.detect_regime(spy_data, vix_value)
        self.logger.info(f"Detected market regime: {self.current_regime.value}")
    
    def _process_symbol(self, symbol):
        """Process a single symbol for potential entry"""
        # Get historical data for symbol
        daily_data = self._fetch_symbol_data(symbol, 100, '1D')
        hourly_data = self._fetch_symbol_data(symbol, 50, '1H')
        
        # Roll up 1H to 4H data
        four_hour_data = self.data_rollup.rollup_1h_to_4h(hourly_data)
        
        if len(four_hour_data) == 0:
            self.logger.warning(f"Insufficient data for {symbol}, skipping")
            return
        
        # Calculate indicators
        daily_data = self._calculate_indicators(daily_data)
        four_hour_data = self._calculate_indicators(four_hour_data)
        
        # Calculate entry score
        entry_score, score_components = self.scoring_engine.calculate_entry_score(
            symbol, daily_data, four_hour_data, self.current_regime
        )
        
        # Apply sentiment gate
        entry_score, sentiment_info = self.sentiment_analyzer.apply_sentiment_gate(
            entry_score, symbol, self.current_regime
        )
        
        self.logger.info(f"{symbol} entry score: {entry_score}, components: {score_components}")
        
        # Check if entry conditions are met
        if entry_score >= 70:
            # Check near-support rule
            support_info = self.support_analyzer.identify_support_level(symbol, daily_data, four_hour_data)
            near_support = self.support_analyzer.is_near_support(support_info)
            
            if near_support:
                self.logger.info(f"{symbol} meets entry criteria near support: {support_info}")
                self._place_entry_order(symbol, daily_data, four_hour_data, support_info, entry_score, score_components)
            else:
                self.logger.info(f"{symbol} meets score criteria but not near support: {support_info}")
    
    def _place_entry_order(self, symbol, daily_data, four_hour_data, support_info, entry_score, score_components):
        """Place an entry order for a symbol"""
        # Calculate position size
        account_equity = 100000  # Example equity
        risk_per_trade = account_equity * 0.01  # 1% risk per trade
        
        current_price = four_hour_data['close'].iloc[-1]
        atr = daily_data['atr_14'].iloc[-1]
        
        # Set stop loss below support
        stop_price = support_info['support_price'] - atr
        risk_per_share = current_price - stop_price
        
        # Calculate position size
        position_size = int(risk_per_trade / risk_per_share)
        
        # Ensure we don't exceed available capital
        max_capital = account_equity * 0.1  # Max 10% per position
        max_shares = int(max_capital / current_price)
        position_size = min(position_size, max_shares)
        
        if position_size <= 0:
            self.logger.warning(f"Position size too small for {symbol}: {position_size}")
            return
        
        # Place bracket order
        take_profit1 = current_price + 1.5 * risk_per_share  # 1.5R
        take_profit2 = current_price + 2.0 * risk_per_share  # 2.0R
        
        order_id = self.order_manager.place_bracket_order(
            symbol, position_size, current_price, stop_price, take_profit1, take_profit2
        )
        
        # Create position record
        position_id = f"POS_{symbol}_{datetime.now().timestamp()}"
        position = Position(
            position_id=position_id,
            symbol=symbol,
            entry_time=datetime.now(EASTERN_TZ),
            entry_price=current_price,
            quantity=position_size,
            status=PositionStatus.ARMED,
            risk_per_share=risk_per_share,
            score_components=score_components
        )
        
        self.positions[position_id] = position
        self.data_manager.save_position(position)
        
        self.logger.info(f"Entry order placed for {symbol}: {position_size} shares at {current_price}")
    
    def _manage_positions(self):
        """Manage existing positions"""
        for position_id, position in list(self.positions.items()):
            if position.status == PositionStatus.EXITED:
                continue
                
            try:
                # Get current data
                daily_data = self._fetch_symbol_data(position.symbol, 100, '1D')
                four_hour_data = self._fetch_symbol_data(position.symbol, 50, '4H')
                hourly_data = self._fetch_symbol_data(position.symbol, 24, '1H')
                
                # Calculate indicators
                daily_data = self._calculate_indicators(daily_data)
                four_hour_data = self._calculate_indicators(four_hour_data)
                hourly_data = self._calculate_indicators(hourly_data)
                
                # Calculate exit score and 1H confirmation
                exit_score, confirmation = self.scoring_engine.calculate_exit_score(
                    position.symbol, daily_data, four_hour_data, hourly_data
                )

                # Update position PnL
                current_price = four_hour_data['close'].iloc[-1]
                position.current_pnl = (current_price - position.entry_price) * position.quantity

                # Apply exit ladder only when 1H confirmation is present
                if confirmation.get("confirmed"):
                    self._apply_exit_ladder(position, exit_score, current_price, daily_data, four_hour_data)

                # Check for trail promotion
                self._check_trail_promotion(position, daily_data, four_hour_data)
                
                # Save updated position
                self.data_manager.save_position(position)
                
            except Exception as e:
                self.logger.error(f"Error managing position {position_id}: {e}")
    
    def _apply_exit_ladder(self, position, exit_score, current_price, daily_data, four_hour_data):
        """Apply exit ladder based on exit score"""
        if exit_score >= 9:
            # Exit 100%
            self._exit_position(position, current_price, "exit_score_9")
            position.status = PositionStatus.EXITED
        elif exit_score >= 7 and position.status != PositionStatus.SCALE_OUT_75:
            # Scale out to 75% total
            self._scale_out_position(position, 0.75, current_price, "exit_score_7")
            position.status = PositionStatus.SCALE_OUT_75
            self._ratchet_stop(position, daily_data, four_hour_data)
        elif exit_score >= 5 and position.status not in [PositionStatus.SCALE_OUT_50, PositionStatus.SCALE_OUT_75]:
            # Scale out to 50% total
            self._scale_out_position(position, 0.50, current_price, "exit_score_5")
            position.status = PositionStatus.SCALE_OUT_50
            self._ratchet_stop(position, daily_data, four_hour_data)
        elif exit_score >= 3 and position.status not in [PositionStatus.SCALE_OUT_25, PositionStatus.SCALE_OUT_50, PositionStatus.SCALE_OUT_75]:
            # Scale out 25%
            self._scale_out_position(position, 0.25, current_price, "exit_score_3")
            position.status = PositionStatus.SCALE_OUT_25
            self._ratchet_stop(position, daily_data, four_hour_data)
    
    def _scale_out_position(self, position, scale_percent, current_price, reason):
        """Scale out of a position"""
        shares_to_sell = int(position.quantity * scale_percent)
        
        # Place sell order
        order_id = f"SO_{position.symbol}_{datetime.now().timestamp()}"
        order = Order(
            order_id=order_id,
            symbol=position.symbol,
            order_type=OrderType.LMT,
            side="SELL",
            quantity=shares_to_sell,
            limit_price=current_price,
            parent_order_id=position.position_id,
            status=OrderStatus.PENDING_NEW
        )
        
        self.order_manager.orders[order_id] = order
        self.data_manager.save_order(order)
        
        # Update position
        position.quantity -= shares_to_sell
        position.realized_pnl += (current_price - position.entry_price) * shares_to_sell

        self.logger.info(f"Scaled out {scale_percent*100}% of {position.symbol}: {shares_to_sell} shares at {current_price}")

    def _exit_position(self, position, current_price, reason):
        """Fully exit a position"""
        order_id = f"EX_{position.symbol}_{datetime.now().timestamp()}"
        order = Order(
            order_id=order_id,
            symbol=position.symbol,
            order_type=OrderType.LMT,
            side="SELL",
            quantity=position.quantity,
            limit_price=current_price,
            parent_order_id=position.position_id,
            status=OrderStatus.PENDING_NEW
        )
        
        self.order_manager.orders[order_id] = order
        self.data_manager.save_order(order)
        
        # Update position
        position.realized_pnl += (current_price - position.entry_price) * position.quantity
        position.quantity = 0
        position.exit_price = current_price
        position.exit_time = datetime.now(EASTERN_TZ)

        self.logger.info(f"Exited {position.symbol}: {position.quantity} shares at {current_price}")

    def _ratchet_stop(self, position, daily_data, four_hour_data):
        """Adjust the protective stop according to the ratchet rules."""
        ema20 = daily_data.get("ema_20", pd.Series()).iloc[-1] if "ema_20" in daily_data else position.entry_price
        atr = daily_data.get("atr_14", pd.Series()).iloc[-1] if "atr_14" in daily_data else 0
        chandelier = position.entry_price
        if len(four_hour_data) >= 22 and atr > 0:
            chandelier = four_hour_data["high"].rolling(22).max().iloc[-1] - 3 * atr

        new_stop = max(position.entry_price, ema20 - atr, chandelier)
        self.order_manager.adjust_stop_loss(position.position_id, new_stop, position.quantity)
    
    def _check_trail_promotion(self, position, daily_data, four_hour_data):
        """Check if position should be promoted to trailing stop"""
        if position.status != PositionStatus.FILLED:
            return
        
        # Check if PT1 hit (simplified)
        current_price = four_hour_data['close'].iloc[-1]
        pt1_price = position.entry_price + 1.5 * position.risk_per_share
        
        if current_price >= pt1_price:
            # Check exit score and daily trend
            exit_score, _ = self.scoring_engine.calculate_exit_score(
                position.symbol, daily_data, four_hour_data, None
            )

            # Check if daily trend is still bullish
            daily_trend_bullish = (
                'supertrend_direction' in daily_data and
                daily_data['supertrend_direction'].iloc[-1] == 'bullish'
            )

            if exit_score <= 2 and daily_trend_bullish:
                # Promote to trailing stop
                atr = daily_data['atr_14'].iloc[-1]
                trail_amount = 2.5 * atr

                success = self.order_manager.promote_to_trailing_stop(
                    position.position_id, trail_amount
                )

                if success:
                    self.logger.info(f"Promoted {position.symbol} to trailing stop: {trail_amount}")

        # Upgrade trail to chandelier on sustained strength
        has_trail = any(
            self.order_manager.orders[oid].order_type == OrderType.TRAIL
            for oid in self.order_manager.position_orders.get(position.position_id, [])
            if oid in self.order_manager.orders
        )
        if has_trail and current_price >= position.entry_price + 2 * position.risk_per_share:
            atr = daily_data['atr_14'].iloc[-1]
            self.order_manager.upgrade_trailing_stop(position.position_id, 3 * atr)
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators for data"""
        data = data.copy()
        
        # Calculate basic indicators
        data['sma_20'] = Indicators.calculate_sma(data['close'], 20)
        data['sma_50'] = Indicators.calculate_sma(data['close'], 50)
        data['sma_200'] = Indicators.calculate_sma(data['close'], 200)
        data['ema_20'] = Indicators.calculate_ema(data['close'], 20)
        data['rsi_14'] = Indicators.calculate_rsi(data['close'], 14)
        
        # Calculate MACD
        macd_line, macd_signal, macd_hist = Indicators.calculate_macd(data['close'])
        data['macd_line'] = macd_line
        data['macd_signal'] = macd_signal
        data['macd_hist'] = macd_hist
        
        # Calculate ATR
        data['atr_14'] = Indicators.calculate_atr(data['high'], data['low'], data['close'], 14)
        
        # Calculate Supertrend
        supertrend, supertrend_direction = Indicators.calculate_supertrend(
            data['high'], data['low'], data['close'], data['atr_14']
        )
        data['supertrend'] = supertrend
        data['supertrend_direction'] = supertrend_direction
        
        # Calculate OBV
        data['obv'] = Indicators.calculate_obv(data['close'], data['volume'])
        
        # Calculate OBV slope (simplified)
        if len(data) >= 20:
            data['obv_slope'] = data['obv'].rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
            )
        
        # Calculate volume averages
        if len(data) >= 20:
            data['volume_20d_avg'] = data['volume'].rolling(20).mean()

        return data

    def _fetch_symbol_data(self, symbol: str, periods: int, timeframe: str) -> pd.DataFrame:
        """Fetch market data from the configured provider."""
        if not self.data_provider:
            raise RuntimeError("No data provider configured")

        if timeframe == '4H':
            interval = self._provider_interval('1H')
            df_1h = self.data_provider.get_historical_data(symbol, interval, periods * 4)
            if df_1h.empty:
                raise ValueError(f"No data returned for {symbol}")
            return self.data_rollup.rollup_1h_to_4h(df_1h)

        interval = self._provider_interval(timeframe)
        df = self.data_provider.get_historical_data(symbol, interval, periods)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        return df

    def _fetch_vix_value(self, spy_data: pd.DataFrame) -> float:
        """
        Return latest VIX quote or fall back to a realized-volatility proxy
        based on SPY ATR percentile.
        """
        vix = None
        if self.data_provider:
            for symbol in ("^VIX", "VIX"):
                try:  # pragma: no cover - network dependent
                    vix = self.data_provider.get_quote(symbol)
                except Exception:  # pragma: no cover - network dependent
                    vix = None
                if vix is not None:
                    return vix

        # Fallback: approximate VIX using SPY ATR percentile
        self.logger.warning(
            "Failed to fetch VIX; using SPY ATR percentile as proxy"
        )
        atr = Indicators.calculate_atr(
            spy_data["high"], spy_data["low"], spy_data["close"], 14
        )
        atr_pct = (atr / spy_data["close"]) * 100
        atr_pct = atr_pct.dropna()
        if len(atr_pct) >= 20:
            percentile = atr_pct.rank(pct=True).iloc[-1]
        else:
            percentile = 0.5
        # Map percentile to an approximate VIX range (12-40)
        vix_proxy = 12 + percentile * 28
        return float(vix_proxy)

    def _provider_interval(self, timeframe: str) -> str:
        mapping = {"1H": "60min", "1D": "daily"}
        return mapping.get(timeframe, "60min")

