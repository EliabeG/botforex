# strategies/momentum/cci_adx.py
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
from collections import deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("cci_adx")

class CCIADXStrategy(BaseStrategy):
    """
    Estratégia combinando CCI (Commodity Channel Index) com ADX
    
    Conceito:
    - CCI detecta condições de sobrecompra/sobrevenda
    - ADX confirma força da tendência
    - Entradas em pullbacks dentro de tendências fortes
    """
    
    def __init__(self):
        super().__init__("CCI_ADX")
        self.suitable_regimes = [MarketRegime.TREND]
        self.min_time_between_signals = 180
        
        # Buffers para cálculos
        self.typical_price_buffer = deque(maxlen=200)
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'cci_period': 20,
            'cci_overbought': 100,
            'cci_oversold': -100,
            'cci_extreme_ob': 200,
            'cci_extreme_os': -200,
            'adx_period': 14,
            'adx_threshold': 25,
            'adx_strong': 40,
            'use_divergence': True,
            'atr_multiplier_sl': 1.5,
            'atr_multiplier_tp': 2.5,
            'use_cci_zero_cross': True,
            'min_bars_in_zone': 2
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula CCI e ADX"""
        try:
            ticks = market_context.get('recent_ticks', [])
            period = max(self.parameters['cci_period'], self.parameters['adx_period'])
            
            if len(ticks) < period + 1:
                return {}
            
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            closes = np.array([t.mid for t in ticks])
            
            # Typical Price para CCI
            typical_prices = (highs + lows + closes) / 3
            
            # CCI Calculation
            cci = self._calculate_cci(typical_prices, self.parameters['cci_period'])
            
            # CCI Rate of Change (para detectar divergências)
            cci_roc = 0
            if len(self.typical_price_buffer) > 10:
                cci_10_bars_ago = self._calculate_cci(
                    np.array(list(self.typical_price_buffer)[-self.parameters['cci_period']-10:-10]),
                    self.parameters['cci_period']
                )
                cci_roc = cci - cci_10_bars_ago
            
            # ADX
            adx, plus_di, minus_di = self._calculate_adx_full(highs, lows, closes, self.parameters['adx_period'])
            
            # ATR
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Detectar zonas
            cci_zone = 'neutral'
            if cci > self.parameters['cci_extreme_ob']:
                cci_zone = 'extreme_overbought'
            elif cci > self.parameters['cci_overbought']:
                cci_zone = 'overbought'
            elif cci < self.parameters['cci_extreme_os']:
                cci_zone = 'extreme_oversold'
            elif cci < self.parameters['cci_oversold']:
                cci_zone = 'oversold'
            
            # Histórico de CCI para detectar tempo em zona
            self.typical_price_buffer.extend(typical_prices[-5:])
            
            indicators = {
                'cci': cci,
                'cci_zone': cci_zone,
                'cci_roc': cci_roc,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'di_difference': plus_di - minus_di,
                'atr': atr,
                'current_price': closes[-1],
                'price_roc': (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) > 10 else 0,
                'typical_price': typical_prices[-1]
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores CCI/ADX: {e}")
            return {}
    
    def _calculate_cci(self, typical_prices: np.ndarray, period: int) -> float:
        """Calcula Commodity Channel Index"""
        if len(typical_prices) < period:
            return 0
        
        tp_slice = typical_prices[-period:]
        sma = np.mean(tp_slice)
        
        # Mean Deviation
        mean_dev = np.mean(np.abs(tp_slice - sma))
        
        if mean_dev == 0:
            return 0
        
        # CCI = (TP - SMA) / (0.015 * Mean Deviation)
        cci = (typical_prices[-1] - sma) / (0.015 * mean_dev)
        
        return cci
    
    def _calculate_adx_full(self, highs: np.ndarray, lows: np.ndarray, 
                           closes: np.ndarray, period: int = 14) -> tuple:
        """Calcula ADX completo com DI+ e DI-"""
        if len(highs) < period * 2:
            return 0, 0, 0
        
        # True Range
        true_ranges = []
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(highs)):
            # TR
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
            
            # Directional Movement
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0)
                minus_dm.append(0)
        
        if len(true_ranges) < period:
            return 0, 0, 0
        
        # Smoothed averages (Wilder's smoothing)
        atr = self._wilders_smoothing(true_ranges, period)
        plus_di_raw = self._wilders_smoothing(plus_dm, period)
        minus_di_raw = self._wilders_smoothing(minus_dm, period)
        
        if atr == 0:
            return 0, 0, 0
        
        # DI+ and DI-
        plus_di = 100 * plus_di_raw / atr
        minus_di = 100 * minus_di_raw / atr
        
        # DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0, plus_di, minus_di
        
        dx = 100 * abs(plus_di - minus_di) / di_sum
        
        # ADX (smoothed DX)
        # Simplificado - usar média móvel
        adx = dx  # Em produção, calcular média móvel de DX
        
        return adx, plus_di, minus_di
    
    def _wilders_smoothing(self, values: list, period: int) -> float:
        """Wilder's smoothing (similar to EMA)"""
        if len(values) < period:
            return 0
        
        # Primeira média
        sma = np.mean(values[:period])
        
        # Aplicar smoothing
        smoothed = sma
        for i in range(period, len(values)):
            smoothed = (smoothed * (period - 1) + values[i]) / period
        
        return smoothed
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinais baseados em CCI e ADX"""
        indicators = self.indicators
        
        if not indicators or 'cci' not in indicators:
            return None
        
        # Verificar força da tendência (ADX)
        if indicators['adx'] < self.parameters['adx_threshold']:
            return None
        
        signal_type = None
        signal_reason = ""
        
        # Estratégia 1: Retorno de extremos com tendência forte
        if indicators['adx'] > self.parameters['adx_strong']:
            # Compra em oversold extremo
            if indicators['cci_zone'] == 'extreme_oversold' and indicators['di_difference'] > 0:
                signal_type = 'buy'
                signal_reason = "CCI Extreme Oversold + Strong Uptrend"
            
            # Venda em overbought extremo
            elif indicators['cci_zone'] == 'extreme_overbought' and indicators['di_difference'] < 0:
                signal_type = 'sell'
                signal_reason = "CCI Extreme Overbought + Strong Downtrend"
        
        # Estratégia 2: Cruzamento da linha zero
        elif self.parameters['use_cci_zero_cross']:
            # Detectar cruzamento (simplificado)
            if indicators['cci'] > 0 and indicators['cci_roc'] > 20:
                if indicators['plus_di'] > indicators['minus_di'] and indicators['adx'] > 25:
                    signal_type = 'buy'
                    signal_reason = "CCI Zero Cross Up + Uptrend"
            
            elif indicators['cci'] < 0 and indicators['cci_roc'] < -20:
                if indicators['minus_di'] > indicators['plus_di'] and indicators['adx'] > 25:
                    signal_type = 'sell'
                    signal_reason = "CCI Zero Cross Down + Downtrend"
        
        # Estratégia 3: Divergência
        if self.parameters['use_divergence'] and signal_type is None:
            divergence = self._check_divergence(indicators)
            if divergence == 'bullish':
                signal_type = 'buy'
                signal_reason = "Bullish CCI Divergence"
            elif divergence == 'bearish':
                signal_type = 'sell'
                signal_reason = "Bearish CCI Divergence"
        
        if signal_type:
            return self._create_cci_signal(signal_type, indicators, signal_reason, market_context)
        
        return None
    
    def _check_divergence(self, indicators: Dict) -> Optional[str]:
        """Verifica divergências entre preço e CCI"""
        # Simplificado - em produção, analisar pivots
        if indicators['price_roc'] > 0 and indicators['cci_roc'] < -10:
            return 'bearish'
        elif indicators['price_roc'] < 0 and indicators['cci_roc'] > 10:
            return 'bullish'
        
        return None
    
    def _create_cci_signal(self, signal_type: str, indicators: Dict,
                          reason: str, market_context: Dict) -> Signal:
        """Cria sinal com gestão de risco"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        # Stops dinâmicos baseados na força da tendência
        sl_multiplier = self.parameters['atr_multiplier_sl']
        tp_multiplier = self.parameters['atr_multiplier_tp']
        
        # Ajustar baseado no ADX
        if indicators['adx'] > 40:
            tp_multiplier *= 1.5  # Alvos maiores em tendências fortes
        
        if signal_type == 'buy':
            stop_loss = price - (atr * sl_multiplier)
            take_profit = price + (atr * tp_multiplier)
        else:
            stop_loss = price + (atr * sl_multiplier)
            take_profit = price - (atr * tp_multiplier)
        
        # Confiança baseada em múltiplos fatores
        confidence = self._calculate_signal_confidence(indicators, signal_type)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=reason,
            metadata={
                'cci': indicators['cci'],
                'cci_zone': indicators['cci_zone'],
                'adx': indicators['adx'],
                'di_difference': indicators['di_difference']
            }
        )
        
        return self.adjust_for_spread(signal, market_context.get('spread', 0))
    
    def _calculate_signal_confidence(self, indicators: Dict, signal_type: str) -> float:
        """Calcula confiança do sinal"""
        confidence = 0.6
        
        # ADX forte
        if indicators['adx'] > 40:
            confidence += 0.15
        elif indicators['adx'] > 30:
            confidence += 0.1
        
        # CCI em extremo
        if abs(indicators['cci']) > 150:
            confidence += 0.1
        
        # DI alinhado
        if signal_type == 'buy' and indicators['di_difference'] > 20:
            confidence += 0.1
        elif signal_type == 'sell' and indicators['di_difference'] < -20:
            confidence += 0.1
        
        # CCI e preço alinhados
        if signal_type == 'buy' and indicators['cci_roc'] > 0 and indicators['price_roc'] > 0:
            confidence += 0.05
        elif signal_type == 'sell' and indicators['cci_roc'] < 0 and indicators['price_roc'] < 0:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Define condições de saída"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída por CCI oposto extremo
        if position.side == 'buy':
            if indicators['cci'] > self.parameters['cci_extreme_ob']:
                return ExitSignal(
                    position_id=position.id,
                    reason="CCI Extreme Overbought",
                    exit_price=current_price
                )
            
            # Saída se tendência enfraquecer
            if indicators['adx'] < 20 or indicators['minus_di'] > indicators['plus_di'] + 10:
                return ExitSignal(
                    position_id=position.id,
                    reason="Trend Weakening",
                    exit_price=current_price
                )
        
        else:  # sell
            if indicators['cci'] < self.parameters['cci_extreme_os']:
                return ExitSignal(
                    position_id=position.id,
                    reason="CCI Extreme Oversold",
                    exit_price=current_price
                )
            
            if indicators['adx'] < 20 or indicators['plus_di'] > indicators['minus_di'] + 10:
                return ExitSignal(
                    position_id=position.id,
                    reason="Trend Weakening",
                    exit_price=current_price
                )
        
        return None