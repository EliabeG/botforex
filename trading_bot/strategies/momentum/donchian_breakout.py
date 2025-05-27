# strategies/momentum/donchian_breakout.py
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("donchian_breakout")

class DonchianBreakoutStrategy(BaseStrategy):
    """
    Estratégia de breakout usando Donchian Channel
    
    Conceito:
    - Compra quando preço rompe máxima de N períodos
    - Vende quando preço rompe mínima de N períodos
    - Usa ATR para stops dinâmicos
    """
    
    def __init__(self):
        super().__init__("DonchianBreakout55")
        self.suitable_regimes = [MarketRegime.TREND]
        self.min_time_between_signals = 300
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'channel_period': 55,
            'exit_period': 20,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 4.0,
            'use_middle_exit': True,
            'min_channel_width': 0.0010,  # 10 pips mínimo
            'breakout_confirmation': 2,    # Ticks de confirmação
            'volume_surge_required': 1.2,  # 20% mais volume
            'use_time_filter': True,
            'avoid_news_minutes': 30
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula Donchian Channel e indicadores auxiliares"""
        try:
            ticks = market_context.get('recent_ticks', [])
            if len(ticks) < self.parameters['channel_period'] + 1:
                return {}
            
            # Extrair dados
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            closes = np.array([t.mid for t in ticks])
            
            # Donchian Channel principal
            period = self.parameters['channel_period']
            upper_channel = np.max(highs[-period:])
            lower_channel = np.min(lows[-period:])
            middle_channel = (upper_channel + lower_channel) / 2
            channel_width = upper_channel - lower_channel
            
            # Donchian para saída
            exit_period = self.parameters['exit_period']
            upper_exit = np.max(highs[-exit_period:])
            lower_exit = np.min(lows[-exit_period:])
            
            # ATR para stops
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Volume
            volumes = []
            for i in range(1, len(ticks)):
                tick_vol = (ticks[i].bid_volume + ticks[i].ask_volume) / 2
                volumes.append(tick_vol)
            
            if volumes:
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # ADX para confirmar tendência
            adx = self._calculate_adx(highs, lows, closes)
            
            indicators = {
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'middle_channel': middle_channel,
                'channel_width': channel_width,
                'upper_exit': upper_exit,
                'lower_exit': lower_exit,
                'current_price': closes[-1],
                'previous_price': closes[-2] if len(closes) > 1 else closes[-1],
                'atr': atr,
                'volume_ratio': volume_ratio,
                'adx': adx,
                'position_relative': (closes[-1] - lower_channel) / channel_width if channel_width > 0 else 0.5
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return {}
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de breakout"""
        indicators = self.indicators
        
        if not indicators or 'upper_channel' not in indicators:
            return None
        
        # Verificar largura mínima do canal
        if indicators['channel_width'] < self.parameters['min_channel_width']:
            return None
        
        # Verificar ADX (tendência)
        if indicators['adx'] < 20:
            return None
        
        current_price = indicators['current_price']
        previous_price = indicators['previous_price']
        
        # Detectar breakout
        signal_type = None
        
        # Breakout de alta
        if (previous_price <= indicators['upper_channel'] and 
            current_price > indicators['upper_channel']):
            
            # Verificar volume
            if indicators['volume_ratio'] >= self.parameters['volume_surge_required']:
                signal_type = 'buy'
        
        # Breakout de baixa
        elif (previous_price >= indicators['lower_channel'] and 
              current_price < indicators['lower_channel']):
            
            # Verificar volume
            if indicators['volume_ratio'] >= self.parameters['volume_surge_required']:
                signal_type = 'sell'
        
        if signal_type:
            return self._create_breakout_signal(signal_type, indicators, market_context)
        
        return None
    
    def _create_breakout_signal(self, signal_type: str, indicators: Dict,
                               market_context: Dict) -> Signal:
        """Cria sinal com stops calculados"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        if signal_type == 'buy':
            stop_loss = indicators['lower_channel'] - (atr * 0.5)
            take_profit = price + (atr * self.parameters['atr_multiplier_tp'])
        else:
            stop_loss = indicators['upper_channel'] + (atr * 0.5)
            take_profit = price - (atr * self.parameters['atr_multiplier_tp'])
        
        confidence = self._calculate_breakout_confidence(indicators, signal_type)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Donchian Breakout {signal_type.upper()} - Width: {indicators['channel_width']*10000:.1f} pips",
            metadata={
                'channel_width': indicators['channel_width'],
                'volume_ratio': indicators['volume_ratio'],
                'adx': indicators['adx'],
                'breakout_level': indicators['upper_channel'] if signal_type == 'buy' else indicators['lower_channel']
            }
        )
        
        return self.adjust_for_spread(signal, market_context.get('spread', 0))
    
    def _calculate_breakout_confidence(self, indicators: Dict, signal_type: str) -> float:
        """Calcula confiança do breakout"""
        confidence = 0.6  # Base
        
        # ADX forte
        if indicators['adx'] > 30:
            confidence += 0.15
        elif indicators['adx'] > 25:
            confidence += 0.1
        
        # Volume alto
        if indicators['volume_ratio'] > 1.5:
            confidence += 0.1
        elif indicators['volume_ratio'] > 1.2:
            confidence += 0.05
        
        # Canal largo (movimento significativo)
        if indicators['channel_width'] > 0.0020:  # 20 pips
            confidence += 0.1
        
        # Posição no canal antes do breakout
        if signal_type == 'buy' and indicators['position_relative'] > 0.7:
            confidence += 0.05  # Já estava na parte superior
        elif signal_type == 'sell' and indicators['position_relative'] < 0.3:
            confidence += 0.05  # Já estava na parte inferior
        
        return min(confidence, 0.95)
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Define condições de saída"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída por canal oposto ou médio
        if position.side == 'buy':
            exit_level = indicators['lower_exit'] if not self.parameters['use_middle_exit'] else indicators['middle_channel']
            
            if current_price <= exit_level:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Donchian Exit - {'Middle' if self.parameters['use_middle_exit'] else 'Lower'} Channel",
                    exit_price=current_price
                )
        
        else:  # sell
            exit_level = indicators['upper_exit'] if not self.parameters['use_middle_exit'] else indicators['middle_channel']
            
            if current_price >= exit_level:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Donchian Exit - {'Middle' if self.parameters['use_middle_exit'] else 'Upper'} Channel",
                    exit_price=current_price
                )
        
        # Trailing stop baseado em ATR
        if position.trailing_stop:
            new_stop = self.calculate_trailing_stop(
                position,
                current_price,
                indicators['atr'],
                1.5  # Multiplicador mais apertado
            )
            
            if position.side == 'buy' and new_stop > position.stop_loss:
                position.stop_loss = new_stop
            elif position.side == 'sell' and new_stop < position.stop_loss:
                position.stop_loss = new_stop
        
        return None
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int = 14) -> float:
        """Calcula ADX simplificado"""
        if len(high) < period + 1:
            return 0
        
        # Implementação simplificada
        # Em produção, usar TA-Lib
        true_ranges = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return 0
        
        atr = np.mean(true_ranges[-period:])
        
        # Movimento direcional
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0)
                minus_dm.append(0)
        
        if len(plus_dm) < period or atr == 0:
            return 0
        
        # DI+ e DI-
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr
        
        # DX
        if plus_di + minus_di == 0:
            return 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        return dx  # Simplificado - normalmente seria média do DX