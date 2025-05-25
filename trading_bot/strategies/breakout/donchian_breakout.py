# strategies/breakout/donchian_breakout.py
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("donchian_breakout_strategy")

class DonchianBreakoutStrategy(BaseStrategy):
    """
    Estratégia de breakout usando Donchian Channel
    
    Setup:
    - Compra quando preço rompe máxima de N períodos
    - Vende quando preço rompe mínima de N períodos
    - Confirma com volume e momentum
    """
    
    def __init__(self):
        super().__init__("DonchianBreakout_55")
        self.suitable_regimes = [MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY]
        self.min_time_between_signals = 600  # 10 minutos
        
        # Tracking breakouts
        self.last_high = None
        self.last_low = None
        self.breakout_confirmed = False
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Donchian parameters
            'lookback_period': 55,
            'exit_lookback': 20,
            'use_mid_exit': True,
            
            # Confirmation filters
            'min_breakout_distance': 0.0005,  # 5 pips mínimo
            'volume_surge_required': 1.5,      # 150% da média
            'momentum_confirmation': True,
            'confirmation_bars': 2,
            
            # Risk management
            'atr_multiplier_sl': 2.0,
            'risk_reward_ratio': 2.5,
            'use_trailing_stop': True,
            'trailing_activation': 0.002,  # 20 pips
            
            # Filters
            'min_atr': 0.0003,  # 3 pips
            'max_spread': 0.0001,  # 1 pip
            'avoid_news_window': 30,  # minutos
            'london_open_filter': True,
            'ny_open_filter': True,
            
            # Position management
            'scale_out_enabled': True,
            'scale_out_levels': [(0.5, 0.5), (1.0, 0.3), (1.5, 0.2)],  # (R múltiplo, % saída)
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula Donchian Channel e indicadores auxiliares"""
        try:
            recent_ticks = market_context.get('recent_ticks', [])
            lookback = self.parameters['lookback_period']
            
            if len(recent_ticks) < lookback + 10:
                return {}
            
            # Converter para arrays
            highs = np.array([t.ask for t in recent_ticks])
            lows = np.array([t.bid for t in recent_ticks])
            closes = np.array([t.mid for t in recent_ticks])
            
            # Calcular Donchian Channel principal
            upper_channel = np.max(highs[-lookback:])
            lower_channel = np.min(lows[-lookback:])
            mid_channel = (upper_channel + lower_channel) / 2
            
            # Donchian de saída (período menor)
            exit_lookback = self.parameters['exit_lookback']
            exit_upper = np.max(highs[-exit_lookback:])
            exit_lower = np.min(lows[-exit_lookback:])
            
            # Preço atual
            current_price = closes[-1]
            current_high = highs[-1]
            current_low = lows[-1]
            
            # Detectar breakout
            breakout_up = current_high > upper_channel
            breakout_down = current_low < lower_channel
            
            # ATR para volatilidade
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Volume analysis
            dom = market_context.get('dom')
            volume_surge = 1.0
            if dom:
                depth = dom.get_depth(5)
                current_volume = depth['bid_volume'] + depth['ask_volume']
                avg_volume = 100000  # Simplificado - deveria calcular média real
                volume_surge = current_volume / avg_volume
            
            # Momentum
            momentum = closes[-1] - closes[-10]  # 10 períodos
            momentum_pct = momentum / closes[-10]
            
            # RSI para filtro
            rsi = self.calculate_rsi(closes, period=14)
            
            # ADX para força da tendência
            adx = self._calculate_adx(highs, lows, closes)
            
            # Distância do breakout
            if breakout_up:
                breakout_distance = current_high - upper_channel
            elif breakout_down:
                breakout_distance = lower_channel - current_low
            else:
                breakout_distance = 0
            
            indicators = {
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'mid_channel': mid_channel,
                'exit_upper': exit_upper,
                'exit_lower': exit_lower,
                'current_price': current_price,
                'current_high': current_high,
                'current_low': current_low,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'breakout_distance': breakout_distance,
                'atr': atr,
                'volume_surge': volume_surge,
                'momentum': momentum,
                'momentum_pct': momentum_pct,
                'rsi': rsi,
                'adx': adx,
                'spread': market_context.get('spread', 0),
                'channel_width': upper_channel - lower_channel,
                'position_in_channel': (current_price - lower_channel) / (upper_channel - lower_channel) if upper_channel > lower_channel else 0.5
            }
            
            # Atualizar tracking
            if self.last_high != upper_channel or self.last_low != lower_channel:
                self.last_high = upper_channel
                self.last_low = lower_channel
                self.breakout_confirmed = False
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return {}
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de breakout"""
        indicators = self.indicators
        
        if not indicators or 'upper_channel' not in indicators:
            return None
        
        # Verificar filtros
        if not self._check_entry_filters(indicators, market_context):
            return None
        
        # Detectar sinal
        signal_type = None
        
        # Breakout de alta
        if indicators['breakout_up'] and not self.breakout_confirmed:
            if self._confirm_breakout('up', indicators):
                signal_type = 'buy'
                self.breakout_confirmed = True
        
        # Breakout de baixa
        elif indicators['breakout_down'] and not self.breakout_confirmed:
            if self._confirm_breakout('down', indicators):
                signal_type = 'sell'
                self.breakout_confirmed = True
        
        if signal_type:
            return self._create_breakout_signal(signal_type, indicators, market_context)
        
        return None
    
    def _check_entry_filters(self, indicators: Dict, market_context: Dict) -> bool:
        """Verifica filtros antes de entrar"""
        # ATR mínimo (evitar mercados parados)
        if indicators['atr'] < self.parameters['min_atr']:
            return False
        
        # Spread máximo
        if indicators['spread'] > self.parameters['max_spread']:
            return False
        
        # Filtro de horário (evitar aberturas se configurado)
        hour = datetime.now().hour
        minute = datetime.now().minute
        
        if self.parameters['london_open_filter']:
            if hour == 7 and minute < 30:  # Primeiros 30 min de Londres
                return False
        
        if self.parameters['ny_open_filter']:
            if hour == 13 and minute < 30:  # Primeiros 30 min de NY
                return False
        
        # Regime adequado
        regime = market_context.get('regime')
        if regime not in self.suitable_regimes:
            return False
        
        return True
    
    def _confirm_breakout(self, direction: str, indicators: Dict) -> bool:
        """Confirma se breakout é válido"""
        # Distância mínima do breakout
        if indicators['breakout_distance'] < self.parameters['min_breakout_distance']:
            return False
        
        # Volume surge
        if indicators['volume_surge'] < self.parameters['volume_surge_required']:
            return False
        
        # Confirmação de momentum
        if self.parameters['momentum_confirmation']:
            if direction == 'up' and indicators['momentum_pct'] < 0.0001:  # 0.01%
                return False
            elif direction == 'down' and indicators['momentum_pct'] > -0.0001:
                return False
        
        # ADX mínimo (tendência)
        if indicators['adx'] < 20:
            return False
        
        # RSI não pode estar extremo
        if direction == 'up' and indicators['rsi'] > 75:
            return False
        elif direction == 'down' and indicators['rsi'] < 25:
            return False
        
        return True
    
    def _create_breakout_signal(self, signal_type: str, indicators: Dict,
                               market_context: Dict) -> Signal:
        """Cria sinal de breakout"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        # Stop loss baseado em ATR
        sl_distance = atr * self.parameters['atr_multiplier_sl']
        
        if signal_type == 'buy':
            # Stop abaixo do canal inferior ou ATR
            stop_loss = max(
                indicators['lower_channel'] - atr * 0.5,
                price - sl_distance
            )
            
            # Take profit baseado em R:R
            risk = price - stop_loss
            take_profit = price + (risk * self.parameters['risk_reward_ratio'])
            
        else:  # sell
            # Stop acima do canal superior ou ATR
            stop_loss = min(
                indicators['upper_channel'] + atr * 0.5,
                price + sl_distance
            )
            
            # Take profit baseado em R:R
            risk = stop_loss - price
            take_profit = price - (risk * self.parameters['risk_reward_ratio'])
        
        # Calcular confiança
        confidence = self._calculate_breakout_confidence(indicators, signal_type)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Donchian Breakout {signal_type.upper()} - ADX: {indicators['adx']:.1f}",
            metadata={
                'channel_width': indicators['channel_width'],
                'breakout_distance': indicators['breakout_distance'],
                'volume_surge': indicators['volume_surge'],
                'momentum': indicators['momentum_pct'],
                'atr': atr
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    def _calculate_breakout_confidence(self, indicators: Dict, signal_type: str) -> float:
        """Calcula confiança do sinal de breakout"""
        confidence = 0.6  # Base
        
        # Volume surge aumenta confiança
        if indicators['volume_surge'] > 2.0:
            confidence += 0.15
        elif indicators['volume_surge'] > 1.5:
            confidence += 0.1
        
        # ADX forte
        if indicators['adx'] > 30:
            confidence += 0.1
        elif indicators['adx'] > 25:
            confidence += 0.05
        
        # Distância do breakout
        breakout_pct = indicators['breakout_distance'] / indicators['current_price']
        if breakout_pct > 0.001:  # 0.1%
            confidence += 0.1
        
        # Momentum confirmando
        if signal_type == 'buy' and indicators['momentum_pct'] > 0.0005:
            confidence += 0.05
        elif signal_type == 'sell' and indicators['momentum_pct'] < -0.0005:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Calcula condições de saída"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Trailing stop
        if self.parameters['use_trailing_stop']:
            if position.pnl / position.entry_price > self.parameters['trailing_activation']:
                new_stop = self.calculate_trailing_stop(
                    position,
                    current_price,
                    indicators['atr'],
                    1.5  # Multiplicador mais apertado para trailing
                )
                
                if position.side == 'buy' and new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                elif position.side == 'sell' and new_stop < position.stop_loss:
                    position.stop_loss = new_stop
        
        # Saída por canal oposto (se configurado)
        if self.parameters['use_mid_exit']:
            if position.side == 'buy':
                # Sair no canal do meio ou inferior
                if current_price <= indicators['mid_channel']:
                    return ExitSignal(
                        position_id=position.id,
                        reason="Preço atingiu canal médio",
                        exit_price=current_price
                    )
            else:  # sell
                # Sair no canal do meio ou superior
                if current_price >= indicators['mid_channel']:
                    return ExitSignal(
                        position_id=position.id,
                        reason="Preço atingiu canal médio",
                        exit_price=current_price
                    )
        
        # Scale out por níveis de R
        if self.parameters['scale_out_enabled']:
            pnl_r = self._calculate_r_multiple(position, current_price)
            
            for r_level, exit_pct in self.parameters['scale_out_levels']:
                if pnl_r >= r_level and not self._already_scaled_out(position, r_level):
                    return ExitSignal(
                        position_id=position.id,
                        reason=f"Scale out em {r_level}R",
                        exit_price=current_price,
                        partial_exit=exit_pct
                    )
        
        return None
    
    def _calculate_r_multiple(self, position: Position, current_price: float) -> float:
        """Calcula múltiplo de R (risco inicial)"""
        initial_risk = abs(position.entry_price - position.stop_loss)
        
        if position.side == 'buy':
            profit = current_price - position.entry_price
        else:
            profit = position.entry_price - current_price
        
        return profit / initial_risk if initial_risk > 0 else 0
    
    def _already_scaled_out(self, position: Position, r_level: float) -> bool:
        """Verifica se já fez scale out neste nível"""
        if 'scaled_out_levels' not in position.metadata:
            position.metadata['scaled_out_levels'] = []
        
        return r_level in position.metadata['scaled_out_levels']
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int = 14) -> float:
        """Calcula ADX simplificado"""
        if len(high) < period + 1:
            return 0
        
        # True Range
        tr = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr.append(max(hl, hc, lc))
        
        if not tr:
            return 0
        
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        
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
        
        if not plus_dm or atr == 0:
            return 0
        
        # DI+ e DI-
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if len(plus_dm) >= period else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if len(minus_dm) >= period else 0
        
        # DX
        if plus_di + minus_di == 0:
            return 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        return dx