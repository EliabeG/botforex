# strategies/momentum/ichimoku_kumo.py
import numpy as np
from typing import Dict, Optional, Any, Tuple
from datetime import datetime

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("ichimoku_kumo")

class IchimokuKumoStrategy(BaseStrategy):
    """
    Estratégia usando Ichimoku Cloud (Kumo)
    
    Sinais:
    - Compra: Preço acima da nuvem + Tenkan > Kijun + Chikou Span confirmando
    - Venda: Preço abaixo da nuvem + Tenkan < Kijun + Chikou Span confirmando
    """
    
    def __init__(self):
        super().__init__("IchimokuKumo")
        self.suitable_regimes = [MarketRegime.TREND]
        self.min_time_between_signals = 600  # 10 minutos
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26,
            'chikou_period': 26,
            'min_kumo_thickness': 0.0005,  # 5 pips
            'use_kumo_breakout': True,
            'use_tk_cross': True,
            'use_chikou_confirmation': True,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'kumo_twist_filter': True
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula componentes do Ichimoku"""
        try:
            ticks = market_context.get('recent_ticks', [])
            min_required = max(
                self.parameters['senkou_b_period'],
                self.parameters['kijun_period'] + self.parameters['displacement']
            ) + 1
            
            if len(ticks) < min_required:
                return {}
            
            # Extrair dados
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            closes = np.array([t.mid for t in ticks])
            
            # Calcular linhas Ichimoku
            tenkan = self._calculate_midpoint(highs, lows, self.parameters['tenkan_period'])
            kijun = self._calculate_midpoint(highs, lows, self.parameters['kijun_period'])
            
            # Senkou Span A (média de Tenkan e Kijun, deslocada)
            senkou_a = (tenkan + kijun) / 2
            
            # Senkou Span B (midpoint de período maior, deslocada)
            senkou_b = self._calculate_midpoint(highs, lows, self.parameters['senkou_b_period'])
            
            # Chikou Span (preço deslocado para trás)
            chikou_span = closes[-1]
            chikou_position = closes[-self.parameters['chikou_period']] if len(closes) > self.parameters['chikou_period'] else closes[0]
            
            # Nuvem (Kumo)
            kumo_top = max(senkou_a, senkou_b)
            kumo_bottom = min(senkou_a, senkou_b)
            kumo_thickness = kumo_top - kumo_bottom
            
            # Direção da nuvem
            kumo_bullish = senkou_a > senkou_b
            
            # Detectar twist (mudança de direção) da nuvem
            kumo_twist = False
            if len(ticks) > self.parameters['displacement'] + 5:
                future_senkou_a = self._calculate_future_senkou_a(highs, lows)
                future_senkou_b = self._calculate_future_senkou_b(highs, lows)
                current_bullish = senkou_a > senkou_b
                future_bullish = future_senkou_a > future_senkou_b
                kumo_twist = current_bullish != future_bullish
            
            # ATR para stops
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Posição do preço em relação à nuvem
            current_price = closes[-1]
            price_vs_kumo = 'above' if current_price > kumo_top else 'below' if current_price < kumo_bottom else 'inside'
            
            indicators = {
                'tenkan': tenkan,
                'kijun': kijun,
                'senkou_a': senkou_a,
                'senkou_b': senkou_b,
                'kumo_top': kumo_top,
                'kumo_bottom': kumo_bottom,
                'kumo_thickness': kumo_thickness,
                'kumo_bullish': kumo_bullish,
                'kumo_twist': kumo_twist,
                'chikou_span': chikou_span,
                'chikou_position': chikou_position,
                'current_price': current_price,
                'price_vs_kumo': price_vs_kumo,
                'tk_cross': tenkan - kijun,
                'atr': atr,
                'spread': market_context.get('spread', 0)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular Ichimoku: {e}")
            return {}
    
    def _calculate_midpoint(self, highs: np.ndarray, lows: np.ndarray, period: int) -> float:
        """Calcula ponto médio (máxima + mínima) / 2"""
        if len(highs) < period:
            return 0
        
        highest = np.max(highs[-period:])
        lowest = np.min(lows[-period:])
        
        return (highest + lowest) / 2
    
    def _calculate_future_senkou_a(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calcula Senkou Span A futuro para detectar twist"""
        displacement = self.parameters['displacement']
        if len(highs) < displacement:
            return 0
        
        # Usar dados mais recentes para projetar
        future_tenkan = self._calculate_midpoint(
            highs[-displacement:],
            lows[-displacement:],
            min(self.parameters['tenkan_period'], displacement)
        )
        future_kijun = self._calculate_midpoint(
            highs[-displacement:],
            lows[-displacement:],
            min(self.parameters['kijun_period'], displacement)
        )
        
        return (future_tenkan + future_kijun) / 2
    
    def _calculate_future_senkou_b(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calcula Senkou Span B futuro"""
        displacement = self.parameters['displacement']
        if len(highs) < displacement:
            return 0
        
        return self._calculate_midpoint(
            highs[-displacement:],
            lows[-displacement:],
            min(self.parameters['senkou_b_period'], displacement)
        )
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinais baseados em Ichimoku"""
        indicators = self.indicators
        
        if not indicators or 'tenkan' not in indicators:
            return None
        
        # Verificar espessura mínima da nuvem
        if indicators['kumo_thickness'] < self.parameters['min_kumo_thickness']:
            return None
        
        # Verificar twist da nuvem
        if self.parameters['kumo_twist_filter'] and indicators['kumo_twist']:
            return None  # Evitar entradas durante mudança de direção da nuvem
        
        signal_type = None
        signal_strength = 0
        
        # Sinais de COMPRA
        if indicators['price_vs_kumo'] == 'above':
            signal_strength += 1
            
            # TK Cross bullish
            if self.parameters['use_tk_cross'] and indicators['tk_cross'] > 0:
                signal_strength += 1
            
            # Chikou Span confirmação
            if self.parameters['use_chikou_confirmation']:
                if indicators['chikou_span'] > indicators['chikou_position']:
                    signal_strength += 1
            
            # Nuvem bullish
            if indicators['kumo_bullish']:
                signal_strength += 0.5
            
            if signal_strength >= 2.5:
                signal_type = 'buy'
        
        # Sinais de VENDA
        elif indicators['price_vs_kumo'] == 'below':
            signal_strength += 1
            
            # TK Cross bearish
            if self.parameters['use_tk_cross'] and indicators['tk_cross'] < 0:
                signal_strength += 1
            
            # Chikou Span confirmação
            if self.parameters['use_chikou_confirmation']:
                if indicators['chikou_span'] < indicators['chikou_position']:
                    signal_strength += 1
            
            # Nuvem bearish
            if not indicators['kumo_bullish']:
                signal_strength += 0.5
            
            if signal_strength >= 2.5:
                signal_type = 'sell'
        
        # Breakout da nuvem
        elif self.parameters['use_kumo_breakout'] and indicators['price_vs_kumo'] == 'inside':
            # Verificar se está saindo da nuvem
            # (simplificado - em produção, comparar com ticks anteriores)
            pass
        
        if signal_type:
            return self._create_ichimoku_signal(signal_type, indicators, signal_strength, market_context)
        
        return None
    
    def _create_ichimoku_signal(self, signal_type: str, indicators: Dict, 
                               signal_strength: float, market_context: Dict) -> Signal:
        """Cria sinal com stops baseados em Ichimoku"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        # Stops baseados na nuvem e ATR
        if signal_type == 'buy':
            # Stop abaixo da nuvem ou Kijun
            stop_loss = min(
                indicators['kumo_bottom'] - (atr * 0.5),
                indicators['kijun'] - (atr * 0.5)
            )
            take_profit = price + (atr * self.parameters['atr_multiplier_tp'])
        
        else:  # sell
            # Stop acima da nuvem ou Kijun
            stop_loss = max(
                indicators['kumo_top'] + (atr * 0.5),
                indicators['kijun'] + (atr * 0.5)
            )
            take_profit = price - (atr * self.parameters['atr_multiplier_tp'])
        
        # Confiança baseada na força do sinal
        confidence = min(0.5 + (signal_strength * 0.15), 0.95)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Ichimoku {signal_type.upper()} - Strength: {signal_strength:.1f}",
            metadata={
                'signal_strength': signal_strength,
                'price_vs_kumo': indicators['price_vs_kumo'],
                'kumo_thickness': indicators['kumo_thickness'],
                'tk_cross': indicators['tk_cross'],
                'kumo_bullish': indicators['kumo_bullish']
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída baseadas em Ichimoku"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída se preço cruzar para o lado oposto da nuvem
        if position.side == 'buy':
            if indicators['price_vs_kumo'] == 'below':
                return ExitSignal(
                    position_id=position.id,
                    reason="Preço cruzou abaixo da nuvem",
                    exit_price=current_price
                )
            
            # Saída se TK Cross virar bearish
            if indicators['tk_cross'] < -0.0002:  # Margem pequena
                return ExitSignal(
                    position_id=position.id,
                    reason="TK Cross bearish",
                    exit_price=current_price
                )
        
        else:  # sell
            if indicators['price_vs_kumo'] == 'above':
                return ExitSignal(
                    position_id=position.id,
                    reason="Preço cruzou acima da nuvem",
                    exit_price=current_price
                )
            
            # Saída se TK Cross virar bullish
            if indicators['tk_cross'] > 0.0002:
                return ExitSignal(
                    position_id=position.id,
                    reason="TK Cross bullish",
                    exit_price=current_price
                )
        
        # Trailing stop usando Kijun
        if position.side == 'buy':
            kijun_stop = indicators['kijun'] - (indicators['atr'] * 0.5)