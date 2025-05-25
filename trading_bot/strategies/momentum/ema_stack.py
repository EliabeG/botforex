# strategies/momentum/ema_stack.py
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("ema_stack_strategy")

class EMAStackStrategy(BaseStrategy):
    """
    Estratégia de momentum usando stack de médias móveis exponenciais.
    
    Sinais de entrada:
    - BUY: EMA8 > EMA21 > EMA50 e preço acima de todas
    - SELL: EMA8 < EMA21 < EMA50 e preço abaixo de todas
    
    Adequada para mercados em tendência forte.
    """
    
    def __init__(self):
        super().__init__("EMAStack_8_21_50")
        self.suitable_regimes = [MarketRegime.TREND]
        self.min_time_between_signals = 300  # 5 minutos
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'ema_fast': 8,
            'ema_medium': 21,
            'ema_slow': 50,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'min_trend_strength': 0.0002,  # Inclinação mínima
            'use_trailing_stop': True,
            'trailing_atr_multiplier': 1.5,
            'min_volume_ratio': 0.8,  # Volume mínimo vs média
            'max_spread_pips': 1.0,
            'filter_news_time': True,
            'news_minutes_before': 30,
            'news_minutes_after': 30
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula indicadores necessários"""
        try:
            ticks = market_context.get('recent_ticks', [])
            if len(ticks) < self.parameters['ema_slow'] + 1:
                return {}
            
            # Extrair preços
            closes = np.array([t.mid for t in ticks])
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            
            # Calcular EMAs
            ema_fast = self.calculate_ema(closes, self.parameters['ema_fast'])
            ema_medium = self.calculate_ema(closes, self.parameters['ema_medium'])
            ema_slow = self.calculate_ema(closes, self.parameters['ema_slow'])
            
            # Calcular inclinação das EMAs (tendência)
            ema_fast_slope = (ema_fast - self.calculate_ema(closes[:-1], self.parameters['ema_fast'])) if len(closes) > 1 else 0
            ema_medium_slope = (ema_medium - self.calculate_ema(closes[:-1], self.parameters['ema_medium'])) if len(closes) > 1 else 0
            ema_slow_slope = (ema_slow - self.calculate_ema(closes[:-1], self.parameters['ema_slow'])) if len(closes) > 1 else 0
            
            # ATR para stops
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Volume
            dom = market_context.get('dom')
            volume_ratio = 1.0
            if dom:
                current_volume = dom.get_depth(5)
                total_volume = current_volume['bid_volume'] + current_volume['ask_volume']
                # Comparar com média (simplificado)
                volume_ratio = total_volume / 100000  # Normalizado
            
            # RSI para filtro adicional
            rsi = self.calculate_rsi(closes, period=14)
            
            # ADX para força da tendência
            adx = self._calculate_adx(highs, lows, closes, period=14)
            
            indicators = {
                'ema_fast': ema_fast,
                'ema_medium': ema_medium,
                'ema_slow': ema_slow,
                'ema_fast_slope': ema_fast_slope,
                'ema_medium_slope': ema_medium_slope,
                'ema_slow_slope': ema_slow_slope,
                'current_price': closes[-1],
                'atr': atr,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'adx': adx,
                'spread': market_context.get('spread', 0)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return {}
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de trading"""
        indicators = self.indicators
        
        if not indicators or 'ema_fast' not in indicators:
            return None
        
        # Verificar filtros básicos
        if not self._check_basic_filters(indicators, market_context):
            return None
        
        # Detectar setup
        signal_type = self._detect_setup(indicators)
        
        if signal_type:
            return self._create_signal(signal_type, indicators, market_context)
        
        return None
    
    def _check_basic_filters(self, indicators: Dict, market_context: Dict) -> bool:
        """Verifica filtros básicos antes de gerar sinal"""
        # Spread máximo
        if indicators['spread'] > self.parameters['max_spread_pips'] / 10000:
            return False
        
        # Volume mínimo
        if indicators['volume_ratio'] < self.parameters['min_volume_ratio']:
            return False
        
        # Filtro de horário
        hour = datetime.now().hour
        if not self.get_time_filter(hour):
            return False
        
        # ADX mínimo (tendência)
        if indicators.get('adx', 0) < 25:
            return False
        
        # Regime de mercado
        if market_context.get('regime') != MarketRegime.TREND:
            return False
        
        return True
    
    def _detect_setup(self, indicators: Dict) -> Optional[str]:
        """Detecta setup de entrada"""
        ema_fast = indicators['ema_fast']
        ema_medium = indicators['ema_medium']
        ema_slow = indicators['ema_slow']
        price = indicators['current_price']
        
        # Verificar alinhamento das EMAs
        emas_bullish = ema_fast > ema_medium > ema_slow
        emas_bearish = ema_fast < ema_medium < ema_slow
        
        # Verificar força da tendência (inclinação)
        min_slope = self.parameters['min_trend_strength']
        
        trend_bullish = (
            indicators['ema_fast_slope'] > min_slope and
            indicators['ema_medium_slope'] > min_slope * 0.8 and
            indicators['ema_slow_slope'] > min_slope * 0.6
        )
        
        trend_bearish = (
            indicators['ema_fast_slope'] < -min_slope and
            indicators['ema_medium_slope'] < -min_slope * 0.8 and
            indicators['ema_slow_slope'] < -min_slope * 0.6
        )
        
        # Setup LONG
        if emas_bullish and trend_bullish and price > ema_fast:
            # RSI não pode estar sobrecomprado
            if indicators['rsi'] < 70:
                return 'buy'
        
        # Setup SHORT
        elif emas_bearish and trend_bearish and price < ema_fast:
            # RSI não pode estar sobrevendido
            if indicators['rsi'] > 30:
                return 'sell'
        
        return None
    
    def _create_signal(self, signal_type: str, indicators: Dict, 
                      market_context: Dict) -> Signal:
        """Cria sinal de trading com stops calculados"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        # Calcular stops
        sl_distance = atr * self.parameters['atr_multiplier_sl']
        tp_distance = atr * self.parameters['atr_multiplier_tp']
        
        if signal_type == 'buy':
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:  # sell
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
        
        # Calcular confiança do sinal
        confidence = self._calculate_signal_confidence(indicators, signal_type)
        
        # Criar sinal
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"EMA Stack {signal_type.upper()} - ADX: {indicators['adx']:.1f}",
            metadata={
                'atr': atr,
                'rsi': indicators['rsi'],
                'adx': indicators['adx'],
                'ema_slopes': {
                    'fast': indicators['ema_fast_slope'],
                    'medium': indicators['ema_medium_slope'],
                    'slow': indicators['ema_slow_slope']
                }
            }
        )
        
        # Ajustar para spread
        signal = self.adjust_for_spread(signal, indicators['spread'])
        
        return signal
    
    def _calculate_signal_confidence(self, indicators: Dict, signal_type: str) -> float:
        """Calcula confiança do sinal baseado em múltiplos fatores"""
        confidence = 0.5  # Base
        
        # ADX (força da tendência)
        adx = indicators.get('adx', 0)
        if adx > 40:
            confidence += 0.2
        elif adx > 30:
            confidence += 0.1
        
        # Alinhamento de inclinações
        slopes = [
            abs(indicators['ema_fast_slope']),
            abs(indicators['ema_medium_slope']),
            abs(indicators['ema_slow_slope'])
        ]
        
        if all(s > self.parameters['min_trend_strength'] for s in slopes):
            confidence += 0.15
        
        # RSI
        rsi = indicators['rsi']
        if signal_type == 'buy' and 40 < rsi < 60:
            confidence += 0.1  # RSI neutro em tendência de alta
        elif signal_type == 'sell' and 40 < rsi < 60:
            confidence += 0.1  # RSI neutro em tendência de baixa
        
        # Volume
        if indicators['volume_ratio'] > 1.2:
            confidence += 0.05
        
        return min(confidence, 0.95)  # Máximo 95%
    
    async def calculate_exit_conditions(self, position: Position, 
                                       current_price: float) -> Optional[ExitSignal]:
        """Calcula condições de saída"""
        # Recalcular indicadores
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Trailing stop
        if self.parameters['use_trailing_stop'] and position.trailing_stop:
            new_stop = self.calculate_trailing_stop(
                position,
                current_price,
                indicators['atr'],
                self.parameters['trailing_atr_multiplier']
            )
            
            # Atualizar stop se moveu favoravelmente
            if position.side == 'buy' and new_stop > position.stop_loss:
                position.stop_loss = new_stop
            elif position.side == 'sell' and new_stop < position.stop_loss:
                position.stop_loss = new_stop
        
        # Verificar quebra de estrutura (EMAs)
        ema_fast = indicators['ema_fast']
        ema_medium = indicators['ema_medium']
        
        # Saída por quebra de estrutura
        if position.side == 'buy':
            # Sair se preço quebrou abaixo da EMA rápida
            if current_price < ema_fast * 0.999:  # 0.1% de tolerância
                return ExitSignal(
                    position_id=position.id,
                    reason="Quebra abaixo da EMA rápida",
                    exit_price=current_price
                )
            
            # Sair se EMAs desalinharam
            if ema_fast < ema_medium:
                return ExitSignal(
                    position_id=position.id,
                    reason="EMAs desalinhadas",
                    exit_price=current_price
                )
        
        else:  # sell
            # Sair se preço quebrou acima da EMA rápida
            if current_price > ema_fast * 1.001:
                return ExitSignal(
                    position_id=position.id,
                    reason="Quebra acima da EMA rápida",
                    exit_price=current_price
                )
            
            # Sair se EMAs desalinharam
            if ema_fast > ema_medium:
                return ExitSignal(
                    position_id=position.id,
                    reason="EMAs desalinhadas",
                    exit_price=current_price
                )
        
        # Saída parcial por lucro
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        if position.side == 'buy' and pnl_pct > 0.002:  # 0.2% lucro
            # Realizar 50% do lucro
            return ExitSignal(
                position_id=position.id,
                reason="Realização parcial de lucro",
                exit_price=current_price,
                partial_exit=0.5
            )
        elif position.side == 'sell' and pnl_pct < -0.002:
            return ExitSignal(
                position_id=position.id,
                reason="Realização parcial de lucro",
                exit_price=current_price,
                partial_exit=0.5
            )
        
        return None
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int = 14) -> float:
        """Calcula ADX (Average Directional Index)"""
        if len(high) < period + 1:
            return 0
        
        # Cálculo simplificado do ADX
        # Em produção, usar talib.ADX ou implementação completa
        
        # True Range
        tr = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr.append(max(hl, hc, lc))
        
        if not tr:
            return 0
        
        # ATR
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
        
        # Índices direcionais
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if len(plus_dm) >= period else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if len(minus_dm) >= period else 0
        
        # ADX
        if plus_di + minus_di == 0:
            return 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        return dx  # Simplificado - normalmente seria a média do DX