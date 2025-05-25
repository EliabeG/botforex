# strategies/mean_reversion/bollinger_fade.py
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime
from collections import deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("bollinger_fade")

class BollingerFadeStrategy(BaseStrategy):
    """
    Estratégia de fade (reversão) usando Bollinger Bands
    
    Conceito:
    - Entradas quando preço toca/ultrapassa bandas
    - Confirmação com %B e largura das bandas
    - Saídas na média móvel ou banda oposta
    """
    
    def __init__(self):
        super().__init__("BollingerBandFade")
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY]
        self.min_time_between_signals = 180
        
        # Histórico para análise
        self.band_touches = deque(maxlen=100)
        self.squeeze_history = deque(maxlen=50)
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'bb_std_dev_extreme': 2.5,
            'min_band_width': 0.0008,  # 8 pips
            'max_band_width': 0.0030,  # 30 pips
            'percent_b_oversold': 0.0,
            'percent_b_overbought': 1.0,
            'use_squeeze': True,
            'squeeze_threshold': 0.0010,
            'consecutive_touches': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'atr_multiplier_sl': 1.2,
            'atr_multiplier_tp': 1.8,
            'use_middle_exit': True
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula Bollinger Bands e indicadores auxiliares"""
        try:
            ticks = market_context.get('recent_ticks', [])
            if len(ticks) < self.parameters['bb_period'] + 1:
                return {}
            
            # Preços
            closes = np.array([t.mid for t in ticks])
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(closes)
            if not bb_data:
                return {}
            
            # RSI
            rsi = self._calculate_rsi(closes, self.parameters['rsi_period'])
            
            # ATR
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Band Width (largura das bandas)
            band_width = bb_data['upper'] - bb_data['lower']
            band_width_pct = (band_width / bb_data['middle']) * 100 if bb_data['middle'] > 0 else 0
            
            # %B (posição do preço nas bandas)
            percent_b = 0
            if band_width > 0:
                percent_b = (closes[-1] - bb_data['lower']) / band_width
            
            # Detectar squeeze (compressão de volatilidade)
            is_squeeze = band_width < self.parameters['squeeze_threshold']
            self.squeeze_history.append(is_squeeze)
            
            # Análise de toques nas bandas
            current_touch = self._analyze_band_touch(closes[-1], bb_data)
            if current_touch:
                self.band_touches.append({
                    'type': current_touch,
                    'price': closes[-1],
                    'timestamp': datetime.now()
                })
            
            # Calcular consecutivos
            consecutive_upper = self._count_consecutive_touches('upper')
            consecutive_lower = self._count_consecutive_touches('lower')
            
            # Velocidade de aproximação das bandas
            if len(closes) > 5:
                price_momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            else:
                price_momentum = 0
            
            indicators = {
                'upper_band': bb_data['upper'],
                'middle_band': bb_data['middle'],
                'lower_band': bb_data['lower'],
                'upper_extreme': bb_data['upper_extreme'],
                'lower_extreme': bb_data['lower_extreme'],
                'band_width': band_width,
                'band_width_pct': band_width_pct,
                'percent_b': percent_b,
                'is_squeeze': is_squeeze,
                'squeeze_count': sum(self.squeeze_history),
                'rsi': rsi,
                'atr': atr,
                'current_price': closes[-1],
                'price_momentum': price_momentum,
                'consecutive_upper': consecutive_upper,
                'consecutive_lower': consecutive_lower,
                'current_touch': current_touch,
                'spread': market_context.get('spread', 0)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular Bollinger Bands: {e}")
            return {}
    
    def _calculate_bollinger_bands(self, closes: np.ndarray) -> Optional[Dict]:
        """Calcula Bollinger Bands com múltiplos desvios"""
        period = self.parameters['bb_period']
        if len(closes) < period:
            return None
        
        # Média móvel simples
        sma = np.mean(closes[-period:])
        
        # Desvio padrão
        std = np.std(closes[-period:])
        
        # Bandas padrão
        upper = sma + (std * self.parameters['bb_std_dev'])
        lower = sma - (std * self.parameters['bb_std_dev'])
        
        # Bandas extremas
        upper_extreme = sma + (std * self.parameters['bb_std_dev_extreme'])
        lower_extreme = sma - (std * self.parameters['bb_std_dev_extreme'])
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'upper_extreme': upper_extreme,
            'lower_extreme': lower_extreme,
            'std': std
        }
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calcula RSI"""
        if len(closes) < period + 1:
            return 50  # Neutro
        
        # Diferenças de preço
        deltas = np.diff(closes[-period-1:])
        
        # Ganhos e perdas
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Médias
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _analyze_band_touch(self, price: float, bb_data: Dict) -> Optional[str]:
        """Analisa se o preço tocou ou ultrapassou bandas"""
        tolerance = 0.00002  # 0.2 pips
        
        # Toque na banda superior
        if price >= bb_data['upper'] - tolerance:
            if price >= bb_data['upper_extreme']:
                return 'upper_extreme'
            return 'upper'
        
        # Toque na banda inferior
        elif price <= bb_data['lower'] + tolerance:
            if price <= bb_data['lower_extreme']:
                return 'lower_extreme'
            return 'lower'
        
        return None
    
    def _count_consecutive_touches(self, band_type: str) -> int:
        """Conta toques consecutivos em uma banda"""
        if not self.band_touches:
            return 0
        
        count = 0
        for touch in reversed(self.band_touches):
            if band_type in touch['type']:
                count += 1
            else:
                break
        
        return count
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinais de fade nas bandas"""
        indicators = self.indicators
        
        if not indicators or 'upper_band' not in indicators:
            return None
        
        # Verificar largura das bandas
        if indicators['band_width'] < self.parameters['min_band_width']:
            return None  # Bandas muito estreitas
        
        if indicators['band_width'] > self.parameters['max_band_width']:
            return None  # Bandas muito largas (alta volatilidade)
        
        signal_type = None
        signal_strength = 0
        
        # VENDA - Toque na banda superior
        if indicators['current_touch'] and 'upper' in indicators['current_touch']:
            signal_strength = 1
            
            # Confirmações
            if indicators['percent_b'] >= self.parameters['percent_b_overbought']:
                signal_strength += 0.5
            
            if indicators['rsi'] > self.parameters['rsi_overbought']:
                signal_strength += 0.5
            
            if indicators['consecutive_upper'] >= self.parameters['consecutive_touches']:
                signal_strength += 0.5
            
            if 'extreme' in indicators['current_touch']:
                signal_strength += 1
            
            if indicators['price_momentum'] > 0.1:  # Momentum forte
                signal_strength += 0.5
            
            if signal_strength >= 2:
                signal_type = 'sell'
        
        # COMPRA - Toque na banda inferior
        elif indicators['current_touch'] and 'lower' in indicators['current_touch']:
            signal_strength = 1
            
            # Confirmações
            if indicators['percent_b'] <= self.parameters['percent_b_oversold']:
                signal_strength += 0.5
            
            if indicators['rsi'] < self.parameters['rsi_oversold']:
                signal_strength += 0.5
            
            if indicators['consecutive_lower'] >= self.parameters['consecutive_touches']:
                signal_strength += 0.5
            
            if 'extreme' in indicators['current_touch']:
                signal_strength += 1
            
            if indicators['price_momentum'] < -0.1:
                signal_strength += 0.5
            
            if signal_strength >= 2:
                signal_type = 'buy'
        
        # Estratégia de squeeze breakout (opcional)
        elif self.parameters['use_squeeze'] and indicators['squeeze_count'] > 10:
            # Após período de squeeze, preparar para breakout
            if indicators['percent_b'] > 1.0:
                signal_type = 'buy'  # Breakout para cima
                signal_strength = 2
            elif indicators['percent_b'] < 0.0:
                signal_type = 'sell'  # Breakout para baixo
                signal_strength = 2
        
        if signal_type:
            return self._create_bollinger_signal(signal_type, indicators, signal_strength, market_context)
        
        return None
    
    def _create_bollinger_signal(self, signal_type: str, indicators: Dict,
                                signal_strength: float, market_context: Dict) -> Signal:
        """Cria sinal com gestão de risco baseada em Bollinger"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        # Stops e alvos
        if signal_type == 'buy':
            # Stop abaixo da banda inferior extrema
            stop_loss = min(
                indicators['lower_extreme'] - (atr * 0.3),
                price - (atr * self.parameters['atr_multiplier_sl'])
            )
            
            # Alvo na média ou banda superior
            if self.parameters['use_middle_exit']:
                take_profit = indicators['middle_band']
            else:
                take_profit = indicators['upper_band'] - (atr * 0.2)
        
        else:  # sell
            stop_loss = max(
                indicators['upper_extreme'] + (atr * 0.3),
                price + (atr * self.parameters['atr_multiplier_sl'])
            )
            
            if self.parameters['use_middle_exit']:
                take_profit = indicators['middle_band']
            else:
                take_profit = indicators['lower_band'] + (atr * 0.2)
        
        # Confiança
        confidence = self._calculate_bollinger_confidence(indicators, signal_strength)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Bollinger Fade {signal_type.upper()} - %B: {indicators['percent_b']:.2f}",
            metadata={
                'percent_b': indicators['percent_b'],
                'band_width': indicators['band_width'],
                'rsi': indicators['rsi'],
                'touch_type': indicators['current_touch'],
                'consecutive_touches': indicators[f'consecutive_{indicators["current_touch"].split("_")[0]}'],
                'signal_strength': signal_strength
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    def _calculate_bollinger_confidence(self, indicators: Dict, signal_strength: float) -> float:
        """Calcula confiança do sinal"""
        confidence = 0.6
        
        # Força do sinal
        if signal_strength >= 3:
            confidence += 0.15
        elif signal_strength >= 2.5:
            confidence += 0.1
        
        # RSI confirmando
        if (indicators['rsi'] > 70 and indicators['current_touch'] and 'upper' in indicators['current_touch']) or \
           (indicators['rsi'] < 30 and indicators['current_touch'] and 'lower' in indicators['current_touch']):
            confidence += 0.1
        
        # Largura ideal das bandas
        if 0.0012 <= indicators['band_width'] <= 0.0020:
            confidence += 0.05
        
        # Toques consecutivos
        touches = indicators.get(f'consecutive_{indicators.get("current_touch", "").split("_")[0]}', 0)
        if touches >= 3:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída para Bollinger fade"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída na média móvel
        if self.parameters['use_middle_exit']:
            if position.side == 'buy' and current_price >= indicators['middle_band']:
                return ExitSignal(
                    position_id=position.id,
                    reason="Reached middle band",
                    exit_price=current_price
                )
            
            elif position.side == 'sell' and current_price <= indicators['middle_band']:
                return ExitSignal(
                    position_id=position.id,
                    reason="Reached middle band",
                    exit_price=current_price
                )
        
        # Saída se tocar banda oposta
        if position.side == 'buy':
            if indicators['current_touch'] and 'upper' in indicators['current_touch']:
                return ExitSignal(
                    position_id=position.id,
                    reason="Touched upper band",
                    exit_price=current_price
                )
        
        else:  # sell
            if indicators['current_touch'] and 'lower' in indicators['current_touch']:
                return ExitSignal(
                    position_id=position.id,
                    reason="Touched lower band",
                    exit_price=current_price
                )
        
        # Saída se %B normalizar
        if position.side == 'buy' and indicators['percent_b'] >= 0.8:
            return ExitSignal(
                position_id=position.id,
                reason="Percent B normalized",
                exit_price=current_price
            )
        
        elif position.side == 'sell' and indicators['percent_b'] <= 0.2:
            return ExitSignal(
                position_id=position.id,
                reason="Percent B normalized",
                exit_price=current_price
            )
        
        return None