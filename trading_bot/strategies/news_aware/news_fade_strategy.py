# strategies/news_aware/news_fade_strategy.py
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import aiohttp
import asyncio

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("news_fade_strategy")

class NewsFadeStrategy(BaseStrategy):
    """
    Estratégia que opera fade de spikes de notícias
    
    Conceito:
    - Detecta spikes causados por notícias
    - Entra contra o movimento exagerado
    - Sai quando volatilidade normaliza
    """
    
    def __init__(self):
        super().__init__("NewsFade")
        self.suitable_regimes = [MarketRegime.HIGH_VOLATILITY]
        self.min_time_between_signals = 1800  # 30 minutos
        
        # Cache de notícias
        self.news_cache = []
        self.last_news_check = None
        self.upcoming_events = []
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # News settings
            'check_calendar': True,
            'high_impact_only': True,
            'pre_news_minutes': 30,
            'post_news_minutes': 60,
            'min_spike_pips': 15,
            
            # Entry criteria
            'spike_velocity_threshold': 20,  # pips/segundo
            'min_volatility_expansion': 3.0,  # vs normal
            'fade_delay_seconds': 30,         # Aguardar antes de fade
            
            # Risk
            'atr_multiplier_sl': 3.0,         # Stop mais largo
            'fixed_tp_pips': 20,
            'max_position_time': 1800,        # 30 minutos
            
            # Filters
            'avoid_major_pairs': ['USDJPY', 'GBPUSD'],  # Correlação
            'min_spread_quality': 0.5,        # Aceitar spread pior
            'require_retracement': True,
            'retracement_pct': 0.3           # 30% de retração
        }
    
    async def initialize(self):
        """Inicializa estratégia e carrega calendário"""
        await super().initialize()
        
        # Carregar eventos do dia
        await self._load_economic_calendar()
    
    async def _load_economic_calendar(self):
        """Carrega calendário econômico (simulado)"""
        # Em produção, usar API real (ForexFactory, Investing.com, etc)
        
        # Eventos simulados
        self.upcoming_events = [
            {
                'time': datetime.now() + timedelta(hours=2),
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'impact': 'high',
                'forecast': '200K',
                'previous': '180K'
            },
            {
                'time': datetime.now() + timedelta(hours=4),
                'currency': 'EUR',
                'event': 'ECB Rate Decision',
                'impact': 'high',
                'forecast': '4.50%',
                'previous': '4.50%'
            }
        ]
        
        logger.info(f"Calendário carregado: {len(self.upcoming_events)} eventos")
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula indicadores para detecção de news spikes"""
        try:
            ticks = market_context.get('recent_ticks', [])
            
            if len(ticks) < 100:
                return {}
            
            # Preços e volatilidade
            prices = np.array([t.mid for t in ticks])
            spreads = np.array([t.spread for t in ticks])
            
            # Detectar spike
            spike_detected, spike_info = self._detect_spike(prices, ticks)
            
            # Verificar proximidade de notícias
            near_news, news_info = await self._check_news_proximity()
            
            # Calcular volatilidade atual vs normal
            current_volatility = np.std(prices[-20:]) * np.sqrt(252 * 24 * 60 * 60)
            normal_volatility = np.std(prices[-100:-20]) * np.sqrt(252 * 24 * 60 * 60)
            volatility_ratio = current_volatility / normal_volatility if normal_volatility > 0 else 1
            
            # ATR
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            closes = prices
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Detectar retração após spike
            retracement = 0
            if spike_detected and spike_info:
                retracement = self._calculate_retracement(prices, spike_info)
            
            indicators = {
                'spike_detected': spike_detected,
                'spike_info': spike_info,
                'near_news': near_news,
                'news_info': news_info,
                'current_volatility': current_volatility,
                'normal_volatility': normal_volatility,
                'volatility_ratio': volatility_ratio,
                'atr': atr,
                'retracement': retracement,
                'current_price': prices[-1],
                'spread': spreads[-1],
                'spread_widening': spreads[-1] / np.mean(spreads[-20:-1]) if len(spreads) > 20 else 1
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores de notícias: {e}")
            return {}
    
    def _detect_spike(self, prices: np.ndarray, ticks: List) -> Tuple[bool, Optional[Dict]]:
        """Detecta spike de preço"""
        if len(prices) < 30:
            return False, None
        
        # Calcular velocidade do movimento
        for i in range(5, 20):  # Janelas de 5 a 20 ticks
            price_change = abs(prices[-1] - prices[-i])
            time_diff = (ticks[-1].timestamp - ticks[-i].timestamp).total_seconds()
            
            if time_diff > 0:
                velocity = (price_change * 10000) / time_diff  # pips/segundo
                
                if velocity >= self.parameters['spike_velocity_threshold']:
                    # Spike detectado
                    direction = 'up' if prices[-1] > prices[-i] else 'down'
                    
                    spike_info = {
                        'start_price': prices[-i],
                        'peak_price': prices[-1],
                        'direction': direction,
                        'velocity': velocity,
                        'magnitude_pips': price_change * 10000,
                        'duration_seconds': time_diff,
                        'start_index': -i
                    }
                    
                    # Verificar magnitude mínima
                    if spike_info['magnitude_pips'] >= self.parameters['min_spike_pips']:
                        return True, spike_info
        
        return False, None
    
    async def _check_news_proximity(self) -> Tuple[bool, Optional[Dict]]:
        """Verifica proximidade de notícias"""
        current_time = datetime.now()
        
        for event in self.upcoming_events:
            time_to_event = (event['time'] - current_time).total_seconds() / 60  # minutos
            
            # Verificar se está na janela de notícia
            if -self.parameters['post_news_minutes'] <= time_to_event <= self.parameters['pre_news_minutes']:
                
                # Filtrar por impacto
                if self.parameters['high_impact_only'] and event['impact'] != 'high':
                    continue
                
                # Verificar se afeta EURUSD
                if event['currency'] in ['EUR', 'USD']:
                    return True, {
                        'event': event['event'],
                        'currency': event['currency'],
                        'impact': event['impact'],
                        'time_to_event': time_to_event,
                        'is_post_news': time_to_event < 0
                    }
        
        return False, None
    
    def _calculate_retracement(self, prices: np.ndarray, spike_info: Dict) -> float:
        """Calcula retração após spike"""
        if not spike_info:
            return 0
        
        start_idx = spike_info['start_index']
        peak_price = spike_info['peak_price']
        start_price = spike_info['start_price']
        current_price = prices[-1]
        
        spike_range = abs(peak_price - start_price)
        
        if spike_range == 0:
            return 0
        
        if spike_info['direction'] == 'up':
            # Para spike de alta, retração é quando preço cai do pico
            retracement_amount = peak_price - current_price
        else:
            # Para spike de baixa, retração é quando preço sobe do fundo
            retracement_amount = current_price - peak_price
        
        retracement_pct = retracement_amount / spike_range
        
        return max(0, min(1, retracement_pct))  # Limitar entre 0 e 1
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de fade de notícia"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Verificar condições
        if not indicators['spike_detected']:
            return None
        
        if not indicators['near_news']:
            return None  # Spike sem notícia não é nosso alvo
        
        spike_info = indicators['spike_info']
        news_info = indicators['news_info']
        
        # Verificar expansão de volatilidade
        if indicators['volatility_ratio'] < self.parameters['min_volatility_expansion']:
            return None
        
        # Aguardar delay configurado
        if spike_info['duration_seconds'] < self.parameters['fade_delay_seconds']:
            return None
        
        # Verificar retração se necessário
        if self.parameters['require_retracement']:
            if indicators['retracement'] < self.parameters['retracement_pct']:
                return None  # Ainda não retraiu o suficiente
        
        # Determinar direção do fade (oposta ao spike)
        if spike_info['direction'] == 'up':
            signal_type = 'sell'  # Fade de alta
        else:
            signal_type = 'buy'   # Fade de baixa
        
        # Verificar se é pós-notícia (mais confiável)
        if news_info['is_post_news']:
            # Aumentar confiança para fades pós-notícia
            confidence_boost = 0.1
        else:
            confidence_boost = 0
        
        return self._create_news_fade_signal(
            signal_type, 
            indicators, 
            confidence_boost,
            market_context
        )
    
    def _create_news_fade_signal(self, signal_type: str, indicators: Dict,
                                confidence_boost: float, market_context: Dict) -> Signal:
        """Cria sinal de fade de notícia"""
        price = indicators['current_price']
        atr = indicators['atr']
        spike_info = indicators['spike_info']
        news_info = indicators['news_info']
        
        # Stops mais largos devido à volatilidade
        sl_distance = atr * self.parameters['atr_multiplier_sl']
        
        if signal_type == 'buy':
            # Fade de spike baixista
            stop_loss = price - sl_distance
            
            # TP conservador
            take_profit = price + (self.parameters['fixed_tp_pips'] / 10000)
            
        else:  # sell
            # Fade de spike altista
            stop_loss = price + sl_distance
            
            # TP conservador
            take_profit = price - (self.parameters['fixed_tp_pips'] / 10000)
        
        # Calcular confiança
        confidence = self._calculate_news_fade_confidence(
            indicators,
            signal_type,
            confidence_boost
        )
        
        # Criar reason detalhado
        reason = f"News Fade {signal_type.upper()} - "
        reason += f"{news_info['event']} ({news_info['currency']}) - "
        reason += f"Spike: {spike_info['magnitude_pips']:.1f} pips @ {spike_info['velocity']:.1f} pips/s"
        
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
                'spike_info': spike_info,
                'news_info': news_info,
                'volatility_ratio': indicators['volatility_ratio'],
                'retracement': indicators['retracement'],
                'spread_widening': indicators['spread_widening']
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    def _calculate_news_fade_confidence(self, indicators: Dict, 
                                      signal_type: str,
                                      confidence_boost: float) -> float:
        """Calcula confiança do fade"""
        confidence = 0.6  # Base
        
        spike_info = indicators['spike_info']
        
        # Magnitude do spike
        if spike_info['magnitude_pips'] > 30:
            confidence += 0.15
        elif spike_info['magnitude_pips'] > 20:
            confidence += 0.1
        
        # Velocidade (spikes muito rápidos são melhores para fade)
        if spike_info['velocity'] > 30:
            confidence += 0.1
        
        # Retração
        if indicators['retracement'] > 0.5:
            confidence += 0.1
        elif indicators['retracement'] > 0.3:
            confidence += 0.05
        
        # Volatilidade expandida
        if indicators['volatility_ratio'] > 4:
            confidence += 0.05
        
        # Boost de confiança (pós-notícia)
        confidence += confidence_boost
        
        # Penalizar spread muito largo
        if indicators['spread_widening'] > 3:
            confidence *= 0.8
        
        return min(confidence, 0.9)  # Máximo 90% para esta estratégia arriscada
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída para fade de notícia"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída se volatilidade normalizar
        if indicators['volatility_ratio'] < 1.5:
            return ExitSignal(
                position_id=position.id,
                reason="Volatility normalized",
                exit_price=current_price
            )
        
        # Saída por tempo máximo
        position_age = (datetime.now() - position.open_time).total_seconds()
        if position_age > self.parameters['max_position_time']:
            return ExitSignal(
                position_id=position.id,
                reason="Max time reached for news trade",
                exit_price=current_price
            )
        
        # Saída se novo spike na mesma direção
        if indicators['spike_detected']:
            new_spike = indicators['spike_info']
            
            if position.side == 'buy' and new_spike['direction'] == 'down':
                return ExitSignal(
                    position_id=position.id,
                    reason="New spike in same direction",
                    exit_price=current_price
                )
            elif position.side == 'sell' and new_spike['direction'] == 'up':
                return ExitSignal(
                    position_id=position.id,
                    reason="New spike in same direction",
                    exit_price=current_price
                )
        
        # Trailing stop mais agressivo para capturar lucros
        if position.pnl > 0:
            # Mover stop para breakeven + 2 pips
            if position.side == 'buy':
                new_stop = position.entry_price + 0.0002
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                new_stop = position.entry_price - 0.0002
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
        
        return None