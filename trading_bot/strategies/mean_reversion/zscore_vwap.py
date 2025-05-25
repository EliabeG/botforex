# strategies/mean_reversion/zscore_vwap.py
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from collections import deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("zscore_vwap_strategy")

class ZScoreVWAPStrategy(BaseStrategy):
    """
    Estratégia de mean reversion usando Z-Score do preço em relação ao VWAP.
    
    Conceito:
    - Calcula VWAP (Volume Weighted Average Price) intraday
    - Mede desvio do preço atual vs VWAP em desvios padrão (Z-Score)
    - Entra quando preço está extremamente desviado, esperando reversão
    
    Adequada para mercados laterais com volume consistente.
    """
    
    def __init__(self):
        super().__init__("ZScoreVWAP")
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.LOW_VOLUME]
        self.min_time_between_signals = 180  # 3 minutos
        
        # Buffers para cálculos
        self.price_buffer = deque(maxlen=1000)
        self.volume_buffer = deque(maxlen=1000)
        self.vwap_buffer = deque(maxlen=200)
        self.last_reset_hour = None
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Z-Score thresholds
            'zscore_entry_threshold': 2.5,     # Entrar quando Z > 2.5
            'zscore_exit_threshold': 0.5,      # Sair quando Z < 0.5
            'zscore_stop_threshold': 3.5,      # Stop quando Z > 3.5 (movimento continuou)
            
            # VWAP settings
            'vwap_period_minutes': 60,         # Período para cálculo VWAP
            'min_volume_percentile': 30,       # Volume mínimo (percentil)
            'reset_vwap_hourly': True,         # Resetar VWAP a cada hora
            
            # Risk settings
            'atr_multiplier_sl': 1.5,
            'risk_reward_ratio': 2.0,
            'max_position_duration': 3600,     # Máximo 1 hora
            
            # Filters
            'min_spread_quality': 0.8,         # Qualidade mínima do spread
            'avoid_news_window': 15,           # Minutos antes/depois de news
            'min_liquidity_score': 0.6,        # Score mínimo de liquidez
            'use_rsi_filter': True,            # Filtrar por RSI extremo
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # Position management
            'scale_in_enabled': True,          # Permite aumentar posição
            'scale_in_zscore': 3.0,           # Z-Score para scale-in
            'partial_exit_profit': 0.001,      # 0.1% para saída parcial
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula indicadores para a estratégia"""
        try:
            tick = market_context.get('tick')
            dom = market_context.get('dom')
            recent_ticks = market_context.get('recent_ticks', [])
            
            if not tick or len(recent_ticks) < 100:
                return {}
            
            # Atualizar buffers
            self._update_buffers(tick, dom)
            
            # Reset VWAP se necessário
            self._check_vwap_reset()
            
            # Calcular VWAP
            vwap, vwap_std = self._calculate_vwap()
            if vwap == 0:
                return {}
            
            # Calcular Z-Score
            current_price = tick.mid
            zscore = (current_price - vwap) / vwap_std if vwap_std > 0 else 0
            
            # Calcular indicadores auxiliares
            closes = np.array([t.mid for t in recent_ticks])
            
            # RSI para filtro
            rsi = self.calculate_rsi(closes, period=14)
            
            # ATR para stops
            highs = np.array([t.ask for t in recent_ticks])
            lows = np.array([t.bid for t in recent_ticks])
            atr = self.calculate_atr(highs, lows, closes, period=14)
            
            # Liquidez
            liquidity_score = self._calculate_liquidity_score(dom)
            
            # Qualidade do spread
            spread_quality = self._calculate_spread_quality(tick, recent_ticks)
            
            # Momentum de volume
            volume_momentum = self._calculate_volume_momentum()
            
            indicators = {
                'vwap': vwap,
                'vwap_std': vwap_std,
                'zscore': zscore,
                'current_price': current_price,
                'rsi': rsi,
                'atr': atr,
                'liquidity_score': liquidity_score,
                'spread_quality': spread_quality,
                'volume_momentum': volume_momentum,
                'spread': tick.spread,
                'ticks_since_reset': len(self.price_buffer)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return {}
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de trading baseado em mean reversion"""
        indicators = self.indicators
        
        if not indicators or 'zscore' not in indicators:
            return None
        
        # Verificar filtros
        if not self._check_entry_filters(indicators, market_context):
            return None
        
        zscore = indicators['zscore']
        entry_threshold = self.parameters['zscore_entry_threshold']
        
        # Detectar oportunidade de mean reversion
        signal_type = None
        
        if abs(zscore) >= entry_threshold:
            # Z-Score positivo alto = preço acima do VWAP = vender
            if zscore >= entry_threshold:
                # Confirmar com RSI
                if not self.parameters['use_rsi_filter'] or indicators['rsi'] > self.parameters['rsi_overbought']:
                    signal_type = 'sell'
            
            # Z-Score negativo baixo = preço abaixo do VWAP = comprar
            elif zscore <= -entry_threshold:
                # Confirmar com RSI
                if not self.parameters['use_rsi_filter'] or indicators['rsi'] < self.parameters['rsi_oversold']:
                    signal_type = 'buy'
        
        if signal_type:
            return self._create_mean_reversion_signal(signal_type, indicators, market_context)
        
        return None
    
    def _update_buffers(self, tick, dom):
        """Atualiza buffers de preço e volume"""
        self.price_buffer.append(tick.mid)
        
        # Estimar volume do tick baseado no DOM
        if dom:
            depth = dom.get_depth(5)
            estimated_volume = (depth['bid_volume'] + depth['ask_volume']) / 2
        else:
            estimated_volume = 100000  # Volume padrão
        
        self.volume_buffer.append(estimated_volume)
    
    def _check_vwap_reset(self):
        """Verifica se deve resetar VWAP (novo período)"""
        current_hour = datetime.now().hour
        
        if self.parameters['reset_vwap_hourly']:
            if self.last_reset_hour != current_hour:
                self.price_buffer.clear()
                self.volume_buffer.clear()
                self.vwap_buffer.clear()
                self.last_reset_hour = current_hour
                logger.info(f"VWAP resetado para nova hora: {current_hour}:00")
    
    def _calculate_vwap(self) -> Tuple[float, float]:
        """Calcula VWAP e desvio padrão"""
        if len(self.price_buffer) < 20:  # Mínimo de dados
            return 0, 0
        
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        
        # VWAP = Σ(Price × Volume) / Σ(Volume)
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        # Desvio padrão dos preços em relação ao VWAP
        deviations = prices - vwap
        vwap_std = np.std(deviations)
        
        # Armazenar para análise
        self.vwap_buffer.append(vwap)
        
        return vwap, vwap_std
    
    def _calculate_liquidity_score(self, dom) -> float:
        """Calcula score de liquidez baseado no DOM"""
        if not dom:
            return 0.5
        
        depth = dom.get_depth(10)
        
        # Fatores de liquidez
        total_volume = depth['bid_volume'] + depth['ask_volume']
        
        # Balanceamento bid/ask
        if total_volume > 0:
            balance = min(depth['bid_volume'], depth['ask_volume']) / (total_volume / 2)
        else:
            balance = 0
        
        # Normalizar volume (assumindo 1M como referência)
        volume_score = min(total_volume / 1000000, 1.0)
        
        # Score final
        liquidity_score = (volume_score * 0.7 + balance * 0.3)
        
        return liquidity_score
    
    def _calculate_spread_quality(self, tick, recent_ticks: List) -> float:
        """Calcula qualidade do spread"""
        current_spread = tick.spread
        
        # Spread médio dos últimos ticks
        if len(recent_ticks) >= 20:
            avg_spread = np.mean([t.spread for t in recent_ticks[-20:]])
            
            if avg_spread > 0:
                # Qualidade = 1 - (spread_atual / spread_médio)
                # Clamped entre 0 e 1
                quality = max(0, min(1, 1 - (current_spread / avg_spread - 1)))
                return quality
        
        return 0.5  # Neutro
    
    def _calculate_volume_momentum(self) -> float:
        """Calcula momentum do volume"""
        if len(self.volume_buffer) < 50:
            return 0
        
        recent_volume = np.mean(list(self.volume_buffer)[-10:])
        older_volume = np.mean(list(self.volume_buffer)[-50:-10])
        
        if older_volume > 0:
            momentum = (recent_volume - older_volume) / older_volume
            return np.clip(momentum, -1, 1)
        
        return 0
    
    def _check_entry_filters(self, indicators: Dict, market_context: Dict) -> bool:
        """Verifica filtros antes de entrar"""
        # Spread quality
        if indicators['spread_quality'] < self.parameters['min_spread_quality']:
            return False
        
        # Liquidez
        if indicators['liquidity_score'] < self.parameters['min_liquidity_score']:
            return False
        
        # Mínimo de dados
        if indicators['ticks_since_reset'] < 50:
            return False
        
        # Regime de mercado
        regime = market_context.get('regime')
        if regime not in self.suitable_regimes:
            return False
        
        # Horário
        hour = datetime.now().hour
        if not self.get_time_filter(hour):
            return False
        
        return True
    
    def _create_mean_reversion_signal(self, signal_type: str, indicators: Dict,
                                     market_context: Dict) -> Signal:
        """Cria sinal de mean reversion"""
        price = indicators['current_price']
        atr = indicators['atr']
        vwap = indicators['vwap']
        zscore = indicators['zscore']
        
        # Stop loss baseado em Z-Score threshold
        stop_threshold = self.parameters['zscore_stop_threshold']
        
        if signal_type == 'buy':
            # Para compra, stop se Z-Score ficar ainda mais negativo
            stop_zscore = -stop_threshold
            stop_price = vwap + (stop_zscore * indicators['vwap_std'])
            stop_loss = min(stop_price, price - (atr * self.parameters['atr_multiplier_sl']))
            
            # Take profit quando voltar ao VWAP
            take_profit = vwap + (indicators['vwap_std'] * 0.5)  # Meio desvio acima
            
        else:  # sell
            # Para venda, stop se Z-Score ficar ainda mais positivo
            stop_zscore = stop_threshold
            stop_price = vwap + (stop_zscore * indicators['vwap_std'])
            stop_loss = max(stop_price, price + (atr * self.parameters['atr_multiplier_sl']))
            
            # Take profit quando voltar ao VWAP
            take_profit = vwap - (indicators['vwap_std'] * 0.5)  # Meio desvio abaixo
        
        # Calcular confiança
        confidence = self._calculate_reversion_confidence(indicators, signal_type)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Mean Reversion - Z-Score: {zscore:.2f}",
            metadata={
                'zscore': zscore,
                'vwap': vwap,
                'liquidity_score': indicators['liquidity_score'],
                'rsi': indicators['rsi']
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    def _calculate_reversion_confidence(self, indicators: Dict, signal_type: str) -> float:
        """Calcula confiança na reversão"""
        confidence = 0.5
        
        # Z-Score extremo aumenta confiança
        zscore_abs = abs(indicators['zscore'])
        if zscore_abs > 3.0:
            confidence += 0.2
        elif zscore_abs > 2.5:
            confidence += 0.1
        
        # RSI confirmando
        rsi = indicators['rsi']
        if signal_type == 'buy' and rsi < 25:
            confidence += 0.15
        elif signal_type == 'sell' and rsi > 75:
            confidence += 0.15
        
        # Boa liquidez
        if indicators['liquidity_score'] > 0.8:
            confidence += 0.1
        
        # Spread quality
        if indicators['spread_quality'] > 0.9:
            confidence += 0.05
        
        return min(confidence, 0.9)
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Calcula condições de saída para mean reversion"""
        # Recalcular indicadores atuais
        indicators = self.indicators
        
        if not indicators or 'zscore' not in indicators:
            return None
        
        zscore = indicators['zscore']
        exit_threshold = self.parameters['zscore_exit_threshold']
        
        # Verificar reversão completa
        if position.side == 'buy':
            # Sair se Z-Score voltou próximo de zero ou ficou positivo
            if zscore >= -exit_threshold:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Reversão completa - Z-Score: {zscore:.2f}",
                    exit_price=current_price
                )
        
        else:  # sell
            # Sair se Z-Score voltou próximo de zero ou ficou negativo
            if zscore <= exit_threshold:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Reversão completa - Z-Score: {zscore:.2f}",
                    exit_price=current_price
                )
        
        # Verificar duração máxima
        position_age = (datetime.now() - position.open_time).total_seconds()
        if position_age > self.parameters['max_position_duration']:
            return ExitSignal(
                position_id=position.id,
                reason="Tempo máximo de posição",
                exit_price=current_price
            )
        
        # Saída parcial por lucro
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        if position.side == 'buy' and pnl_pct > self.parameters['partial_exit_profit']:
            return ExitSignal(
                position_id=position.id,
                reason="Realização parcial de lucro",
                exit_price=current_price,
                partial_exit=0.5
            )
        elif position.side == 'sell' and pnl_pct < -self.parameters['partial_exit_profit']:
            return ExitSignal(
                position_id=position.id,
                reason="Realização parcial de lucro",
                exit_price=current_price,
                partial_exit=0.5
            )
        
        return None