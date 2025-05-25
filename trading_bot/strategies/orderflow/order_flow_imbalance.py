# strategies/orderflow/order_flow_imbalance.py
import numpy as np
from typing import Dict, Optional, Any, List, Deque
from datetime import datetime, timedelta
from collections import deque, defaultdict

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("order_flow_imbalance_strategy")

class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Estratégia baseada em desequilíbrio de fluxo de ordens
    
    Conceito:
    - Analisa fluxo de ordens agressivas (market orders)
    - Detecta desequilíbrios significativos entre compra/venda
    - Entra na direção do fluxo dominante
    """
    
    def __init__(self):
        super().__init__("OrderFlowImbalance")
        self.suitable_regimes = [MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY]
        self.min_time_between_signals = 120  # 2 minutos
        
        # Buffers para análise de fluxo
        self.trade_flow_buffer = deque(maxlen=1000)
        self.dom_snapshots = deque(maxlen=100)
        self.volume_profile = defaultdict(lambda: {'buy': 0, 'sell': 0})
        self.delta_cumulative = 0
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Flow parameters
            'imbalance_threshold': 0.65,      # 65% de desequilíbrio
            'min_volume_threshold': 500000,   # Volume mínimo para considerar
            'lookback_seconds': 60,           # Janela de análise
            'delta_threshold': 1000000,       # Delta cumulativo mínimo
            
            # DOM analysis
            'dom_levels': 10,                 # Níveis do DOM para analisar
            'absorption_ratio': 3.0,          # Ratio para detectar absorção
            'sweep_velocity': 100,            # Ticks/segundo para sweep
            
            # Confirmations
            'use_vwap_filter': True,
            'vwap_deviation': 0.0005,         # 5 pips do VWAP
            'momentum_confirmation': True,
            'min_tick_volume': 50,            # Ticks mínimos na janela
            
            # Risk
            'atr_multiplier_sl': 1.5,
            'fixed_tp_pips': 10,
            'use_dom_stops': True,            # Usar níveis do DOM para stops
            
            # Filters
            'min_spread': 0.5,                # Pips
            'max_spread': 1.5,                # Pips
            'avoid_news': True,
            'min_liquidity_score': 0.7
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula métricas de order flow"""
        try:
            tick = market_context.get('tick')
            dom = market_context.get('dom')
            
            if not tick or not dom:
                return {}
            
            # Atualizar buffers
            self._update_flow_buffers(tick, dom)
            
            # Calcular time window
            cutoff_time = datetime.now() - timedelta(seconds=self.parameters['lookback_seconds'])
            
            # Filtrar trades recentes
            recent_trades = [t for t in self.trade_flow_buffer if t['timestamp'] > cutoff_time]
            
            if len(recent_trades) < self.parameters['min_tick_volume']:
                return {}
            
            # Calcular volume direcional
            buy_volume = sum(t['volume'] for t in recent_trades if t['side'] == 'buy')
            sell_volume = sum(t['volume'] for t in recent_trades if t['side'] == 'sell')
            total_volume = buy_volume + sell_volume
            
            # Order Flow Imbalance
            if total_volume > 0:
                ofi = (buy_volume - sell_volume) / total_volume
                buy_ratio = buy_volume / total_volume
                sell_ratio = sell_volume / total_volume
            else:
                ofi = 0
                buy_ratio = 0.5
                sell_ratio = 0.5
            
            # Delta cumulativo
            delta = buy_volume - sell_volume
            self.delta_cumulative += delta
            
            # DOM analysis
            dom_metrics = self._analyze_dom(dom)
            
            # VWAP
            vwap = self._calculate_vwap(recent_trades)
            
            # Detectar eventos especiais
            absorption = self._detect_absorption(recent_trades, dom_metrics)
            sweep = self._detect_sweep(recent_trades)
            
            # Volume profile analysis
            poc = self._find_point_of_control()
            
            # Momentum do fluxo
            flow_momentum = self._calculate_flow_momentum(recent_trades)
            
            indicators = {
                'ofi': ofi,
                'buy_ratio': buy_ratio,
                'sell_ratio': sell_ratio,
                'total_volume': total_volume,
                'delta': delta,
                'delta_cumulative': self.delta_cumulative,
                'vwap': vwap,
                'current_price': tick.mid,
                'spread': tick.spread,
                'poc': poc,
                'flow_momentum': flow_momentum,
                'absorption_detected': absorption,
                'sweep_detected': sweep,
                'trade_count': len(recent_trades),
                **dom_metrics
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return {}
    
    def _update_flow_buffers(self, tick, dom):
        """Atualiza buffers de fluxo de ordens"""
        # Estimar side baseado no preço vs bid/ask
        if tick.mid >= tick.ask:
            side = 'buy'
        elif tick.mid <= tick.bid:
            side = 'sell'
        else:
            # Usar tick rule
            if len(self.trade_flow_buffer) > 0:
                last_price = self.trade_flow_buffer[-1]['price']
                side = 'buy' if tick.mid > last_price else 'sell'
            else:
                side = 'neutral'
        
        # Estimar volume (simplificado - em produção usar dados reais)
        volume = (tick.bid_volume + tick.ask_volume) / 2
        
        trade = {
            'timestamp': datetime.now(),
            'price': tick.mid,
            'side': side,
            'volume': volume,
            'bid': tick.bid,
            'ask': tick.ask
        }
        
        self.trade_flow_buffer.append(trade)
        
        # Atualizar volume profile
        price_level = round(tick.mid, 5)  # Arredondar para nível
        self.volume_profile[price_level][side] += volume
        
        # Snapshot do DOM
        self.dom_snapshots.append({
            'timestamp': datetime.now(),
            'snapshot': dom
        })
    
    def _analyze_dom(self, dom) -> Dict:
        """Analisa profundidade do mercado"""
        depth = dom.get_depth(self.parameters['dom_levels'])
        
        # Calcular métricas
        bid_volume = depth['bid_volume']
        ask_volume = depth['ask_volume']
        total_depth = bid_volume + ask_volume
        
        # Imbalance do DOM
        if total_depth > 0:
            dom_imbalance = (bid_volume - ask_volume) / total_depth
        else:
            dom_imbalance = 0
        
        # Liquidez por nível
        bid_levels = depth['bids']
        ask_levels = depth['asks']
        
        # Detectar walls (ordens grandes)
        bid_wall = max(bid_levels, key=lambda x: x[1]) if bid_levels else (0, 0)
        ask_wall = max(ask_levels, key=lambda x: x[1]) if ask_levels else (0, 0)
        
        # Calcular pressão de compra/venda
        if len(bid_levels) > 3 and len(ask_levels) > 3:
            near_bid_volume = sum(vol for _, vol in bid_levels[:3])
            near_ask_volume = sum(vol for _, vol in ask_levels[:3])
            near_pressure = near_bid_volume / (near_bid_volume + near_ask_volume) if (near_bid_volume + near_ask_volume) > 0 else 0.5
        else:
            near_pressure = 0.5
        
        return {
            'dom_imbalance': dom_imbalance,
            'total_depth': total_depth,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_wall_price': bid_wall[0],
            'bid_wall_size': bid_wall[1],
            'ask_wall_price': ask_wall[0],
            'ask_wall_size': ask_wall[1],
            'near_pressure': near_pressure
        }
    
    def _detect_absorption(self, trades: List[Dict], dom_metrics: Dict) -> bool:
        """Detecta absorção de ordens"""
        if len(trades) < 20:
            return False
        
        # Verificar se preço não se moveu apesar do volume
        price_range = max(t['price'] for t in trades[-20:]) - min(t['price'] for t in trades[-20:])
        volume_recent = sum(t['volume'] for t in trades[-20:])
        
        # Absorção = muito volume sem movimento de preço
        if price_range < 0.0002 and volume_recent > self.parameters['min_volume_threshold']:
            # Verificar se tem wall no DOM
            if dom_metrics['bid_wall_size'] > volume_recent * 0.3 or dom_metrics['ask_wall_size'] > volume_recent * 0.3:
                return True
        
        return False
    
    def _detect_sweep(self, trades: List[Dict]) -> bool:
        """Detecta sweep de liquidez"""
        if len(trades) < 10:
            return False
        
        # Calcular velocidade do movimento
        time_span = (trades[-1]['timestamp'] - trades[-10]['timestamp']).total_seconds()
        if time_span <= 0:
            return False
        
        price_change = abs(trades[-1]['price'] - trades[-10]['price'])
        velocity = (price_change * 10000) / time_span  # pips por segundo
        
        # Sweep = movimento rápido com volume
        volume_burst = sum(t['volume'] for t in trades[-10:])
        
        return velocity > self.parameters['sweep_velocity'] and volume_burst > self.parameters['min_volume_threshold']
    
    def _calculate_vwap(self, trades: List[Dict]) -> float:
        """Calcula VWAP dos trades recentes"""
        if not trades:
            return 0
        
        total_volume = sum(t['volume'] for t in trades)
        if total_volume == 0:
            return trades[-1]['price']
        
        vwap = sum(t['price'] * t['volume'] for t in trades) / total_volume
        return vwap
    
    def _find_point_of_control(self) -> float:
        """Encontra Point of Control (preço com maior volume)"""
        if not self.volume_profile:
            return 0
        
        poc_price = 0
        max_volume = 0
        
        for price, volumes in self.volume_profile.items():
            total_vol = volumes['buy'] + volumes['sell']
            if total_vol > max_volume:
                max_volume = total_vol
                poc_price = price
        
        return poc_price
    
    def _calculate_flow_momentum(self, trades: List[Dict]) -> float:
        """Calcula momentum do fluxo de ordens"""
        if len(trades) < 20:
            return 0
        
        # Dividir em duas metades
        mid_point = len(trades) // 2
        first_half = trades[:mid_point]
        second_half = trades[mid_point:]
        
        # Calcular OFI para cada metade
        def calc_ofi(trade_list):
            buy_vol = sum(t['volume'] for t in trade_list if t['side'] == 'buy')
            sell_vol = sum(t['volume'] for t in trade_list if t['side'] == 'sell')
            total = buy_vol + sell_vol
            return (buy_vol - sell_vol) / total if total > 0 else 0
        
        ofi_first = calc_ofi(first_half)
        ofi_second = calc_ofi(second_half)
        
        # Momentum = mudança no OFI
        return ofi_second - ofi_first
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal baseado em order flow"""
        indicators = self.indicators
        
        if not indicators or 'ofi' not in indicators:
            return None
        
        # Verificar volume mínimo
        if indicators['total_volume'] < self.parameters['min_volume_threshold']:
            return None
        
        # Verificar spread
        if indicators['spread'] < self.parameters['min_spread'] / 10000:
            return None
        if indicators['spread'] > self.parameters['max_spread'] / 10000:
            return None
        
        signal_type = None
        signal_strength = 0
        
        # Sinal de COMPRA
        if indicators['buy_ratio'] >= self.parameters['imbalance_threshold']:
            signal_strength = indicators['ofi']
            
            # Confirmações
            if self.parameters['momentum_confirmation'] and indicators['flow_momentum'] > 0.1:
                signal_strength += 0.2
            
            if indicators['dom_imbalance'] > 0.3:
                signal_strength += 0.1
            
            if indicators['delta_cumulative'] > self.parameters['delta_threshold']:
                signal_strength += 0.2
            
            # Eventos especiais
            if indicators['sweep_detected']:
                signal_strength += 0.3
            
            if indicators['absorption_detected'] and indicators['near_pressure'] > 0.6:
                signal_strength += 0.2
            
            # Filtro VWAP
            if self.parameters['use_vwap_filter']:
                vwap_distance = indicators['current_price'] - indicators['vwap']
                if abs(vwap_distance) > self.parameters['vwap_deviation']:
                    signal_strength *= 0.5  # Reduzir se muito longe do VWAP
            
            if signal_strength >= 0.6:
                signal_type = 'buy'
        
        # Sinal de VENDA
        elif indicators['sell_ratio'] >= self.parameters['imbalance_threshold']:
            signal_strength = abs(indicators['ofi'])
            
            # Confirmações
            if self.parameters['momentum_confirmation'] and indicators['flow_momentum'] < -0.1:
                signal_strength += 0.2
            
            if indicators['dom_imbalance'] < -0.3:
                signal_strength += 0.1
            
            if indicators['delta_cumulative'] < -self.parameters['delta_threshold']:
                signal_strength += 0.2
            
            # Eventos especiais
            if indicators['sweep_detected']:
                signal_strength += 0.3
            
            if indicators['absorption_detected'] and indicators['near_pressure'] < 0.4:
                signal_strength += 0.2
            
            # Filtro VWAP
            if self.parameters['use_vwap_filter']:
                vwap_distance = indicators['current_price'] - indicators['vwap']
                if abs(vwap_distance) > self.parameters['vwap_deviation']:
                    signal_strength *= 0.5
            
            if signal_strength >= 0.6:
                signal_type = 'sell'
        
        if signal_type:
            return self._create_flow_signal(signal_type, indicators, signal_strength, market_context)
        
        return None
    
    def _create_flow_signal(self, signal_type: str, indicators: Dict,
                           signal_strength: float, market_context: Dict) -> Signal:
        """Cria sinal com stops baseados em order flow"""
        price = indicators['current_price']
        
        # Calcular stops
        if self.parameters['use_dom_stops']:
            # Usar walls do DOM como referência
            if signal_type == 'buy':
                # Stop abaixo do bid wall
                stop_loss = min(
                    indicators['bid_wall_price'] - 0.0002,
                    price - (indicators.get('atr', 0.0010) * self.parameters['atr_multiplier_sl'])
                )
                take_profit = price + (self.parameters['fixed_tp_pips'] / 10000)
            else:
                # Stop acima do ask wall
                stop_loss = max(
                    indicators['ask_wall_price'] + 0.0002,
                    price + (indicators.get('atr', 0.0010) * self.parameters['atr_multiplier_sl'])
                )
                take_profit = price - (self.parameters['fixed_tp_pips'] / 10000)
        else:
            # Stops fixos
            if signal_type == 'buy':
                stop_loss = price - (indicators.get('atr', 0.0010) * self.parameters['atr_multiplier_sl'])
                take_profit = price + (self.parameters['fixed_tp_pips'] / 10000)
            else:
                stop_loss = price + (indicators.get('atr', 0.0010) * self.parameters['atr_multiplier_sl'])
                take_profit = price - (self.parameters['fixed_tp_pips'] / 10000)
        
        # Calcular confiança
        confidence = min(0.5 + (signal_strength * 0.3), 0.95)
        
        # Reason string
        reason = f"Order Flow {signal_type.upper()} - OFI: {indicators['ofi']:.2f}"
        if indicators['sweep_detected']:
            reason += " [SWEEP]"
        if indicators['absorption_detected']:
            reason += " [ABSORPTION]"
        
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
                'ofi': indicators['ofi'],
                'delta': indicators['delta'],
                'total_volume': indicators['total_volume'],
                'dom_imbalance': indicators['dom_imbalance'],
                'signal_strength': signal_strength,
                'sweep': indicators['sweep_detected'],
                'absorption': indicators['absorption_detected']
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída baseadas em order flow"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída se flow reverter
        if position.side == 'buy':
            # Sair se fluxo virar vendedor
            if indicators['sell_ratio'] >= self.parameters['imbalance_threshold']:
                return ExitSignal(
                    position_id=position.id,
                    reason="Order flow reversal - Selling pressure",
                    exit_price=current_price
                )
            
            # Sair se delta cumulativo virar negativo
            if indicators['delta_cumulative'] < -self.parameters['delta_threshold'] * 0.5:
                return ExitSignal(
                    position_id=position.id,
                    reason="Cumulative delta turned negative",
                    exit_price=current_price
                )
        
        else:  # sell
            # Sair se fluxo virar comprador
            if indicators['buy_ratio'] >= self.parameters['imbalance_threshold']:
                return ExitSignal(
                    position_id=position.id,
                    reason="Order flow reversal - Buying pressure",
                    exit_price=current_price
                )
            
            # Sair se delta cumulativo virar positivo
            if indicators['delta_cumulative'] > self.parameters['delta_threshold'] * 0.5:
                return ExitSignal(
                    position_id=position.id,
                    reason="Cumulative delta turned positive",
                    exit_price=current_price
                )
        
        # Saída se detectar absorção contra a posição
        if indicators['absorption_detected']:
            if (position.side == 'buy' and indicators['near_pressure'] < 0.4) or \
               (position.side == 'sell' and indicators['near_pressure'] > 0.6):
                return ExitSignal(
                    position_id=position.id,
                    reason="Absorption detected against position",
                    exit_price=current_price
                )
        
        return None