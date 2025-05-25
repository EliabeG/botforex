# strategies/liquidity_hunt/stop_hunt_strategy.py
import numpy as np
from typing import Dict, Optional, Any, List, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("stop_hunt_strategy")

class StopHuntStrategy(BaseStrategy):
    """
    Estratégia que detecta e opera stop hunts/liquidity sweeps
    
    Conceitos:
    - Identifica níveis com concentração de stops
    - Detecta sweeps rápidos desses níveis
    - Entra na reversão após o sweep
    """
    
    def __init__(self):
        super().__init__("LiquidityHunt")
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.TREND]
        self.min_time_between_signals = 300  # 5 minutos
        
        # Tracking de níveis
        self.liquidity_levels = defaultdict(float)
        self.recent_highs = deque(maxlen=100)
        self.recent_lows = deque(maxlen=100)
        self.sweep_events = deque(maxlen=20)
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Level detection
            'lookback_periods': 50,
            'cluster_threshold': 0.0003,      # 3 pips para cluster
            'min_touches': 3,                 # Toques mínimos no nível
            'level_strength_threshold': 0.7,
            
            # Sweep detection
            'sweep_velocity': 15,             # pips/segundo
            'sweep_penetration': 0.0002,      # 2 pips além do nível
            'max_sweep_duration': 10,         # segundos
            'reversal_confirmation': 3,       # ticks de confirmação
            
            # DOM analysis
            'dom_imbalance_threshold': 2.0,  # Ratio para stop hunt
            'min_liquidity_size': 1000000,    # $1M
            'absorption_detection': True,
            
            # Risk
            'atr_multiplier_sl': 1.5,
            'risk_reward_ratio': 2.5,
            'use_liquidity_stops': True,
            
            # Filters
            'round_number_buffer': 0.0005,    # 5 pips de números redondos
            'session_filter': True,
            'avoid_news_window': 15           # minutos
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula níveis de liquidez e detecta sweeps"""
        try:
            ticks = market_context.get('recent_ticks', [])
            dom = market_context.get('dom')
            
            if len(ticks) < self.parameters['lookback_periods']:
                return {}
            
            # Atualizar highs/lows recentes
            for tick in ticks[-10:]:
                self.recent_highs.append(tick.ask)
                self.recent_lows.append(tick.bid)
            
            # Identificar níveis de liquidez
            liquidity_levels = self._identify_liquidity_levels(ticks)
            
            # Detectar sweep atual
            sweep_detected, sweep_info = self._detect_sweep(ticks, liquidity_levels)
            
            # Analisar DOM para stops ocultos
            hidden_liquidity = self._analyze_hidden_liquidity(dom) if dom else {}
            
            # Calcular métricas
            prices = np.array([t.mid for t in ticks])
            current_price = prices[-1]
            
            # ATR
            highs = np.array([t.ask for t in ticks])
            lows = np.array([t.bid for t in ticks])
            atr = self.calculate_atr(highs, lows, prices, period=14)
            
            # Nearest levels
            nearest_resistance = self._find_nearest_level(current_price, liquidity_levels, 'above')
            nearest_support = self._find_nearest_level(current_price, liquidity_levels, 'below')
            
            indicators = {
                'liquidity_levels': liquidity_levels,
                'sweep_detected': sweep_detected,
                'sweep_info': sweep_info,
                'hidden_liquidity': hidden_liquidity,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'current_price': current_price,
                'atr': atr,
                'spread': ticks[-1].spread,
                'recent_sweeps': len(self.sweep_events),
                'session': self._get_current_session()
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores de liquidity hunt: {e}")
            return {}
    
    def _identify_liquidity_levels(self, ticks: List) -> Dict[float, Dict]:
        """Identifica níveis com provável concentração de stops"""
        levels = {}
        
        # Analisar swing highs/lows
        prices = [t.mid for t in ticks]
        
        for i in range(2, len(prices) - 2):
            # Swing high
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                self._add_level(levels, prices[i], 'resistance')
            
            # Swing low
            elif prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
                 prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                self._add_level(levels, prices[i], 'support')
        
        # Adicionar níveis redondos
        current_price = prices[-1]
        round_levels = self._find_round_numbers(current_price)
        
        for level in round_levels:
            self._add_level(levels, level, 'round')
        
        # Filtrar níveis por força
        strong_levels = {}
        for price, info in levels.items():
            if info['touches'] >= self.parameters['min_touches'] or \
               info['type'] == 'round' or \
               info['strength'] >= self.parameters['level_strength_threshold']:
                strong_levels[price] = info
        
        return strong_levels
    
    def _add_level(self, levels: Dict, price: float, level_type: str):
        """Adiciona ou atualiza nível de liquidez"""
        # Procurar nível próximo existente
        for level_price, info in levels.items():
            if abs(level_price - price) <= self.parameters['cluster_threshold']:
                # Atualizar nível existente
                info['touches'] += 1
                info['strength'] = min(1.0, info['strength'] + 0.1)
                info['last_touch'] = datetime.now()
                return
        
        # Criar novo nível
        levels[price] = {
            'type': level_type,
            'touches': 1,
            'strength': 0.5,
            'created': datetime.now(),
            'last_touch': datetime.now()
        }
    
    def _find_round_numbers(self, current_price: float) -> List[float]:
        """Encontra números redondos próximos"""
        round_numbers = []
        
        # Níveis principais (00, 50)
        base = int(current_price * 100) / 100
        
        for offset in [-1, 0, 1]:
            # .00
            level = base + offset
            if abs(level - current_price) <= 0.0050:  # 50 pips
                round_numbers.append(level)
            
            # .50
            level = base + offset + 0.005
            if abs(level - current_price) <= 0.0050:
                round_numbers.append(level)
        
        return round_numbers
    
    def _detect_sweep(self, ticks: List, levels: Dict) -> tuple:
        """Detecta sweep de liquidez"""
        if len(ticks) < 20:
            return False, None
        
        prices = [t.mid for t in ticks[-20:]]
        timestamps = [t.timestamp for t in ticks[-20:]]
        
        # Verificar cada nível
        for level_price, level_info in levels.items():
            # Verificar se preço cruzou o nível recentemente
            crossed = False
            cross_index = -1
            
            for i in range(1, len(prices)):
                if level_info['type'] in ['resistance', 'round'] and \
                   prices[i-1] < level_price and prices[i] > level_price:
                    crossed = True
                    cross_index = i
                    break
                elif level_info['type'] in ['support', 'round'] and \
                     prices[i-1] > level_price and prices[i] < level_price:
                    crossed = True
                    cross_index = i
                    break
            
            if crossed and cross_index > 0:
                # Calcular velocidade do movimento
                price_change = abs(prices[cross_index] - prices[0])
                time_diff = (timestamps[cross_index] - timestamps[0]).total_seconds()
                
                if time_diff > 0:
                    velocity = (price_change * 10000) / time_diff
                    
                    # Verificar se é sweep
                    if velocity >= self.parameters['sweep_velocity'] and \
                       time_diff <= self.parameters['max_sweep_duration']:
                        
                        # Verificar penetração
                        if level_info['type'] == 'resistance':
                            penetration = max(prices[cross_index:]) - level_price
                        else:
                            penetration = level_price - min(prices[cross_index:])
                        
                        if penetration >= self.parameters['sweep_penetration']:
                            # Verificar reversão
                            if self._check_reversal(prices[cross_index:], level_price, level_info['type']):
                                sweep_info = {
                                    'level': level_price,
                                    'level_type': level_info['type'],
                                    'velocity': velocity,
                                    'penetration': penetration,
                                    'timestamp': timestamps[cross_index],
                                    'direction': 'up' if level_info['type'] == 'resistance' else 'down'
                                }
                                
                                # Adicionar ao histórico
                                self.sweep_events.append(sweep_info)
                                
                                return True, sweep_info
        
        return False, None
    
    def _check_reversal(self, prices: List[float], level: float, 
                       level_type: str) -> bool:
        """Verifica se houve reversão após sweep"""
        if len(prices) < self.parameters['reversal_confirmation']:
            return False
        
        if level_type == 'resistance':
            # Para sweep de resistência, preço deve voltar abaixo
            return prices[-1] < level
        else:
            # Para sweep de suporte, preço deve voltar acima
            return prices[-1] > level
    
    def _analyze_hidden_liquidity(self, dom) -> Dict:
        """Analisa DOM para liquidez oculta"""
        if not dom:
            return {}
        
        depth = dom.get_depth(10)
        
        # Procurar desequilíbrios grandes (possíveis stops)
        bid_levels = depth['bids']
        ask_levels = depth['asks']
        
        hidden_stops = {
            'buy_stops': [],
            'sell_stops': []
        }
        
        # Analisar gaps no book
        for i in range(1, len(bid_levels)):
            price_gap = bid_levels[i-1][0] - bid_levels[i][0]
            
            if price_gap > 0.0002:  # Gap > 2 pips
                # Possível área de buy stops
                hidden_stops['buy_stops'].append({
                    'price': (bid_levels[i-1][0] + bid_levels[i][0]) / 2,
                    'gap_size': price_gap
                })
        
        for i in range(1, len(ask_levels)):
            price_gap = ask_levels[i][0] - ask_levels[i-1][0]
            
            if price_gap > 0.0002:  # Gap > 2 pips
                # Possível área de sell stops
                hidden_stops['sell_stops'].append({
                    'price': (ask_levels[i-1][0] + ask_levels[i][0]) / 2,
                    'gap_size': price_gap
                })
        
        return hidden_stops
    
    def _find_nearest_level(self, current_price: float, levels: Dict, 
                           direction: str) -> Optional[Dict]:
        """Encontra nível mais próximo"""
        nearest = None
        min_distance = float('inf')
        
        for level_price, level_info in levels.items():
            distance = abs(level_price - current_price)
            
            if direction == 'above' and level_price > current_price:
                if distance < min_distance:
                    min_distance = distance
                    nearest = {'price': level_price, 'info': level_info}
            
            elif direction == 'below' and level_price < current_price:
                if distance < min_distance:
                    min_distance = distance
                    nearest = {'price': level_price, 'info': level_info}
        
        return nearest
    
    def _get_current_session(self) -> str:
        """Retorna sessão atual"""
        hour = datetime.now().hour
        
        if 7 <= hour < 16:
            return 'london'
        elif 13 <= hour < 22:
            return 'newyork'
        elif 23 <= hour or hour < 8:
            return 'asia'
        else:
            return 'transition'
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de stop hunt"""
        indicators = self.indicators
        
        if not indicators or not indicators['sweep_detected']:
            return None
        
        sweep_info = indicators['sweep_info']
        
        # Filtros
        if self.parameters['session_filter']:
            if indicators['session'] not in ['london', 'newyork']:
                return None  # Apenas sessões principais
        
        # Determinar direção (reversa ao sweep)
        if sweep_info['direction'] == 'up':
            # Sweep de alta = vender
            signal_type = 'sell'
        else:
            # Sweep de baixa = comprar
            signal_type = 'buy'
        
        # Verificar liquidez oculta confirmando
        hidden = indicators.get('hidden_liquidity', {})
        
        confirmation = False
        if signal_type == 'buy' and hidden.get('sell_stops'):
            confirmation = True  # Sell stops foram ativados
        elif signal_type == 'sell' and hidden.get('buy_stops'):
            confirmation = True  # Buy stops foram ativados
        
        return self._create_hunt_signal(
            signal_type,
            indicators,
            sweep_info,
            confirmation,
            market_context
        )
    
    def _create_hunt_signal(self, signal_type: str, indicators: Dict,
                           sweep_info: Dict, confirmation: bool,
                           market_context: Dict) -> Signal:
        """Cria sinal de stop hunt"""
        price = indicators['current_price']
        atr = indicators['atr']
        
        # Stops baseados em níveis de liquidez
        if self.parameters['use_liquidity_stops']:
            if signal_type == 'buy':
                # Stop abaixo do nível swept
                stop_loss = sweep_info['level'] - (atr * 0.5)
                
                # TP no próximo nível de resistência
                if indicators['nearest_resistance']:
                    take_profit = indicators['nearest_resistance']['price'] - 0.0001
                else:
                    take_profit = price + (atr * self.parameters['risk_reward_ratio'])
            
            else:  # sell
                # Stop acima do nível swept
                stop_loss = sweep_info['level'] + (atr * 0.5)
                
                # TP no próximo nível de suporte
                if indicators['nearest_support']:
                    take_profit = indicators['nearest_support']['price'] + 0.0001
                else:
                    take_profit = price - (atr * self.parameters['risk_reward_ratio'])
        else:
            # Stops padrão
            if signal_type == 'buy':
                stop_loss = price - (atr * self.parameters['atr_multiplier_sl'])
                take_profit = price + (atr * self.parameters['atr_multiplier_sl'] * self.parameters['risk_reward_ratio'])
            else:
                stop_loss = price + (atr * self.parameters['atr_multiplier_sl'])
                take_profit = price - (atr * self.parameters['atr_multiplier_sl'] * self.parameters['risk_reward_ratio'])
        
        # Calcular confiança
        confidence = 0.7  # Base alta para stop hunts
        
        if confirmation:
            confidence += 0.1
        
        if sweep_info['velocity'] > 20:
            confidence += 0.1
        
        if indicators['recent_sweeps'] < 3:
            confidence += 0.05  # Não está over-trading
        
        confidence = min(confidence, 0.95)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Stop Hunt {signal_type.upper()} - Level {sweep_info['level']:.5f} swept @ {sweep_info['velocity']:.1f} pips/s",
            metadata={
                'sweep_info': sweep_info,
                'liquidity_levels': len(indicators['liquidity_levels']),
                'hidden_liquidity_confirmed': confirmation
            }
        )
        
        return self.adjust_for_spread(signal, indicators['spread'])
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída para stop hunt"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        # Saída se novo sweep na mesma direção (falha)
        if indicators['sweep_detected']:
            new_sweep = indicators['sweep_info']
            
            if position.side == 'buy' and new_sweep['direction'] == 'down':
                return ExitSignal(
                    position_id=position.id,
                    reason="New sweep against position",
                    exit_price=current_price
                )
            elif position.side == 'sell' and new_sweep['direction'] == 'up':
                return ExitSignal(
                    position_id=position.id,
                    reason="New sweep against position",
                    exit_price=current_price
                )
        
        # Trailing stop agressivo após lucro
        if position.pnl > 0:
            profit_pips = abs(current_price - position.entry_price) * 10000
            
            if profit_pips > 5:
                # Trailing de 3 pips
                if position.side == 'buy':
                    new_stop = current_price - 0.0003
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                else:
                    new_stop = current_price + 0.0003
                    if new_stop < position.stop_loss:
                        position.stop_loss = new_stop
        
        return None