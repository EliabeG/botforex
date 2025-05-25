# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("base_strategy")

@dataclass
class Signal:
    """Estrutura de sinal de trading"""
    timestamp: datetime
    strategy_name: str
    side: str  # 'buy' ou 'sell'
    confidence: float  # 0-1
    stop_loss: float
    take_profit: float
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    reason: str = ""
    metadata: Dict = None
    
    def is_valid(self) -> bool:
        """Verifica se o sinal é válido"""
        return (
            self.side in ['buy', 'sell'] and
            0 <= self.confidence <= 1 and
            self.stop_loss > 0 and
            self.take_profit > 0
        )

@dataclass
class Position:
    """Estrutura de posição aberta"""
    id: str
    strategy_name: str
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    pnl: float = 0.0
    trailing_stop: bool = False
    metadata: Dict = None

@dataclass
class ExitSignal:
    """Sinal de saída de posição"""
    position_id: str
    reason: str
    exit_price: Optional[float] = None
    partial_exit: float = 1.0  # Percentual para sair (1.0 = 100%)

class BaseStrategy(ABC):
    """Classe base abstrata para todas as estratégias"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.active = False
        self.parameters = self.get_default_parameters()
        self.state = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        self.suitable_regimes = []
        self.indicators = {}
        self.last_signal_time = None
        self.min_time_between_signals = 60  # segundos
        
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Retorna parâmetros padrão da estratégia"""
        pass
    
    @abstractmethod
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula indicadores necessários para a estratégia"""
        pass
    
    @abstractmethod
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinal de entrada baseado no contexto de mercado"""
        pass
    
    @abstractmethod
    async def calculate_exit_conditions(self, position: Position, 
                                       current_price: float) -> Optional[ExitSignal]:
        """Calcula condições de saída para posição"""
        pass
    
    async def initialize(self):
        """Inicializa a estratégia"""
        logger.info(f"Inicializando estratégia {self.name}")
        self.state = {}
        self.indicators = {}
    
    async def activate(self):
        """Ativa a estratégia"""
        logger.info(f"Ativando estratégia {self.name}")
        self.active = True
    
    async def deactivate(self):
        """Desativa a estratégia"""
        logger.info(f"Desativando estratégia {self.name}")
        self.active = False
    
    async def process_tick(self, market_context: Dict) -> Optional[Signal]:
        """Processa tick e retorna sinal se houver"""
        if not self.active:
            return None
        
        try:
            # Verificar cooldown entre sinais
            if self.last_signal_time:
                time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
                if time_since_last < self.min_time_between_signals:
                    return None
            
            # Calcular indicadores
            self.indicators = await self.calculate_indicators(market_context)
            
            # Gerar sinal
            signal = await self.generate_signal(market_context)
            
            if signal and signal.is_valid():
                self.last_signal_time = datetime.now()
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erro em {self.name}.process_tick: {e}")
            return None
    
    async def check_exit_conditions(self, position: Position, 
                                   current_price: float) -> Optional[ExitSignal]:
        """Verifica condições de saída"""
        if not self.active:
            return None
        
        try:
            return await self.calculate_exit_conditions(position, current_price)
        except Exception as e:
            logger.error(f"Erro em {self.name}.check_exit_conditions: {e}")
            return None
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Atualiza parâmetros da estratégia"""
        self.parameters.update(new_parameters)
        logger.info(f"Parâmetros atualizados para {self.name}: {new_parameters}")
    
    def update_performance(self, trade_result: Dict):
        """Atualiza métricas de performance"""
        self.performance['total_trades'] += 1
        
        pnl = trade_result.get('pnl', 0)
        self.performance['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        # Calcular win rate
        if self.performance['total_trades'] > 0:
            self.performance['win_rate'] = (
                self.performance['winning_trades'] / 
                self.performance['total_trades']
            )
    
    def calculate_position_size(self, signal: Signal, account_balance: float, 
                               risk_per_trade: float) -> float:
        """Calcula tamanho da posição baseado em risco"""
        if signal.entry_price and signal.stop_loss:
            # Calcular risco em pips
            risk_pips = abs(signal.entry_price - signal.stop_loss) * 10000
            
            # Calcular valor por pip
            pip_value = 10  # Para lote padrão de 100k
            
            # Calcular tamanho do lote
            risk_amount = account_balance * risk_per_trade
            lot_size = risk_amount / (risk_pips * pip_value)
            
            # Arredondar para precisão do broker
            return round(lot_size, 2)
        
        return 0.01  # Lote mínimo padrão
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Retorna métricas de performance"""
        metrics = self.performance.copy()
        
        # Calcular métricas adicionais
        if metrics['total_trades'] > 0:
            metrics['avg_win'] = (
                metrics['total_pnl'] / metrics['winning_trades']
                if metrics['winning_trades'] > 0 else 0
            )
            metrics['avg_loss'] = (
                metrics['total_pnl'] / metrics['losing_trades']
                if metrics['losing_trades'] > 0 else 0
            )
            metrics['expectancy'] = metrics['total_pnl'] / metrics['total_trades']
        
        return metrics
    
    def reset_state(self):
        """Reseta estado interno da estratégia"""
        self.state = {}
        self.indicators = {}
        self.last_signal_time = None
    
    # Métodos auxiliares comuns
    
    def calculate_sma(self, values: np.ndarray, period: int) -> float:
        """Calcula média móvel simples"""
        if len(values) >= period:
            return np.mean(values[-period:])
        return np.mean(values)
    
    def calculate_ema(self, values: np.ndarray, period: int) -> float:
        """Calcula média móvel exponencial"""
        if len(values) == 0:
            return 0
        
        if len(values) < period:
            return np.mean(values)
        
        multiplier = 2 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, values: np.ndarray, period: int = 14) -> float:
        """Calcula RSI"""
        if len(values) < period + 1:
            return 50  # Neutro
        
        deltas = np.diff(values)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                     close: np.ndarray, period: int = 14) -> float:
        """Calcula ATR (Average True Range)"""
        if len(high) < 2:
            return 0
        
        tr_list = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        
        if len(tr_list) >= period:
            return np.mean(tr_list[-period:])
        elif tr_list:
            return np.mean(tr_list)
        
        return 0
    
    def calculate_bollinger_bands(self, values: np.ndarray, 
                                 period: int = 20, 
                                 num_std: float = 2) -> Tuple[float, float, float]:
        """Calcula Bandas de Bollinger"""
        if len(values) < period:
            current = values[-1] if len(values) > 0 else 0
            return current, current, current
        
        sma = np.mean(values[-period:])
        std = np.std(values[-period:])
        
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        
        return upper, sma, lower
    
    def detect_pattern(self, values: np.ndarray, pattern_type: str) -> bool:
        """Detecta padrões de preço comuns"""
        if len(values) < 3:
            return False
        
        if pattern_type == "bullish_engulfing":
            # Padrão de engolfo de alta
            return (values[-2] < values[-3] and  # Candle anterior bearish
                    values[-1] > values[-2] and   # Candle atual bullish
                    values[-1] > values[-3])      # Fecha acima da abertura anterior
        
        elif pattern_type == "bearish_engulfing":
            # Padrão de engolfo de baixa
            return (values[-2] > values[-3] and  # Candle anterior bullish
                    values[-1] < values[-2] and   # Candle atual bearish
                    values[-1] < values[-3])      # Fecha abaixo da abertura anterior
        
        elif pattern_type == "hammer":
            # Martelo (reversão de alta)
            body = abs(values[-1] - values[-2])
            total = max(values[-1], values[-2]) - min(values[-1], values[-2])
            return body < total * 0.3 and values[-1] > values[-2]
        
        elif pattern_type == "shooting_star":
            # Estrela cadente (reversão de baixa)
            body = abs(values[-1] - values[-2])
            total = max(values[-1], values[-2]) - min(values[-1], values[-2])
            return body < total * 0.3 and values[-1] < values[-2]
        
        return False
    
    def calculate_support_resistance(self, highs: np.ndarray, 
                                   lows: np.ndarray, 
                                   window: int = 20) -> Tuple[float, float]:
        """Calcula níveis de suporte e resistência"""
        if len(highs) < window or len(lows) < window:
            return 0, 0
        
        # Resistência = máxima recente
        resistance = np.max(highs[-window:])
        
        # Suporte = mínima recente
        support = np.min(lows[-window:])
        
        return support, resistance
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calcula pontos pivot"""
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def calculate_risk_reward_ratio(self, entry: float, stop_loss: float, 
                                   take_profit: float) -> float:
        """Calcula relação risco/recompensa"""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def is_valid_setup(self, signal: Signal, min_rr_ratio: float = 1.5) -> bool:
        """Valida se o setup atende critérios mínimos"""
        if not signal.is_valid():
            return False
        
        # Verificar relação risco/recompensa
        if signal.entry_price:
            rr_ratio = self.calculate_risk_reward_ratio(
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )
            
            if rr_ratio < min_rr_ratio:
                return False
        
        # Verificar confiança mínima
        if signal.confidence < 0.6:
            return False
        
        return True
    
    def adjust_for_spread(self, signal: Signal, spread: float) -> Signal:
        """Ajusta sinal considerando spread"""
        spread_adjustment = spread / 2
        
        if signal.side == 'buy':
            if signal.entry_price:
                signal.entry_price += spread_adjustment
            signal.stop_loss -= spread_adjustment
            signal.take_profit += spread_adjustment
        else:  # sell
            if signal.entry_price:
                signal.entry_price -= spread_adjustment
            signal.stop_loss += spread_adjustment
            signal.take_profit -= spread_adjustment
        
        return signal
    
    def calculate_trailing_stop(self, position: Position, current_price: float, 
                               atr: float, multiplier: float = 2.0) -> float:
        """Calcula novo nível de trailing stop"""
        distance = atr * multiplier
        
        if position.side == 'buy':
            new_stop = current_price - distance
            # Só move stop para cima
            return max(position.stop_loss, new_stop)
        else:  # sell
            new_stop = current_price + distance
            # Só move stop para baixo
            return min(position.stop_loss, new_stop)
    
    def should_scale_in(self, position: Position, current_price: float, 
                       entry_price: float) -> bool:
        """Verifica se deve aumentar posição (scale-in)"""
        if position.side == 'buy':
            # Scale-in se preço caiu X% desde entrada
            return current_price < entry_price * 0.995
        else:  # sell
            # Scale-in se preço subiu X% desde entrada
            return current_price > entry_price * 1.005
    
    def should_scale_out(self, position: Position, current_price: float, 
                        entry_price: float) -> bool:
        """Verifica se deve reduzir posição (scale-out)"""
        pnl_pct = (current_price - entry_price) / entry_price
        
        if position.side == 'buy':
            # Scale-out se lucro > 1%
            return pnl_pct > 0.01
        else:  # sell
            # Scale-out se lucro > 1%
            return pnl_pct < -0.01
    
    def get_time_filter(self, hour: int) -> bool:
        """Filtro de horário para trading"""
        # Evitar horários de baixa liquidez
        if hour < 7 or hour > 22:  # Fora do horário Londres/NY
            return False
        
        # Evitar abertura/fechamento de mercados (alta volatilidade)
        if hour in [8, 13, 16, 21]:  # Aberturas principais
            return False
        
        return True
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, 
                                 avg_loss: float) -> float:
        """Calcula fração ótima de capital para apostar (Kelly Criterion)"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        # Limitar a 25% (Kelly conservador)
        return min(max(kelly, 0), 0.25)
    
    def __repr__(self) -> str:
        return f"{self.name}(active={self.active}, trades={self.performance['total_trades']})"