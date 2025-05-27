# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union 
from datetime import datetime, timezone 
from dataclasses import dataclass, field 
import numpy as np
import pandas as pd 
import talib 

from utils.logger import setup_logger
from config.settings import CONFIG 

base_strategy_logger = setup_logger("base_strategy_class") 

@dataclass
class Signal:
    """Estrutura de sinal de trading para uma ordem."""
    strategy_name: str
    side: str  
    confidence: float 
    symbol: str = field(default_factory=lambda: CONFIG.SYMBOL if hasattr(CONFIG, 'SYMBOL') else "EURUSD") # Default para symbol
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc)) 
    entry_price: Optional[float] = None 
    stop_loss: Optional[float] = None 
    take_profit: Optional[float] = None 
    order_type: str = "Market" 
    expiration_time: Optional[datetime] = None 
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict) 

    def is_valid(self) -> bool:
        if self.side.lower() not in ['buy', 'sell']:
            base_strategy_logger.warning(f"Sinal invalido (ID Estrategia: {self.strategy_name}): Lado '{self.side}' nao e 'buy' ou 'sell'.")
            return False
        if not (0.0 <= self.confidence <= 1.0):
            base_strategy_logger.warning(f"Sinal invalido (ID Estrategia: {self.strategy_name}): Confianca {self.confidence} fora do range [0,1].")
            return False
        if self.stop_loss is not None and self.stop_loss <= 0:
            base_strategy_logger.warning(f"Sinal invalido (ID Estrategia: {self.strategy_name}): Stop Loss {self.stop_loss} deve ser positivo.")
            return False
        if self.take_profit is not None and self.take_profit <= 0:
            base_strategy_logger.warning(f"Sinal invalido (ID Estrategia: {self.strategy_name}): Take Profit {self.take_profit} deve ser positivo.")
            return False
        if not self.symbol: 
            base_strategy_logger.warning(f"Sinal invalido (ID Estrategia: {self.strategy_name}): Simbolo nao definido.")
            return False
        return True

@dataclass
class Position: 
    """Estrutura de uma posicao de trading aberta."""
    id: str 
    strategy_name: str
    symbol: str
    side: str  
    entry_price: float 
    size: float 
    open_time: datetime 
    stop_loss: Optional[float] = None 
    take_profit: Optional[float] = None 
    unrealized_pnl: float = 0.0 
    metadata: Dict[str, Any] = field(default_factory=dict) 

@dataclass
class ExitSignal:
    """Sinal para sair de uma posicao existente."""
    position_id_to_close: str 
    reason: str 
    exit_price: Optional[float] = None 
    exit_size_lots: Optional[float] = None


class BaseStrategy(ABC):
    """Classe base abstrata para todas as estrategias de trading."""

    def __init__(self, name: Optional[str] = None): 
        self.name: str = name or self.__class__.__name__
        self.active: bool = False
        self.parameters: Dict[str, Any] = self.get_default_parameters()
        self.internal_state: Dict[str, Any] = {} 
        self.suitable_regimes: List[str] = [] 
        self.current_indicators: Dict[str, Any] = {} 
        self.last_signal_generated_time: Optional[datetime] = None 
        self.min_time_between_signals_sec: int = 60  

        self.logger = setup_logger(f"strategy.{self.name}")
        self.logger.info(f"Estrategia '{self.name}' instanciada.")

    # ... (Resto da classe BaseStrategy como na sua ultima versao corrigida) ...
    # Cole o restante da classe BaseStrategy que eu forneci na resposta anterior,
    # certificando-se de que a ordem dos campos em Signal e Position está correta
    # e que 'symbol' em Signal agora tem um default_factory.

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        pass

    @abstractmethod
    async def evaluate_exit_conditions(self, open_position: Position, 
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]: 
        pass

    async def initialize_strategy(self): 
        self.logger.info(f"Inicializando estado da estrategia {self.name}...")
        self.internal_state = {}
        self.current_indicators = {}
        self.last_signal_generated_time = None
        
    async def activate_strategy(self): 
        if not self.active:
            self.active = True
            self.logger.info(f"Estrategia {self.name} ATIVADA.")
        else:
            self.logger.debug(f"Estrategia {self.name} ja esta ativa.")

    async def deactivate_strategy(self): 
        if self.active:
            self.active = False
            self.logger.info(f"Estrategia {self.name} DESATIVADA.")
        else:
            self.logger.debug(f"Estrategia {self.name} ja esta inativa.")

    async def on_tick(self, market_context: Dict[str, Any]) -> Optional[Signal]: 
        if not self.active:
            return None
        try:
            if self.last_signal_generated_time:
                time_since_last_sig = (datetime.now(timezone.utc) - self.last_signal_generated_time).total_seconds() 
                if time_since_last_sig < self.min_time_between_signals_sec:
                    return None 
            await self.calculate_indicators(market_context) 
            entry_signal = await self.generate_signal(market_context) 
            if entry_signal and entry_signal.is_valid():
                if self._is_setup_conditionally_valid(entry_signal, market_context): 
                    self.last_signal_generated_time = datetime.now(timezone.utc) 
                    self.logger.info(f"Sinal de ENTRADA gerado: {entry_signal.side} {entry_signal.symbol} @ {entry_signal.entry_price or 'Market'}")
                    return entry_signal
                else:
                    self.logger.debug(f"Sinal gerado por {self.name} nao passou na validacao condicional.")
            return None
        except Exception as e_tick: 
            self.logger.exception(f"Erro em {self.name}.on_tick:") 
            return None

    async def check_exit_for_position(self, open_position: Position, 
                                   market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        try:
            await self.calculate_indicators(market_context)
            return await self.evaluate_exit_conditions(open_position, market_context)
        except Exception as e_exit: 
            self.logger.exception(f"Erro em {self.name}.check_exit_for_position para pos {open_position.id}:")
            return None

    def update_parameters(self, new_parameters: Dict[str, Any]):
        valid_new_params = {k: v for k, v in new_parameters.items() if k in self.parameters}
        self.parameters.update(valid_new_params)
        self.logger.info(f"Parametros atualizados para {self.name}: {valid_new_params}")

    def reset_internal_state(self): 
        self.internal_state = {}
        self.current_indicators = {} 
        self.last_signal_generated_time = None
        self.logger.info(f"Estado interno da estrategia {self.name} resetado.")

    def _get_prices_from_context(self, market_context: Dict[str, Any], price_type: str = 'mid', lookback: Optional[int]=None) -> np.ndarray:
        ticks_list_any = market_context.get('recent_ticks', []) 
        if not ticks_list_any: return np.array([], dtype=float)
        if lookback:
            ticks_list_any = ticks_list_any[-lookback:]
        prices = []
        for tick_item in ticks_list_any:
            price_val = 0.0
            if hasattr(tick_item, price_type): 
                price_val = getattr(tick_item, price_type, 0.0)
            elif isinstance(tick_item, dict): 
                price_val = tick_item.get(price_type, tick_item.get('mid', 0.0))
            if price_val is None: price_val = 0.0 
            try: prices.append(float(price_val))
            except (ValueError, TypeError): prices.append(0.0) 
        return np.array(prices, dtype=float)

    def calculate_sma(self, price_series: np.ndarray, period: int) -> Optional[float]: 
        if len(price_series) >= period:
            try:
                sma_values = talib.SMA(price_series, timeperiod=period)
                return sma_values[-1] if not np.isnan(sma_values[-1]) else None
            except Exception as e_talib: 
                self.logger.error(f"Erro no calculo TA-Lib SMA ({period}): {e_talib}")
                return None
        return None

    def calculate_ema(self, price_series: np.ndarray, period: int) -> Optional[float]:
        if len(price_series) >= period:
            try:
                ema_values = talib.EMA(price_series, timeperiod=period)
                return ema_values[-1] if not np.isnan(ema_values[-1]) else None
            except Exception as e_talib:
                self.logger.error(f"Erro no calculo TA-Lib EMA ({period}): {e_talib}")
                return None
        return None

    def calculate_rsi(self, price_series: np.ndarray, period: int = 14) -> Optional[float]:
        if len(price_series) >= period + 1: 
            try:
                rsi_values = talib.RSI(price_series, timeperiod=period)
                return rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0 
            except Exception as e_talib:
                self.logger.error(f"Erro no calculo TA-Lib RSI ({period}): {e_talib}")
                return 50.0 
        return 50.0 

    def calculate_atr(self, high_series: np.ndarray, low_series: np.ndarray,
                     close_series: np.ndarray, period: int = 14) -> Optional[float]:
        if len(high_series) >= period and len(low_series) >= period and len(close_series) >= period:
            try:
                atr_values = talib.ATR(high_series, low_series, close_series, timeperiod=period)
                return atr_values[-1] if not np.isnan(atr_values[-1]) else None
            except Exception as e_talib:
                self.logger.error(f"Erro no calculo TA-Lib ATR ({period}): {e_talib}")
                return None
        return None

    def calculate_bollinger_bands(self, price_series: np.ndarray,
                                 period: int = 20,
                                 num_std_dev: float = 2.0) -> Optional[Tuple[float, float, float]]: 
        if len(price_series) >= period:
            try:
                upper, middle, lower = talib.BBANDS(price_series, timeperiod=period, nbdevup=num_std_dev, nbdevdn=num_std_dev, matype=0) 
                if not np.isnan(upper[-1]) and not np.isnan(middle[-1]) and not np.isnan(lower[-1]):
                    return upper[-1], middle[-1], lower[-1]
                return None
            except Exception as e_talib:
                self.logger.error(f"Erro no calculo TA-Lib BBANDS ({period}, {num_std_dev}): {e_talib}")
                return None
        return None

    def detect_price_pattern(self, open_prices: np.ndarray, high_prices: np.ndarray, 
                           low_prices: np.ndarray, close_prices: np.ndarray,
                           pattern_name: str) -> int: 
        min_len = min(len(open_prices), len(high_prices), len(low_prices), len(close_prices))
        if min_len == 0: return 0
        o, h, l, c = open_prices[-min_len:], high_prices[-min_len:], low_prices[-min_len:], close_prices[-min_len:]
        try:
            pattern_function = getattr(talib, pattern_name.upper(), None) 
            if pattern_function:
                pattern_results = pattern_function(o, h, l, c)
                return pattern_results[-1] if len(pattern_results) > 0 else 0
            else:
                self.logger.warning(f"Padrao TA-Lib '{pattern_name}' nao encontrado.")
                return 0
        except Exception as e_talib_pattern:
            self.logger.error(f"Erro ao detectar padrao TA-Lib '{pattern_name}': {e_talib_pattern}")
            return 0

    def calculate_support_resistance(self, high_series: np.ndarray,
                                   low_series: np.ndarray,
                                   window_periods: int = 20) -> Tuple[Optional[float], Optional[float]]: 
        if len(high_series) >= window_periods and len(low_series) >= window_periods:
            resistance = np.max(high_series[-window_periods:])
            support = np.min(low_series[-window_periods:])
            return support, resistance
        return None, None

    def calculate_standard_pivot_points(self, prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]: 
        pivot = (prev_high + prev_low + prev_close) / 3.0 
        r1 = (2 * pivot) - prev_low
        s1 = (2 * pivot) - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        return {
            'pivot': round(pivot, 5), 'r1': round(r1, 5), 'r2': round(r2, 5), 'r3': round(r3, 5),
            's1': round(s1, 5), 's2': round(s2, 5), 's3': round(s3, 5)
        }

    def calculate_risk_reward_ratio(self, entry_price_val: float, stop_loss_val: float, 
                                   take_profit_val: float) -> float:
        if entry_price_val == stop_loss_val: return 0.0
        potential_risk = abs(entry_price_val - stop_loss_val)
        potential_reward = abs(take_profit_val - entry_price_val)
        return potential_reward / potential_risk if potential_risk > 0 else 0.0

    def _is_setup_conditionally_valid(self, signal_obj: Signal, market_context: Dict[str, Any]) -> bool: 
        if not signal_obj.is_valid(): return False 
        min_rr = self.parameters.get('min_required_rr_ratio', 1.0) 
        entry_price_for_rr = signal_obj.entry_price
        current_tick = market_context.get('tick') 
        if signal_obj.order_type.upper() == "MARKET" and current_tick:
            entry_price_for_rr = current_tick.ask if signal_obj.side.lower() == 'buy' else current_tick.bid

        if entry_price_for_rr and signal_obj.stop_loss and signal_obj.take_profit:
            rr_ratio = self.calculate_risk_reward_ratio(
                entry_price_for_rr, signal_obj.stop_loss, signal_obj.take_profit
            )
            if rr_ratio < min_rr:
                self.logger.debug(f"Sinal para {signal_obj.symbol} rejeitado: R:R ({rr_ratio:.2f}) < Minimo ({min_rr:.2f}).")
                return False
        max_spread_pips_strat = self.parameters.get('max_spread_pips_for_entry')
        if max_spread_pips_strat is not None and current_tick:
            current_spread_val = current_tick.spread
            pip_size = 0.0001 if "JPY" not in signal_obj.symbol.upper() else 0.01
            current_spread_pips = current_spread_val / pip_size
            if current_spread_pips > max_spread_pips_strat:
                self.logger.debug(f"Sinal para {signal_obj.symbol} rejeitado: Spread atual ({current_spread_pips:.1f} pips) > Maximo da estrategia ({max_spread_pips_strat:.1f} pips).")
                return False
        return True

    def adjust_signal_for_spread(self, signal_obj: Signal, current_spread_val: float) -> Signal: 
        return signal_obj

    def calculate_dynamic_trailing_stop(self, 
                                 open_position: Position, 
                                 current_market_price: float, 
                                 atr_value: Optional[float] = None, 
                                 atr_multiplier: Optional[float] = None) -> Optional[float]: 
        if not atr_value: 
            atr_pips_from_indicator = self.current_indicators.get('atr_pips', 0.0) 
            pip_size_ts = 0.0001 if "JPY" not in open_position.symbol.upper() else 0.01
            atr_value = atr_pips_from_indicator * pip_size_ts 
        
        if not atr_value or atr_value <= 0:
            self.logger.debug(f"ATR invalido ({atr_value}) para trailing stop da posicao {open_position.id}.")
            return None 
        multiplier = atr_multiplier if atr_multiplier is not None else self.parameters.get('trailing_stop_atr_multiplier', 2.0)
        trailing_distance = atr_value * multiplier
        new_potential_stop: float 
        current_stop_loss_val = open_position.stop_loss 
        if current_stop_loss_val is None: 
            return None
        if open_position.side.lower() == 'buy':
            new_potential_stop = current_market_price - trailing_distance
            if new_potential_stop > current_stop_loss_val:
                return new_potential_stop
        elif open_position.side.lower() == 'sell':
            new_potential_stop = current_market_price + trailing_distance
            if new_potential_stop < current_stop_loss_val:
                return new_potential_stop
        return None 

    def get_time_filter_for_strategy(self, current_utc_hour: int) -> bool: 
        start_trading_hour = getattr(CONFIG, 'TRADING_SESSION_START_HOUR_UTC', 7) 
        end_trading_hour = getattr(CONFIG, 'TRADING_SESSION_END_HOUR_UTC', 22) 
        if current_utc_hour < start_trading_hour or current_utc_hour >= end_trading_hour:
            return False 
        return True

    def __repr__(self) -> str:
        active_status = "ATIVA" if self.active else "INATIVA" 
        num_params = len(self.parameters)
        return f"{self.name}(Status: {active_status}, Params: {num_params})"