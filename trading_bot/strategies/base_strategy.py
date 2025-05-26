# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union # Adicionado Union
from datetime import datetime, timezone # Adicionado timezone
from dataclasses import dataclass, field # Adicionado field
import numpy as np
import pandas as pd # Necessário para uso em subclasses ou futuras expansões
import talib # Usar TA-Lib para indicadores padronizados

from utils.logger import setup_logger
# from config.settings import CONFIG # Para configurações globais se necessário

# Logger específico para a classe base pode ser útil para logs genéricos de estratégia
base_strategy_logger = setup_logger("base_strategy_class") # Renomeado para evitar conflito

@dataclass
class Signal:
    """Estrutura de sinal de trading para uma ordem."""
    timestamp: datetime # Timestamp de geração do sinal (UTC)
    strategy_name: str
    symbol: str # Adicionado símbolo ao sinal
    side: str  # 'buy' ou 'sell'
    confidence: float  # 0.0 - 1.0
    entry_price: Optional[float] = None # Preço de entrada sugerido (para ordens limite/stop ou referência para mercado)
    stop_loss: Optional[float] = None # Preço absoluto do Stop Loss
    take_profit: Optional[float] = None # Preço absoluto do Take Profit
    # position_size: Optional[float] = None # Tamanho da posição é geralmente decidido pelo RiskManager
    order_type: str = "Market" # Ex: "Market", "Limit", "Stop". Usar string para flexibilidade ou Enum.
    expiration_time: Optional[datetime] = None # Para ordens pendentes
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict) # Usar Any e default_factory

    def is_valid(self) -> bool:
        """Verifica se os campos essenciais do sinal são válidos."""
        if self.side.lower() not in ['buy', 'sell']:
            base_strategy_logger.warning(f"Sinal inválido (ID Estratégia: {self.strategy_name}): Lado '{self.side}' não é 'buy' ou 'sell'.")
            return False
        if not (0.0 <= self.confidence <= 1.0):
            base_strategy_logger.warning(f"Sinal inválido (ID Estratégia: {self.strategy_name}): Confiança {self.confidence} fora do range [0,1].")
            return False
        # Stop Loss e Take Profit são opcionais na criação do sinal,
        # mas se fornecidos, devem ser válidos.
        # A validação de que SL/TP são logicamente posicionados em relação ao entry_price
        # (ex: SL abaixo de entry para buy) deve ser feita no contexto do preço de mercado atual
        # ou no ExecutionEngine antes de enviar ao broker.
        if self.stop_loss is not None and self.stop_loss <= 0:
            base_strategy_logger.warning(f"Sinal inválido (ID Estratégia: {self.strategy_name}): Stop Loss {self.stop_loss} deve ser positivo.")
            return False
        if self.take_profit is not None and self.take_profit <= 0:
            base_strategy_logger.warning(f"Sinal inválido (ID Estratégia: {self.strategy_name}): Take Profit {self.take_profit} deve ser positivo.")
            return False
        if not self.symbol:
            base_strategy_logger.warning(f"Sinal inválido (ID Estratégia: {self.strategy_name}): Símbolo não definido.")
            return False

        return True

@dataclass
class Position: # Esta dataclass representa uma POSIÇÃO ABERTA
    """Estrutura de uma posição de trading aberta."""
    id: str # ID único da posição (geralmente do broker)
    strategy_name: str
    symbol: str
    side: str  # 'Buy' ou 'Sell' (conforme a posição real)
    entry_price: float # Preço médio de entrada da posição
    size: float # Tamanho atual da posição em lotes
    stop_loss: Optional[float] # SL atual da posição
    take_profit: Optional[float] # TP atual da posição
    open_time: datetime # Timestamp de abertura da posição (UTC)
    unrealized_pnl: float = 0.0 # PnL não realizado (atualizado externamente) # Renomeado de pnl
    # trailing_stop: bool = False # Este estado (se TS está ativo) pode ser parte do metadata
    # trailing_stop_price: Optional[float] = None # O SL já reflete o TS price
    metadata: Dict[str, Any] = field(default_factory=dict) # Usar Any e default_factory

@dataclass
class ExitSignal:
    """Sinal para sair de uma posição existente."""
    position_id_to_close: str # ID da posição a ser fechada # Renomeado
    reason: str # Motivo da saída
    exit_price: Optional[float] = None # Preço de saída se for ordem limite/stop, None para mercado
    # partial_exit_fraction: float = 1.0  # Fração da posição a ser fechada (0.0 a 1.0) # Renomeado
    exit_size_lots: Optional[float] = None # Tamanho a ser fechado em lotes (se None, fechar tudo)


class BaseStrategy(ABC):
    """Classe base abstrata para todas as estratégias de trading."""

    def __init__(self, name: Optional[str] = None): # name pode ser None
        self.name: str = name or self.__class__.__name__
        self.active: bool = False
        self.parameters: Dict[str, Any] = self.get_default_parameters()
        self.internal_state: Dict[str, Any] = {} # Renomeado de state para clareza
        # Performance é geralmente rastreada externamente (DataManager, Orchestrator)
        # self.performance = {...} # Removido, pois pode causar confusão com rastreamento global

        self.suitable_regimes: List[str] = [] # Lista de MarketRegime.VALUE
        self.current_indicators: Dict[str, Any] = {} # Renomeado de indicators
        self.last_signal_generated_time: Optional[datetime] = None # Renomeado
        self.min_time_between_signals_sec: int = 60  # Renomeado e unidade clara

        # Logger específico da instância da estratégia
        self.logger = setup_logger(f"strategy.{self.name}")
        self.logger.info(f"Estratégia '{self.name}' instanciada.")


    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Retorna os parâmetros padrão configuráveis para a estratégia."""
        pass

    @abstractmethod
    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """
        Calcula e armazena os indicadores necessários em self.current_indicators.
        Este método deve ATUALIZAR self.current_indicators em vez de retorná-lo.
        """
        # Exemplo de implementação:
        # self.current_indicators['sma_20'] = self.calculate_sma(market_context['recent_ticks_df']['close'], 20)
        pass


    @abstractmethod
    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera um SINAL DE ENTRADA (Signal) com base nos indicadores atuais e no contexto de mercado.
        Retorna None se nenhuma oportunidade de entrada for encontrada.
        """
        pass

    @abstractmethod
    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado de position
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]: # Renomeado current_price
        """
        Avalia as condições de SAÍDA para uma posição aberta.
        Retorna um ExitSignal se uma condição de saída for atendida, senão None.
        """
        pass

    async def initialize_strategy(self): # Renomeado de initialize
        """Inicializa o estado específico da estratégia (ex: carregar dados históricos para indicadores)."""
        self.logger.info(f"Inicializando estado da estratégia {self.name}...")
        self.internal_state = {}
        self.current_indicators = {}
        self.last_signal_generated_time = None
        # Carregar parâmetros otimizados ou de config aqui, se não feito pelo Orchestrator
        # Ex: self.parameters.update(load_optimized_params_for_strategy(self.name))

    async def activate_strategy(self): # Renomeado de activate
        """Ativa a estratégia para processamento de ticks e geração de sinais."""
        if not self.active:
            self.active = True
            self.logger.info(f"Estratégia {self.name} ATIVADA.")
            # Pode haver lógica adicional ao ativar (ex: resetar contadores)
        else:
            self.logger.debug(f"Estratégia {self.name} já está ativa.")


    async def deactivate_strategy(self): # Renomeado de deactivate
        """Desativa a estratégia, parando a geração de novos sinais."""
        if self.active:
            self.active = False
            self.logger.info(f"Estratégia {self.name} DESATIVADA.")
            # Lógica adicional ao desativar (ex: limpar buffers de sinal pendentes)
        else:
            self.logger.debug(f"Estratégia {self.name} já está inativa.")


    async def on_tick(self, market_context: Dict[str, Any]) -> Optional[Signal]: # Renomeado de process_tick
        """
        Processa um novo tick (ou barra) e retorna um SINAL DE ENTRADA se um setup for identificado.
        Este é o método principal chamado pelo Orchestrator para cada atualização de mercado.
        """
        if not self.active:
            return None

        try:
            # Cooldown entre sinais da MESMA estratégia
            if self.last_signal_generated_time:
                time_since_last_sig = (datetime.now(timezone.utc) - self.last_signal_generated_time).total_seconds() # Usar UTC
                if time_since_last_sig < self.min_time_between_signals_sec:
                    return None # Ainda em cooldown

            # 1. Calcular/Atualizar Indicadores
            await self.calculate_indicators(market_context) # Atualiza self.current_indicators

            # 2. Gerar Sinal de Entrada (se houver)
            entry_signal = await self.generate_signal(market_context) # Renomeado signal

            if entry_signal and entry_signal.is_valid():
                # Aplicar filtros básicos do sinal antes de retornar (ex: R:R mínimo se já calculado)
                if self._is_setup_conditionally_valid(entry_signal, market_context): # Renomeado
                    self.last_signal_generated_time = datetime.now(timezone.utc) # Usar UTC
                    self.logger.info(f"Sinal de ENTRADA gerado: {entry_signal.side} {entry_signal.symbol} @ {entry_signal.entry_price or 'Market'}")
                    return entry_signal
                else:
                    self.logger.debug(f"Sinal gerado por {self.name} não passou na validação condicional.")

            return None

        except Exception as e_tick: # Renomeado
            self.logger.exception(f"Erro em {self.name}.on_tick:") # Usar logger.exception
            return None


    async def check_exit_for_position(self, open_position: Position, # Renomeado de check_exit_conditions
                                   market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Verifica condições de saída para uma posição aberta específica."""
        # Não checar self.active aqui, pois posições abertas por esta estratégia
        # devem ser gerenciadas por ela mesma, mesmo se a estratégia for desativada para NOVAS entradas.
        
        try:
            # Calcular/Atualizar Indicadores primeiro, pois as condições de saída podem depender deles
            await self.calculate_indicators(market_context)
            return await self.evaluate_exit_conditions(open_position, market_context)
        except Exception as e_exit: # Renomeado
            self.logger.exception(f"Erro em {self.name}.check_exit_for_position para pos {open_position.id}:")
            return None


    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Atualiza os parâmetros da estratégia com validação (opcional)."""
        # Filtrar apenas parâmetros que existem nos defaults para evitar adicionar chaves desconhecidas
        valid_new_params = {k: v for k, v in new_parameters.items() if k in self.parameters}
        # Poderia adicionar validação de tipo/range aqui
        self.parameters.update(valid_new_params)
        self.logger.info(f"Parâmetros atualizados para {self.name}: {valid_new_params}")
        # Após atualizar parâmetros, pode ser necessário resetar algum estado ou re-calcular indicadores
        # await self.initialize_strategy() # Ou um método mais leve de "reconfigurar"


    # Performance é rastreada externamente, este método foi removido.
    # def update_performance(self, trade_result: Dict): ...
    # def get_performance_metrics(self) -> Dict[str, float]: ...


    # Position sizing é feito pelo RiskManager, não pela estratégia.
    # def calculate_position_size(...) -> float: ...


    def reset_internal_state(self): # Renomeado de reset_state
        """Reseta o estado interno da estratégia (não os parâmetros)."""
        self.internal_state = {}
        self.current_indicators = {} # Limpar indicadores cacheados
        self.last_signal_generated_time = None
        self.logger.info(f"Estado interno da estratégia {self.name} resetado.")


    # --- Métodos Auxiliares Comuns para Indicadores (usar TA-Lib quando possível) ---
    # As implementações originais de SMA, EMA, RSI, ATR, Bollinger são mantidas abaixo
    # para referência, mas o ideal é usar TA-Lib consistentemente.

    def _get_prices_from_context(self, market_context: Dict[str, Any], price_type: str = 'mid', lookback: Optional[int]=None) -> np.ndarray:
        """Helper para extrair uma série de preços do market_context (lista de ticks)."""
        ticks_list = market_context.get('recent_ticks', [])
        if not ticks_list: return np.array([], dtype=float)

        if lookback:
            ticks_list = ticks_list[-lookback:]
        
        prices = [getattr(tick, price_type, tick.mid if hasattr(tick, 'mid') else 0.0) for tick in ticks_list if hasattr(tick, price_type) or hasattr(tick, 'mid')]
        return np.array(prices, dtype=float)


    def calculate_sma(self, price_series: np.ndarray, period: int) -> Optional[float]: # Retorna Optional[float]
        """Calcula Média Móvel Simples usando TA-Lib."""
        if len(price_series) >= period:
            try:
                sma_values = talib.SMA(price_series, timeperiod=period)
                return sma_values[-1] if not np.isnan(sma_values[-1]) else None
            except Exception as e_talib: # Renomeado
                self.logger.error(f"Erro no cálculo TA-Lib SMA ({period}): {e_talib}")
                return None
        return None

    def calculate_ema(self, price_series: np.ndarray, period: int) -> Optional[float]:
        """Calcula Média Móvel Exponencial usando TA-Lib."""
        if len(price_series) >= period:
            try:
                ema_values = talib.EMA(price_series, timeperiod=period)
                return ema_values[-1] if not np.isnan(ema_values[-1]) else None
            except Exception as e_talib:
                self.logger.error(f"Erro no cálculo TA-Lib EMA ({period}): {e_talib}")
                return None
        return None

    def calculate_rsi(self, price_series: np.ndarray, period: int = 14) -> Optional[float]:
        """Calcula RSI usando TA-Lib."""
        if len(price_series) >= period + 1: # RSI precisa de um pouco mais de dados
            try:
                rsi_values = talib.RSI(price_series, timeperiod=period)
                return rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0 # Default 50 se NaN
            except Exception as e_talib:
                self.logger.error(f"Erro no cálculo TA-Lib RSI ({period}): {e_talib}")
                return 50.0 # Default em caso de erro
        return 50.0 # Default se dados insuficientes

    def calculate_atr(self, high_series: np.ndarray, low_series: np.ndarray,
                     close_series: np.ndarray, period: int = 14) -> Optional[float]:
        """Calcula ATR (Average True Range) usando TA-Lib."""
        # TA-Lib ATR precisa de pelo menos 'period' elementos, mas mais é melhor para convergência.
        if len(high_series) >= period and len(low_series) >= period and len(close_series) >= period:
            try:
                atr_values = talib.ATR(high_series, low_series, close_series, timeperiod=period)
                return atr_values[-1] if not np.isnan(atr_values[-1]) else None
            except Exception as e_talib:
                self.logger.error(f"Erro no cálculo TA-Lib ATR ({period}): {e_talib}")
                return None
        return None


    def calculate_bollinger_bands(self, price_series: np.ndarray,
                                 period: int = 20,
                                 num_std_dev: float = 2.0) -> Optional[Tuple[float, float, float]]: # (Upper, Middle, Lower)
        """Calcula Bandas de Bollinger usando TA-Lib."""
        if len(price_series) >= period:
            try:
                upper, middle, lower = talib.BBANDS(price_series, timeperiod=period, nbdevup=num_std_dev, nbdevdn=num_std_dev, matype=0) # MA_Type.SMA
                if not np.isnan(upper[-1]) and not np.isnan(middle[-1]) and not np.isnan(lower[-1]):
                    return upper[-1], middle[-1], lower[-1]
                return None
            except Exception as e_talib:
                self.logger.error(f"Erro no cálculo TA-Lib BBANDS ({period}, {num_std_dev}): {e_talib}")
                return None
        return None

    # _detect_pattern, _calculate_support_resistance, _calculate_pivot_points foram mantidos como estavam
    # pois são mais baseados em lógica de preço do que indicadores padrão de TA-Lib.
    # Pequenos ajustes de robustez.

    def detect_price_pattern(self, open_prices: np.ndarray, high_prices: np.ndarray, # Renomeado e mais explícito
                           low_prices: np.ndarray, close_prices: np.ndarray,
                           pattern_name: str) -> int: # Retorna int como TA-Lib (0, 100, -100)
        """Detecta padrões de candlestick usando TA-Lib."""
        # TA-Lib espera que os arrays tenham o mesmo tamanho.
        min_len = min(len(open_prices), len(high_prices), len(low_prices), len(close_prices))
        if min_len == 0: return 0

        # Pegar os últimos N elementos de cada, onde N é min_len
        o, h, l, c = open_prices[-min_len:], high_prices[-min_len:], low_prices[-min_len:], close_prices[-min_len:]

        try:
            pattern_function = getattr(talib, pattern_name.upper(), None) # Ex: CDLENGULFING
            if pattern_function:
                pattern_results = pattern_function(o, h, l, c)
                return pattern_results[-1] if len(pattern_results) > 0 else 0
            else:
                self.logger.warning(f"Padrão TA-Lib '{pattern_name}' não encontrado.")
                return 0
        except Exception as e_talib_pattern:
            self.logger.error(f"Erro ao detectar padrão TA-Lib '{pattern_name}': {e_talib_pattern}")
            return 0


    def calculate_support_resistance(self, high_series: np.ndarray,
                                   low_series: np.ndarray,
                                   window_periods: int = 20) -> Tuple[Optional[float], Optional[float]]: # Renomeado e Optional
        """Calcula níveis simples de suporte (mínima recente) e resistência (máxima recente)."""
        if len(high_series) >= window_periods and len(low_series) >= window_periods:
            resistance = np.max(high_series[-window_periods:])
            support = np.min(low_series[-window_periods:])
            return support, resistance
        return None, None


    def calculate_standard_pivot_points(self, prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]: # Renomeado
        """Calcula pontos pivot padrão para o período ATUAL, usando dados do período ANTERIOR."""
        pivot = (prev_high + prev_low + prev_close) / 3.0 # Usar float

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


    def calculate_risk_reward_ratio(self, entry_price_val: float, stop_loss_val: float, # Renomeado
                                   take_profit_val: float) -> float:
        """Calcula a relação Risco/Recompensa."""
        if entry_price_val == stop_loss_val: # Evitar divisão por zero
            return 0.0
        
        potential_risk = abs(entry_price_val - stop_loss_val)
        potential_reward = abs(take_profit_val - entry_price_val)

        return potential_reward / potential_risk if potential_risk > 0 else 0.0


    def _is_setup_conditionally_valid(self, signal_obj: Signal, market_context: Dict[str, Any]) -> bool: # Renomeado
        """
        Valida se o setup do sinal atende a critérios mínimos condicionais
        (ex: R:R, spread atual).
        Esta é uma validação DA ESTRATÉGIA, antes de passar para o RiskManager.
        """
        if not signal_obj.is_valid(): return False # Checagem básica primeiro

        # Verificar R:R mínimo se todos os preços estiverem definidos
        min_rr = self.parameters.get('min_required_rr_ratio', 1.0) # Ex: pegar de params
        if signal_obj.entry_price and signal_obj.stop_loss and signal_obj.take_profit:
            rr_ratio = self.calculate_risk_reward_ratio(
                signal_obj.entry_price, signal_obj.stop_loss, signal_obj.take_profit
            )
            if rr_ratio < min_rr:
                self.logger.debug(f"Sinal para {signal_obj.symbol} rejeitado: R:R ({rr_ratio:.2f}) < Mínimo ({min_rr:.2f}).")
                return False

        # Verificar spread máximo permitido pela estratégia (se configurado)
        max_spread_pips_strat = self.parameters.get('max_spread_pips_for_entry')
        if max_spread_pips_strat is not None:
            current_spread_pips = market_context.get('spread', 0.0) * (10000 if "JPY" not in signal_obj.symbol.upper() else 100)
            if current_spread_pips > max_spread_pips_strat:
                self.logger.debug(f"Sinal para {signal_obj.symbol} rejeitado: Spread atual ({current_spread_pips:.1f} pips) > Máximo da estratégia ({max_spread_pips_strat:.1f} pips).")
                return False

        # Adicionar outras validações específicas da estratégia aqui...

        return True


    def adjust_signal_for_spread(self, signal_obj: Signal, current_spread_val: float) -> Signal: # Renomeado
        """
        Ajusta o preço de entrada, SL e TP de um SINAL para considerar o spread.
        Isto é para quando a estratégia decide entrar a mercado.
        Para ordens limite, o preço já é definido.
        """
        # Não modificar o sinal original, retornar um novo ou modificar uma cópia.
        # Esta função parece redundante se o ExecutionEngine já lida com slippage/spread.
        # Se a intenção é ajustar os NÍVEIS do sinal ANTES de enviá-lo,
        # e a entrada for a mercado:
        if signal_obj.order_type.lower() == "market":
            if signal_obj.side.lower() == 'buy':
                # Para compra a mercado, a entrada será no ASK. SL/TP são relativos a este ASK.
                # O `signal_obj.entry_price` aqui seria o preço de referência (ex: MID) no momento do sinal.
                # Se o ExecutionEngine for entrar no ASK, não precisamos ajustar entry_price aqui.
                # Mas se SL/TP foram calculados a partir do MID, eles podem precisar de ajuste.
                # Ex: signal_obj.stop_loss -= current_spread_val / 2.0 # Tornar SL mais conservador
                #     signal_obj.take_profit += current_spread_val / 2.0
                pass
            elif signal_obj.side.lower() == 'sell':
                # Para venda a mercado, a entrada será no BID.
                # signal_obj.stop_loss += current_spread_val / 2.0
                # signal_obj.take_profit -= current_spread_val / 2.0
                pass
        # Esta função pode ser mais complexa ou desnecessária dependendo de como o ExecutionEngine opera.
        # O original estava ajustando SL/TP para ambos os lados, o que pode não ser sempre o desejado.
        return signal_obj


    def calculate_dynamic_trailing_stop(self, # Renomeado de calculate_trailing_stop
                                 open_position: Position, # Renomeado
                                 current_market_price: float, # Renomeado
                                 atr_value: Optional[float] = None, # Renomeado
                                 atr_multiplier: Optional[float] = None) -> Optional[float]: # Renomeado
        """
        Calcula um novo nível de trailing stop dinâmico.
        Retorna o novo preço de stop loss, ou None se não deve ser atualizado.
        """
        if not atr_value: # Tentar obter ATR dos indicadores atuais se não fornecido
            atr_value = self.current_indicators.get('atr') # Supondo que 'atr' está em current_indicators
        if not atr_value or atr_value <= 0:
            self.logger.debug(f"ATR inválido ({atr_value}) para trailing stop da posição {open_position.id}.")
            return None # Não pode calcular sem ATR

        multiplier = atr_multiplier if atr_multiplier is not None else self.parameters.get('trailing_stop_atr_multiplier', 2.0)
        trailing_distance = atr_value * multiplier

        new_potential_stop: float # Adicionada tipagem
        current_stop_loss = open_position.stop_loss or (0.0 if open_position.side.lower() == 'buy' else float('inf'))


        if open_position.side.lower() == 'buy':
            new_potential_stop = current_market_price - trailing_distance
            # Só mover o stop para cima (a favor da posição)
            if new_potential_stop > current_stop_loss:
                return new_potential_stop
        elif open_position.side.lower() == 'sell':
            new_potential_stop = current_market_price + trailing_distance
            # Só mover o stop para baixo (a favor da posição)
            if new_potential_stop < current_stop_loss:
                return new_potential_stop
        
        return None # Nenhum ajuste necessário


    # Métodos de scale-in/scale-out foram removidos.
    # Esta lógica é complexa e geralmente gerenciada pelo Orchestrator ou
    # por uma meta-estratégia de gestão de posição, não pela estratégia de sinal individual.
    # Se necessário, podem ser re-adicionados com lógica mais clara.


    def get_time_filter_for_strategy(self, current_utc_hour: int) -> bool: # Renomeado e com arg
        """
        Filtro de horário específico da estratégia (se houver).
        Retorna True se a estratégia PODE operar neste horário.
        """
        # Exemplo: esta estratégia só opera durante o overlap Londres/NY
        # allowed_hours = self.parameters.get('allowed_trading_hours_utc', list(range(24))) # Default: todas as horas
        # if current_utc_hour not in allowed_hours:
        #     return False

        # A lógica original era global, não específica da estratégia.
        # Se a intenção é um filtro global, ele deve estar no Orchestrator.
        # Se for específico da estratégia, carregar de self.parameters.
        # Mantendo a lógica original como exemplo de filtro global que uma estratégia poderia consultar:
        if current_utc_hour < getattr(CONFIG, 'TRADING_SESSION_START_HOUR_UTC', 7) or \
           current_utc_hour >= getattr(CONFIG, 'TRADING_SESSION_END_HOUR_UTC', 22):
            return False # Fora do horário principal Londres/NY

        # Evitar horários de baixa liquidez ou alta volatilidade de aberturas/fechamentos
        # if current_utc_hour in [8, 13, 16, 21]: # Horas de abertura/fechamento
        #     minute_of_hour = datetime.now(timezone.utc).minute
        #     if minute_of_hour < 15 or minute_of_hour > 45: # Evitar primeiros/últimos 15 min da hora
        #         return False
        return True


    # calculate_kelly_criterion foi removido pois já existe em PositionSizer
    # e é mais apropriado lá.

    def __repr__(self) -> str:
        active_status = "ATIVA" if self.active else "INATIVA" # Renomeado
        # Performance.total_trades não existe mais aqui.
        # Poderia logar o número de sinais gerados ou estado interno.
        num_params = len(self.parameters)
        return f"{self.name}(Status: {active_status}, Params: {num_params})"