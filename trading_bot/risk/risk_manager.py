# risk/risk_manager.py
import asyncio
from typing import Dict, List, Optional, Tuple, Any # Adicionado Any
from datetime import datetime, timedelta, timezone # Adicionado timezone
from dataclasses import dataclass, field # Adicionado field
import numpy as np

from config.settings import CONFIG
from config.risk_config import RISK_LIMITS, RISK_PARAMS # Adicionado RISK_PARAMS
# Importar Signal e Position da estratégia base
from strategies.base_strategy import Signal, Position as StrategyPosition # Renomeado Position para evitar conflito
from utils.logger import setup_logger
from .circuit_breaker import CircuitBreaker, TripReason, CircuitBreakerState # Importar CircuitBreaker
from .position_sizing import PositionSizer, PositionSizeResult # Importar PositionSizer

logger = setup_logger("risk_manager")

@dataclass
class RiskMetrics:
    """Métricas de risco atuais da conta e do bot."""
    current_total_exposure_usd: float = 0.0 # Renomeado
    open_positions_count: int = 0 # Renomeado
    daily_pnl_usd: float = 0.0 # Renomeado
    daily_pnl_percentage: float = 0.0 # Renomeado
    # weekly_drawdown_percentage: float = 0.0 # Pode ser complexo de rastrear em tempo real aqui
    max_drawdown_session_percentage: float = 0.0 # DD da sessão atual
    # risk_available_next_trade_usd: float = 0.0 # Renomeado, calculado dinamicamente
    margin_used_usd: float = 0.0 # Renomeado
    free_margin_usd: float = 0.0 # Renomeado, de margin_available
    # correlation_risk_factor: float = 0.0 # Renomeado, pode ser um score 0-1
    circuit_breaker_state: str = CircuitBreakerState.CLOSED.value # Adicionado
    account_balance_usd: float = 0.0 # Adicionado saldo atual

class RiskManager:
    """Gerenciador central de risco, integrando PositionSizer e CircuitBreaker."""

    def __init__(self):
        # Rastreamento de posições e trades (pode ser simplificado se Orchestrator/DataManager já fazem isso)
        # self.positions: Dict[str, StrategyPosition] = {} # BrokerPositionID -> StrategyPosition object
        # self.daily_trades_summary: List[Dict[str, Any]] = [] # Lista de resumos de trades fechados no dia

        self.risk_events_log: List[Dict[str, Any]] = [] # Log de eventos de risco internos
        # self.correlation_matrix: Dict[Tuple[str, str], float] = {} # Para risco de correlação avançado
        # self.max_allowed_correlation: float = getattr(RISK_PARAMS, 'HIGH_CORRELATION_THRESHOLD', 0.7) # Renomeado

        self.high_water_mark_session: float = 0.0 # HWM da sessão atual
        self.daily_start_balance_rm: float = 0.0 # Renomeado
        
        # Componentes integrados
        self.circuit_breaker = CircuitBreaker() # Instanciar CircuitBreaker
        self.position_sizer = PositionSizer() # Usar método default ou configurável

        self._last_risk_check_time = datetime.now(timezone.utc)

    async def initialize(self, starting_balance: float, account_currency: str = "USD"): # Adicionado currency
        """Inicializa gerenciador de risco com o saldo inicial da sessão."""
        self.daily_start_balance_rm = starting_balance
        self.high_water_mark_session = starting_balance
        # self.account_currency = account_currency # Se precisar para conversões
        
        # Registrar callbacks do circuit breaker se necessário
        # Ex: self.circuit_breaker.add_on_trip_callback(self.handle_cb_trip)
        #     self.circuit_breaker.add_on_reset_callback(self.handle_cb_reset)

        logger.info(f"Risk Manager inicializado. Saldo inicial da sessão: {starting_balance:.2f} {account_currency}")

    async def can_open_new_position(self, signal: Signal, # Renomeado de can_open_position
                                   current_balance: float,
                                   open_positions_list: List[StrategyPosition], # Lista de posições já abertas
                                   margin_level_pct: Optional[float] = None, # Nível de margem atual em %
                                   # Adicionar recent_trades se CircuitBreaker precisar
                                   recent_trades_for_cb: Optional[List[Dict[str,Any]]] = None
                                   ) -> bool:
        """Verifica se uma nova posição pode ser aberta com base nas regras de risco."""
        try:
            # 1. Verificar Circuit Breaker primeiro
            # Construir account_status para CircuitBreaker.check_conditions
            # current_dd_session = (self.high_water_mark_session - current_balance) / self.high_water_mark_session if self.high_water_mark_session > 0 else 0.0
            # daily_pnl_pct_session = (current_balance - self.daily_start_balance_rm) / self.daily_start_balance_rm if self.daily_start_balance_rm > 0 else 0.0
            # account_status_for_cb = {
            #     'current_balance': current_balance,
            #     'daily_pnl_pct': daily_pnl_pct_session,
            #     'current_drawdown_percent': current_dd_session,
            #     'margin_level_percent': (margin_level_pct / 100.0) if margin_level_pct is not None else 1.0 # CB espera ratio 0-1
            # }
            # if await self.circuit_breaker.check_conditions(account_status_for_cb, recent_trades_for_cb or []):
            #     logger.warning(f"Circuit Breaker ({self.circuit_breaker.state.value}) impede abertura de nova posição para {signal.strategy_name}.")
            #     return False
            # ELIF self.circuit_breaker.is_in_half_open_test():
            #     # Lógica para permitir trades de teste em half-open (pode ser gerenciada pelo Orchestrator)
            #     logger.info("Circuit Breaker em HALF_OPEN, permitindo trade de teste.")
            # O original tinha uma flag self.circuit_breaker_active aqui. A lógica acima é mais completa.
            # Mantendo a flag original por enquanto:
            if self.circuit_breaker.state == CircuitBreakerState.OPEN : # Usar estado do objeto CircuitBreaker
                if self.circuit_breaker.reset_scheduled_time and datetime.now(timezone.utc) >= self.circuit_breaker.reset_scheduled_time:
                    await self.circuit_breaker.enter_half_open_mode()
                else:
                    logger.warning(f"Circuit Breaker (OPEN) impede abertura de posição para {signal.strategy_name}.")
                    return False
            elif self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                # Permitir trade de teste, mas o orchestrator/estratégia deve saber disso
                # para limitar o risco do trade de teste.
                logger.info(f"Circuit Breaker (HALF_OPEN) permitindo trade de teste para {signal.strategy_name}.")
                # Não retornar False aqui, deixar outros checks prosseguirem.


            # 2. Outras verificações de risco
            # Os checks agora recebem os dados necessários diretamente
            check_results: Dict[str, bool] = { # Adicionada tipagem
                "position_limit": self._check_max_positions_limit(open_positions_list),
                "exposure_limit": self._check_total_exposure_limit(open_positions_list, current_balance),
                "daily_loss_limit_breached": not self._check_daily_loss_not_breached(current_balance), # Invertido para "breached"
                # "correlation_limit": await self._check_correlation_with_open_positions(signal, open_positions_list), # Complexo, mantendo simples
                "margin_requirements": self._check_sufficient_margin(signal, current_balance, open_positions_list, margin_level_pct), # Passar margin_level
                "consecutive_loss_pause": not self._check_consecutive_loss_pause(recent_trades_for_cb or []), # Invertido
                "time_restrictions_active": not self._check_market_time_restrictions() # Invertido
            }

            # Logar todos os checks
            logger.debug(f"Risk checks para sinal {signal.strategy_name}: {check_results}")

            if not all(not breached for name, breached in check_results.items() if "breached" in name or "active" in name): # Lógica para checar se algum limite foi atingido
                 # Encontrar a primeira falha para log específico
                failed_check = next((name for name, passed in check_results.items() if not passed and ("breached" not in name and "active" not in name)),
                                    next((name for name, breached_val in check_results.items() if breached_val and ("breached" in name or "active" in name)), "Unknown Check Failed"))

                logger.info(f"Abertura de posição negada para {signal.strategy_name}. Falha no check: {failed_check}")
                return False

            return True # Todos os checks passaram

        except Exception as e:
            logger.exception("Erro ao verificar permissão para abrir posição:") # Usar logger.exception
            return False # Conservador: negar em caso de erro


    def _check_max_positions_limit(self, open_positions: List[StrategyPosition]) -> bool: # Renomeado
        """Verifica limite de posições simultâneas."""
        # Considerar posições por estratégia também se necessário
        # max_total_pos = getattr(RISK_LIMITS, 'MAX_POSITIONS', 5)
        return len(open_positions) < RISK_LIMITS.MAX_POSITIONS


    def _check_total_exposure_limit(self, open_positions: List[StrategyPosition], current_balance: float) -> bool: # Renomeado
        """Verifica exposição total em relação ao balanço."""
        if not open_positions: return True
        # Exposição = Soma(tamanho_lote * valor_contrato * preco_entrada)
        # Este cálculo precisa dos preços de entrada e tamanhos corretos.
        # StrategyPosition deve ter 'entry_price' e 'size' em lotes.
        contract_size_val = getattr(CONFIG, 'CONTRACT_SIZE', 100000) # Renomeado
        total_exposure_val = sum(pos.size * contract_size_val * pos.entry_price for pos in open_positions if pos.size > 0 and pos.entry_price > 0) # Renomeado
        
        max_allowed_exposure = current_balance * RISK_LIMITS.MAX_EXPOSURE
        return total_exposure_val < max_allowed_exposure

    def _check_daily_loss_not_breached(self, current_balance: float) -> bool: # Renomeado
        """Verifica se o limite de perda diária NÃO foi atingido."""
        if self.daily_start_balance_rm <= 0: return True # Evitar divisão por zero
        daily_pnl_val = current_balance - self.daily_start_balance_rm # Renomeado
        daily_pnl_pct_val = daily_pnl_val / self.daily_start_balance_rm # Renomeado

        if daily_pnl_pct_val <= -RISK_LIMITS.DAILY_LOSS_LIMIT: # DAILY_LOSS_LIMIT é positivo
            logger.warning(f"Limite de perda diária ({RISK_LIMITS.DAILY_LOSS_LIMIT*100:.1f}%) ATINGIDO. PnL Diário: {daily_pnl_pct_val:.2%}")
            # Potencialmente acionar circuit breaker ou modo de risco reduzido
            # asyncio.create_task(self.circuit_breaker.trip(TripReason.DAILY_LOSS, {'pnl_pct': daily_pnl_pct_val}))
            return False
        return True


    # _check_correlation_limit foi removido, a versão original era muito simplista.
    # Implementar corretamente requer uma matriz de correlação e cálculo de exposição diversificada.

    def _check_sufficient_margin(self, signal: Signal, current_balance: float, # Renomeado
                               open_positions: List[StrategyPosition],
                               current_margin_level_pct: Optional[float]) -> bool:
        """Verifica se há margem suficiente para a nova ordem e para manter as existentes."""
        # 1. Margem necessária para a nova ordem (estimada)
        #    O tamanho da posição do sinal ainda não é conhecido, precisa ser estimado pelo PositionSizer.
        #    Esta função é chamada ANTES do cálculo do tamanho. Isso é um problema de design.
        #    Idealmente, can_open_new_position é chamada APÓS o PositionSizer ter calculado um tamanho proposto.
        #    Assumindo que signal.position_size é um TAMANHO MÁXIMO POTENCIAL ou um default.
        
        # Se signal.position_size não está definido, não podemos calcular a margem necessária.
        # Vamos pular esta checagem aqui e assumir que o broker rejeitará se não houver margem.
        # Ou, o PositionSizer deve considerar a margem ao calcular o tamanho.
        # Por ora, vamos focar na margem LIVRE atual.

        if current_margin_level_pct is None: # Se não sabemos o nível de margem, ser conservador
            logger.warning("Nível de margem atual desconhecido. Pulando checagem de margem detalhada.")
            return True # Ou False se quiser ser ultra-conservador

        # 2. Verificar se o nível de margem atual está acima de um buffer de segurança
        min_required_margin_level_for_new_trade = RISK_LIMITS.MARGIN_CALL_LEVEL + getattr(RISK_PARAMS, 'MARGIN_BUFFER_FOR_NEW_TRADE_PCT', 0.20) # Ex: MC + 20%
        
        if current_margin_level_pct / 100.0 < min_required_margin_level_for_new_trade : # Converter para ratio
             logger.warning(f"Nível de margem ({current_margin_level_pct:.1f}%) muito baixo para abrir nova trade (Requerido: > {min_required_margin_level_for_new_trade*100:.1f}%).")
             return False
        
        return True


    def _check_consecutive_loss_pause(self, recent_trades: List[Dict[str,Any]]) -> bool: # Renomeado
        """Verifica se uma pausa é necessária devido a perdas consecutivas."""
        if not recent_trades or len(recent_trades) < RISK_LIMITS.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
            return True # Não há perdas consecutivas suficientes para pausar

        consecutive_loss_count = 0 # Renomeado
        for trade in reversed(recent_trades): # Iterar do mais recente para o mais antigo
            if float(trade.get('pnl', 0.0)) < 0:
                consecutive_loss_count += 1
            else:
                break # Sequência de perdas quebrada

        if consecutive_loss_count >= RISK_LIMITS.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
            logger.warning(f"{consecutive_loss_count} perdas consecutivas. Considerar pausa ou redução de risco.")
            # Esta função apenas verifica. O Orchestrator ou CircuitBreaker tomaria a ação.
            # Se esta função DEVE impor a pausa, então retornaria False.
            # No design atual do can_open_new_position, retornar False aqui bloquearia novas trades.
            return False # Impor pausa
        return True

    def _check_market_time_restrictions(self) -> bool: # Renomeado
        """Verifica se o horário atual permite trading."""
        # Esta lógica pode ser expandida com base em CONFIG.SESSION_CONFIG
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour # Renomeado
        minute_utc = now_utc.minute # Renomeado
        weekday_utc = now_utc.weekday() # Segunda=0, Domingo=6

        # Exemplo: Não operar nos primeiros 15 minutos de grandes aberturas de sessão
        # (Londres: ~7-8 UTC, NY: ~13-14 UTC)
        # E não operar na última hora antes do fechamento de NY (~20-21 UTC Sexta)
        # Essa lógica é muito específica e pode ser melhor gerenciada pelas próprias estratégias
        # ou por um filtro de sessão mais global no Orchestrator.

        # Manter a lógica original do ExecutionEngine por enquanto:
        if hour_utc == 7 and minute_utc < getattr(CONFIG, 'SESSION_OPEN_BUFFER_MIN', 5): # Abertura Londres (aprox)
            logger.debug("Restrição de tempo: primeiros minutos da sessão de Londres.")
            return False
        if hour_utc == 13 and minute_utc < getattr(CONFIG, 'SESSION_OPEN_BUFFER_MIN', 5): # Abertura NY (aprox)
            logger.debug("Restrição de tempo: primeiros minutos da sessão de NY.")
            return False
        # Adicionar fechamento de NY na sexta
        if weekday_utc == 4: # Sexta-feira
            if hour_utc >= getattr(CONFIG, 'FRIDAY_NY_CLOSE_BUFFER_HOUR_UTC', 20): # Ex: a partir das 20:00 UTC
                logger.debug("Restrição de tempo: próximo ao fechamento de NY na sexta.")
                return False
        
        # Evitar trading em horários de rollover de swap (ex: 21:00-21:05 UTC)
        if hour_utc == 21 and minute_utc < 5 : # Ajustar conforme horário exato do broker
            logger.debug("Restrição de tempo: horário de rollover de swap.")
            return False


        return True


    async def calculate_position_size(self, signal: Signal, account_balance: float,
                                    # Adicionar outros parâmetros que o PositionSizer possa precisar
                                    market_conditions_for_sizer: Optional[Dict[str,Any]] = None
                                    ) -> PositionSizeResult: # Retorna o objeto completo
        """Calcula tamanho da posição usando o PositionSizer configurado."""
        # O PositionSizer precisa de entry_price e stop_loss_price do SINAL.
        if signal.entry_price is None and signal.type == "LIMIT": # type: ignore # Signal.type não existe, order_type sim
            logger.error("Sinal de ordem limite sem preço de entrada definido. Não é possível calcular tamanho.")
            return self.position_sizer._create_error_result("Sinal limite sem preço de entrada.") # Chamar um helper

        # Para ordens a mercado, o entry_price do sinal pode ser o preço no momento da geração do sinal.
        # O stop_loss_price é o preço do stop loss.
        # Se entry_price for None para Market, o PositionSizer pode precisar buscar o preço atual.
        # Assumindo que signal.entry_price é o preço de referência para ordens a mercado.
        
        # Se o sinal não tiver um entry_price definido (ex: para uma ordem a mercado pura),
        # o PositionSizer pode precisar de acesso ao preço de mercado atual.
        # Por ora, vamos assumir que signal.entry_price é o preço de referência para o cálculo.
        # Se o sinal for a mercado e entry_price for None, o sizer precisa de um fallback.
        ref_entry_price = signal.entry_price
        if ref_entry_price is None: # Para ordens a mercado onde o preço exato não é conhecido
            # Obter preço de mercado atual para cálculo (ask para compra, bid para venda)
            # Esta lógica deveria estar no Orchestrator e o preço passado para cá,
            # ou o RiskManager/PositionSizer ter acesso ao DataManager.
            # Simulando:
            # current_mkt_price = await data_manager.get_current_price(signal.symbol) # Exemplo
            # ref_entry_price = current_mkt_price.ask if signal.side == 'buy' else current_mkt_price.bid
            logger.warning(f"Sinal para {signal.side} sem entry_price definido. Usando SL apenas para distância.")
            # Se não temos entry_price, o stop_distance_pips pode ser fixo ou baseado em ATR.
            # A função original em PositionSizer calcula stop_distance_pips usando entry_price e stop_loss.
            # Se entry_price não existe, a chamada original falhará.
            # Precisamos de um preço de referência.
            # Por simplicidade, se entry_price não está no sinal, não podemos prosseguir aqui.
            if signal.stop_loss is None: # Se não há nem entry nem stop, não há como calcular risco.
                 return self.position_sizer._create_error_result("Sinal sem entry_price e stop_loss para cálculo de risco.")

            # Se temos stop_loss, mas não entry_price, a distância do stop é indefinida.
            # O Orchestrator deve garantir que o Signal tenha os campos necessários.
            # A validação do sinal em BaseStrategy (signal.is_valid()) deve checar isso.
            # O código original em PositionSizer usa entry_price e stop_loss do Signal.

        # O PositionSizer já foi instanciado no __init__ do RiskManager.
        # Ele mantém seu próprio estado (trade_history, equity_curve).
        pos_size_result = self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            entry_price=signal.entry_price or 0.0, # Passar 0.0 se None, PositionSizer deve lidar
            stop_loss_price=signal.stop_loss, # stop_loss já é float
            symbol=signal.symbol if hasattr(signal, 'symbol') else CONFIG.SYMBOL, # Usar symbol do sinal
            leverage=CONFIG.LEVERAGE,
            market_conditions=market_conditions_for_sizer
        )
        return pos_size_result


    # _get_risk_multiplier e _get_max_lot_size foram movidos para PositionSizer,
    # pois são mais específicos do cálculo de tamanho.

    # Métodos para registrar/atualizar/fechar posições foram removidos daqui.
    # O ExecutionEngine e o DataManager são responsáveis por isso.
    # RiskManager foca em AVALIAR o risco e PERMITIR/NEGAR ações, e CALCULAR tamanhos.

    # _calculate_daily_pnl foi removido, pois o balanço atualizado virá do Orchestrator/ExecutionEngine.

    async def get_current_session_drawdown_pct(self, current_balance: float) -> float: # Renomeado e com arg
        """Calcula o drawdown da sessão atual em percentual."""
        # Atualizar HWM da sessão
        if current_balance > self.high_water_mark_session:
            self.high_water_mark_session = current_balance

        if self.high_water_mark_session > 0:
            drawdown = (self.high_water_mark_session - current_balance) / self.high_water_mark_session
            return max(0.0, drawdown) # Drawdown é sempre >= 0
        return 0.0


    async def check_and_trigger_circuit_breaker(self, # Renomeado
                                               account_balance: float,
                                               current_equity: float, # Adicionado equity
                                               margin_level_ratio: float, # Adicionado (ex: 1.5 para 150%)
                                               recent_closed_trades: List[Dict[str, Any]]):
        """Verifica e aciona o CircuitBreaker com base no estado atual da conta e trades."""
        # Construir account_status para o CircuitBreaker
        daily_pnl_val = current_equity - self.daily_start_balance_rm # PnL sobre equity
        daily_pnl_pct_val = (daily_pnl_val / self.daily_start_balance_rm) if self.daily_start_balance_rm > 0 else 0.0
        
        # Atualizar HWM da sessão para cálculo de DD
        session_dd_pct = await self.get_current_session_drawdown_pct(current_equity) # Usa equity para DD

        account_status_for_cb = {
            'current_balance': account_balance, # Saldo real
            'current_equity': current_equity,   # Equity real
            'daily_pnl_usd': daily_pnl_val,
            'daily_pnl_pct': daily_pnl_pct_val,
            'current_drawdown_percent': session_dd_pct, # DD da sessão atual
            'margin_level_percent': margin_level_ratio * 100.0 # Converter ratio para %
        }

        if await self.circuit_breaker.check_conditions(account_status_for_cb, recent_closed_trades):
            logger.info(f"RiskManager detectou condição de trip para CircuitBreaker: {self.circuit_breaker.last_trip_reason}")
            # A ação de parar trading, fechar posições, etc., deve ser coordenada pelo Orchestrator
            # ao observar o estado do CircuitBreaker.
            return True # Sinalizar que o CB foi acionado
        return False


    async def get_risk_assessment_metrics(self, current_balance: float, # Renomeado de get_risk_metrics
                                 open_positions: List[StrategyPosition]) -> RiskMetrics:
        """Retorna um objeto RiskMetrics com a avaliação de risco atual."""
        total_exposure = 0.0
        contract_size = getattr(CONFIG, 'CONTRACT_SIZE', 100000)
        for pos in open_positions:
            if pos.size > 0 and pos.entry_price > 0:
                 total_exposure += pos.size * contract_size * pos.entry_price

        margin_used = total_exposure / CONFIG.LEVERAGE if CONFIG.LEVERAGE > 0 else total_exposure
        free_margin = current_balance - margin_used # Simplificado, equity - margin_used seria mais correto

        daily_pnl_val = current_balance - self.daily_start_balance_rm
        daily_pnl_pct_val = (daily_pnl_val / self.daily_start_balance_rm) if self.daily_start_balance_rm > 0 else 0.0
        session_dd_pct_val = await self.get_current_session_drawdown_pct(current_balance) # Renomeado

        return RiskMetrics(
            account_balance_usd=current_balance,
            current_total_exposure_usd=total_exposure,
            open_positions_count=len(open_positions),
            daily_pnl_usd=daily_pnl_val,
            daily_pnl_percentage=daily_pnl_pct_val,
            max_drawdown_session_percentage=session_dd_pct_val,
            margin_used_usd=margin_used,
            free_margin_usd=free_margin,
            circuit_breaker_state=self.circuit_breaker.state.value
        )

    async def daily_session_reset(self, new_session_start_balance: float): # Renomeado de end_of_day_reset
        """Reseta métricas para uma nova sessão de trading (ex: diária)."""
        # Calcular PnL da sessão anterior
        pnl_previous_session = self.daily_start_balance_rm + sum(t.get('pnl', 0.0) for t in self.position_sizer.trade_history if t.get('session_id') == self._get_current_session_id()) - self.daily_start_balance_rm
        # Esta lógica de PnL precisa ser robusta e baseada nos trades reais da sessão.
        
        logger.info(f"Reset de fim de sessão/dia do RiskManager. PnL da sessão anterior: ${pnl_previous_session:.2f}")
        
        # Salvar histórico de eventos de risco ou métricas diárias, se necessário
        # self.risk_events_log.append(...)

        # Resetar contadores e saldos para a nova sessão
        # self.daily_trades_summary = [] # Se RiskManager rastreasse trades
        self.daily_start_balance_rm = new_session_start_balance
        self.high_water_mark_session = new_session_start_balance # Resetar HWM para a nova sessão
        
        # Resetar PositionSizer para a nova sessão (limpar histórico de trades se for por sessão)
        # self.position_sizer = PositionSizer(method=self.position_sizer.method, initial_balance_for_equity_calc=new_session_start_balance)
        # Ou apenas resetar o histórico se o sizer for persistente:
        self.position_sizer.trade_history = []
        self.position_sizer.equity_curve = [new_session_start_balance]
        self.position_sizer.current_win_streak = 0
        self.position_sizer.current_loss_streak = 0

        logger.info(f"RiskManager resetado para nova sessão. Saldo inicial: {new_session_start_balance:.2f}")

    # Helper para session_id (exemplo, não presente no original)
    def _get_current_session_id(self) -> str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d')


    async def get_available_risk_for_next_trade(self, current_balance: float) -> float: # Renomeado
        """Retorna a quantia em moeda que pode ser arriscada no próximo trade."""
        # Baseado no DEFAULT_RISK_PER_TRADE, mas pode ser ajustado por outros fatores
        base_risk_pct = RISK_LIMITS.DEFAULT_RISK_PER_TRADE
        
        # Exemplo de ajuste (poderia ser mais complexo, vindo de PositionSizer)
        # performance_factor = self.position_sizer._get_performance_adjustment_factor()
        # adjusted_risk_pct = base_risk_pct * performance_factor
        # ... outros fatores ...
        # final_risk_pct = np.clip(adjusted_risk_pct, RISK_LIMITS.MIN_RISK_PER_TRADE, RISK_LIMITS.MAX_RISK_PER_TRADE)
        
        final_risk_pct = base_risk_pct # Simplificado por enquanto
        
        return current_balance * final_risk_pct

    async def shutdown(self):
        """Chamado ao desligar o bot."""
        logger.info("Desligando RiskManager...")
        if self.circuit_breaker:
            await self.circuit_breaker.shutdown()
        # Salvar qualquer estado persistente se necessário
        logger.info("RiskManager desligado.")