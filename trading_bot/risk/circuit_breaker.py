# risk/circuit_breaker.py
"""Sistema de circuit breaker para proteção de capital"""
import asyncio
from typing import Dict, List, Optional, Callable, Any, Coroutine # Adicionado Coroutine e Any
from datetime import datetime, timedelta, timezone # Adicionado timezone
from enum import Enum
import json

from config.risk_config import RISK_LIMITS
from utils.logger import setup_logger
# from config.settings import CONFIG # Se precisar de configs globais

logger = setup_logger("circuit_breaker")

class CircuitBreakerState(Enum):
    """Estados do circuit breaker"""
    CLOSED = "closed"      # Operando normalmente
    OPEN = "open"          # Bloqueado - não permite operações
    HALF_OPEN = "half_open"  # Permitindo operações limitadas para teste

class TripReason(Enum):
    """Razões para acionamento do circuit breaker"""
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    RAPID_LOSSES = "rapid_losses" # Perdas significativas em poucas trades
    MARGIN_LEVEL = "margin_level_too_low" # Mais descritivo
    SYSTEM_ERROR = "internal_system_error" # Mais descritivo
    MANUAL_TRIGGER = "manual_trigger"
    HIGH_VOLATILITY_EXTERNAL = "high_volatility_external_event" # Ex: Notícia de alto impacto
    # Adicionar outras razões específicas se necessário

class CircuitBreaker:
    """Circuit breaker para proteção contra perdas catastróficas."""

    def __init__(self):
        self.state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self.trip_count: int = 0
        self.last_trip_time: Optional[datetime] = None
        self.last_trip_reason: Optional[TripReason] = None
        self.trip_history: List[Dict[str, Any]] = [] # Detalhes de cada trip
        self.reset_scheduled_time: Optional[datetime] = None # Quando o modo OPEN deve tentar ir para HALF_OPEN

        # Callbacks (lista de callables que podem ser corrotinas ou funções síncronas)
        self.on_trip_callbacks: List[Callable[[TripReason, Dict[str, Any]], Coroutine[Any, Any, None]]] = []
        self.on_reset_callbacks: List[Callable[[], Coroutine[Any, Any, None]]] = []
        self.on_half_open_callbacks: List[Callable[[], Coroutine[Any, Any, None]]] = [] # Novo callback


        # Contadores e estado para verificações de condições
        # Estes podem ser atualizados externamente ou por este módulo se ele tiver acesso aos dados.
        # O design original sugere que account_status e recent_trades são passados para check_conditions.
        # self.consecutive_losses = 0 # Não é estado interno, é calculado a partir de recent_trades
        # self.rapid_loss_window = [] # Idem
        # self.daily_loss = 0.0       # Idem, de account_status
        # self.current_drawdown = 0.0 # Idem, de account_status

        # Configurações (carregar de RISK_LIMITS ou CONFIG)
        self.pause_duration_hours: int = RISK_LIMITS.CIRCUIT_BREAKER_PAUSE_HOURS
        self.max_trips_per_week: int = getattr(RISK_LIMITS, 'CIRCUIT_BREAKER_MAX_TRIPS_WEEK', 3) # Default
        self.half_open_test_trades_limit: int = getattr(RISK_LIMITS, 'CB_HALF_OPEN_TEST_TRADES', 3) # Renomeado
        self.half_open_required_success_rate: float = getattr(RISK_LIMITS, 'CB_HALF_OPEN_SUCCESS_RATE', 0.66) # Renomeado

        # Estado para modo HALF_OPEN
        self._half_open_trades_done: int = 0
        self._half_open_successful_trades: int = 0
        self._reset_task: Optional[asyncio.Task] = None # Para gerenciar a tarefa de reset

    async def check_conditions(self,
                             account_status: Dict[str, Any], # Status atual da conta
                             recent_trades: List[Dict[str, Any]]) -> bool: # Trades recentes
        """
        Verifica condições para acionar circuit breaker.
        Retorna True se o circuit breaker ESTÁ ou DEVE SER acionado, False caso contrário.
        """
        if self.state == CircuitBreakerState.OPEN:
            # Se já estiver aberto, verificar se o tempo de pausa expirou para tentar HALF_OPEN
            if self.reset_scheduled_time and datetime.now(timezone.utc) >= self.reset_scheduled_time:
                await self.enter_half_open_mode() # Renomeado
            return True # Continua "acionado" (OPEN ou esperando para HALF_OPEN)

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Em HALF_OPEN, não re-aciona por estas condições,
            # o resultado dos trades de teste decidirá.
            return False # Não está "acionado" no sentido de bloquear novas verificações, mas está em teste.


        # Verificar se o limite semanal de trips foi atingido
        if self._check_weekly_trip_limit():
            logger.critical(f"Limite semanal de {self.max_trips_per_week} acionamentos do circuit breaker atingido. Trading suspenso até revisão manual ou fim da semana.")
            # Considerar um estado "PERMANENTLY_OPEN_FOR_WEEK" ou similar
            # Por enquanto, apenas aciona normalmente, mas o log indica o problema.
            await self.trip(TripReason.MANUAL_TRIGGER, {"detail": "Limite semanal de trips atingido."}) # Ou uma nova TripReason
            return True


        # Mapear verificações para razões
        condition_checks: List[Tuple[Callable, TripReason]] = [
            (lambda: self._check_drawdown(account_status), TripReason.MAX_DRAWDOWN),
            (lambda: self._check_daily_loss(account_status), TripReason.DAILY_LOSS),
            (lambda: self._check_consecutive_losses_from_list(recent_trades), TripReason.CONSECUTIVE_LOSSES), # Renomeado
            (lambda: self._check_rapid_losses_from_list(recent_trades, account_status.get('current_balance', CONFIG.INITIAL_BALANCE)), TripReason.RAPID_LOSSES), # Renomeado, passar balanço de referência
            (lambda: self._check_margin_level(account_status), TripReason.MARGIN_LEVEL)
        ]

        for check_func, reason_enum in condition_checks: # Renomeado
            details_or_none = check_func() # Renomeado
            if details_or_none: # Se a função de checagem retornar detalhes (não None)
                await self.trip(reason_enum, details_or_none)
                return True # Acionado

        return False # Nenhuma condição de trip encontrada, continua CLOSED

    def _check_drawdown(self, account_status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        drawdown_pct = account_status.get('current_drawdown_percent', 0.0) # Esperar em percentual (0.0 a 1.0)
        if drawdown_pct >= RISK_LIMITS.MAX_DRAWDOWN:
            return {'drawdown_percent': drawdown_pct, 'limit_percent': RISK_LIMITS.MAX_DRAWDOWN}
        return None

    def _check_daily_loss(self, account_status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        daily_pnl_pct_val = account_status.get('daily_pnl_pct', 0.0) # Esperar em percentual
        if daily_pnl_pct_val <= -RISK_LIMITS.DAILY_LOSS_LIMIT: # DAILY_LOSS_LIMIT é positivo
            return {'daily_loss_percent': daily_pnl_pct_val, 'limit_percent': -RISK_LIMITS.DAILY_LOSS_LIMIT}
        return None

    def _check_consecutive_losses_from_list(self, recent_trades_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]: # Renomeado
        if not recent_trades_list: return None
        consecutive_count = 0 # Renomeado
        for trade in reversed(recent_trades_list):
            if float(trade.get('pnl', 0.0)) < 0:
                consecutive_count += 1
            else:
                break
        if consecutive_count >= RISK_LIMITS.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
            return {'consecutive_losses': consecutive_count, 'limit': RISK_LIMITS.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES}
        return None

    def _check_rapid_losses_from_list(self, recent_trades_list: List[Dict[str, Any]], reference_capital: float) -> Optional[Dict[str, Any]]: # Renomeado
        num_trades_for_rapid_loss = getattr(RISK_LIMITS, 'CB_RAPID_LOSS_TRADES_WINDOW', 5) # Renomeado
        if not recent_trades_list or len(recent_trades_list) < num_trades_for_rapid_loss:
            return None

        last_n_trades = recent_trades_list[-num_trades_for_rapid_loss:]
        total_pnl_last_n = sum(float(t.get('pnl', 0.0)) for t in last_n_trades)

        if reference_capital <= 0: # Evitar divisão por zero
            logger.warning("Capital de referência inválido para checagem de perdas rápidas.")
            return None

        loss_pct_val = abs(total_pnl_last_n / reference_capital) if total_pnl_last_n < 0 else 0.0

        if total_pnl_last_n < 0 and loss_pct_val >= RISK_LIMITS.CIRCUIT_BREAKER_LOSS_THRESHOLD:
            return {
                'trades_window': num_trades_for_rapid_loss,
                'total_loss_value': total_pnl_last_n,
                'loss_percent_of_capital': loss_pct_val,
                'threshold_percent': RISK_LIMITS.CIRCUIT_BREAKER_LOSS_THRESHOLD
            }
        return None

    def _check_margin_level(self, account_status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        margin_level_pct_val = account_status.get('margin_level_percent', 1.0) # Esperar 0.0 a N.0 (ex: 1.0 para 100%)
        if margin_level_pct_val <= RISK_LIMITS.MARGIN_STOP_OUT: # MARGIN_STOP_OUT também 0.0 a 1.0
            return {'margin_level_percent': margin_level_pct_val, 'stop_out_level_percent': RISK_LIMITS.MARGIN_STOP_OUT}
        return None

    def _check_weekly_trip_limit(self) -> bool:
        """Verifica se o número de acionamentos na última semana excedeu o limite."""
        if not self.trip_history:
            return False
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        trips_in_last_week = sum(1 for trip_rec in self.trip_history if trip_rec['timestamp'] >= one_week_ago) # Renomeado
        return trips_in_last_week >= self.max_trips_per_week


    async def trip(self, reason: TripReason, details: Optional[Dict[str, Any]] = None): # details pode ser None
        if self.state == CircuitBreakerState.OPEN:
            logger.info(f"Circuit breaker já está OPEN. Novo motivo de trip ({reason.value}) ignorado por enquanto.")
            return

        self.state = CircuitBreakerState.OPEN
        self.last_trip_time = datetime.now(timezone.utc)
        self.last_trip_reason = reason
        self.trip_count += 1
        details = details or {} # Garantir que details seja um dict

        trip_record_data: Dict[str, Any] = { # Renomeado
            'timestamp': self.last_trip_time, # Já é datetime
            'reason': reason.value,
            'details': details,
            'trip_number_overall': self.trip_count
        }
        self.trip_history.append(trip_record_data)
        if len(self.trip_history) > 100: # Manter histórico dos últimos 100 trips
            self.trip_history = self.trip_history[-100:]


        self.reset_scheduled_time = self.last_trip_time + timedelta(hours=self.pause_duration_hours)

        logger.critical(f"CIRCUIT BREAKER ACIONADO! Razão: {reason.value}. Detalhes: {json.dumps(details, default=str)}")
        logger.critical(f"Trading pausado. Próxima tentativa de reset (HALF_OPEN) em: {self.reset_scheduled_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        for callback in self.on_trip_callbacks:
            try:
                await callback(reason, details)
            except Exception as e_cb_trip: # Renomeado
                logger.error(f"Erro em callback de trip do circuit breaker: {e_cb_trip}")

        # Agendar a transição para HALF_OPEN
        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel() # Cancelar tarefa de reset anterior se houver
        self._reset_task = asyncio.create_task(self._wait_for_reset_period())


    async def _wait_for_reset_period(self): # Renomeado de _schedule_reset
        """Aguarda o período de pausa e então tenta transitar para HALF_OPEN."""
        if not self.last_trip_time: return # Não deveria acontecer se tripado

        # Calcular tempo de espera restante
        now_utc = datetime.now(timezone.utc)
        # self.reset_scheduled_time já deve estar definido e em UTC
        if self.reset_scheduled_time and now_utc < self.reset_scheduled_time:
             wait_seconds = (self.reset_scheduled_time - now_utc).total_seconds()
             if wait_seconds > 0:
                logger.info(f"Circuit Breaker em modo OPEN. Aguardando {wait_seconds:.0f} segundos para tentar HALF_OPEN.")
                try:
                    await asyncio.sleep(wait_seconds)
                except asyncio.CancelledError:
                    logger.info("Espera para reset do Circuit Breaker cancelada.")
                    return # Sair se cancelado
        
        # Após a espera (ou se o tempo já passou), tentar entrar em HALF_OPEN
        if self.state == CircuitBreakerState.OPEN: # Checar novamente o estado
            await self.enter_half_open_mode()


    async def enter_half_open_mode(self): # Renomeado
        """Transita o circuit breaker para o estado HALF_OPEN."""
        if self.state != CircuitBreakerState.OPEN:
            logger.info(f"Tentativa de entrar em HALF_OPEN, mas estado atual é {self.state.value}. Ignorando.")
            return

        self.state = CircuitBreakerState.HALF_OPEN
        self._half_open_trades_done = 0
        self._half_open_successful_trades = 0
        logger.info("Circuit breaker agora em modo HALF-OPEN. Permitindo trades de teste limitados.")

        for callback in self.on_half_open_callbacks: # Chamar callbacks de half-open
            try:
                await callback()
            except Exception as e_cb_half: # Renomeado
                logger.error(f"Erro em callback de half_open do circuit breaker: {e_cb_half}")


    async def record_half_open_trade_result(self, trade_is_successful: bool): # Renomeado
        """Registra o resultado de um trade de teste em modo HALF_OPEN."""
        if self.state != CircuitBreakerState.HALF_OPEN:
            logger.warning("Resultado de trade recebido, mas circuit breaker não está em HALF_OPEN.")
            return

        self._half_open_trades_done += 1
        if trade_is_successful:
            self._half_open_successful_trades += 1

        logger.info(f"Trade de teste em HALF_OPEN: {'Sucesso' if trade_is_successful else 'Falha'}. "
                   f"Contagem: {self._half_open_successful_trades}/{self._half_open_trades_done} (Meta: {self.half_open_required_success_rate*100:.0f}% de {self.half_open_test_trades_limit} trades)")


        # Verificar se o número de trades de teste foi atingido
        if self._half_open_trades_done >= self.half_open_test_trades_limit:
            current_success_rate = self._half_open_successful_trades / self._half_open_trades_done
            if current_success_rate >= self.half_open_required_success_rate:
                logger.info(f"Trades de teste em HALF_OPEN bem-sucedidos (Taxa: {current_success_rate:.2%}). Resetando circuit breaker.")
                await self.reset_circuit_breaker() # Renomeado
            else:
                logger.warning(f"Trades de teste em HALF_OPEN falharam (Taxa: {current_success_rate:.2%}). Voltando para OPEN.")
                # Re-trip, mas com um detalhe indicando falha no teste HALF_OPEN
                await self.trip(self.last_trip_reason or TripReason.SYSTEM_ERROR, # Usar a razão anterior ou uma nova
                                {'detail': 'Falha nos trades de teste em modo HALF_OPEN.'})


    async def reset_circuit_breaker(self): # Renomeado de reset
        """Reseta o circuit breaker para o estado CLOSED."""
        if self.state == CircuitBreakerState.CLOSED:
            logger.info("Circuit breaker já está CLOSED. Nenhuma ação de reset necessária.")
            return

        previous_state_val = self.state.value # Renomeado
        self.state = CircuitBreakerState.CLOSED
        self.reset_scheduled_time = None
        # Contadores de trip (trip_count, last_trip_time, last_trip_reason) são mantidos para histórico, não resetados aqui.
        # Contadores de condições (como consecutive_losses) são resetados pelo RiskManager ou Orchestrator.

        logger.info(f"CIRCUIT BREAKER RESETADO. Estado anterior: {previous_state_val}. Trading pode ser retomado.")

        for callback in self.on_reset_callbacks:
            try:
                await callback()
            except Exception as e_cb_reset: # Renomeado
                logger.error(f"Erro em callback de reset do circuit breaker: {e_cb_reset}")


    async def force_trip_manual(self, reason_text: str = "Intervenção manual do operador"): # Renomeado
        """Acionamento manual do circuit breaker."""
        logger.warning(f"ACIONAMENTO MANUAL DO CIRCUIT BREAKER solicitado. Razão: {reason_text}")
        await self.trip(TripReason.MANUAL_TRIGGER, {'manual_reason': reason_text, 'triggered_by': 'operator'})

    def is_trading_allowed(self) -> bool: # Renomeado de is_operational
        """Verifica se o estado atual permite operações de trading normais."""
        return self.state == CircuitBreakerState.CLOSED

    def is_in_half_open_test(self) -> bool: # Renomeado de is_testing
        """Verifica se está em modo de teste HALF_OPEN."""
        return self.state == CircuitBreakerState.HALF_OPEN


    def add_on_trip_callback(self, callback: Callable[[TripReason, Dict[str, Any]], Coroutine[Any, Any, None]]): # Renomeado
        """Registra callback para quando circuit breaker é acionado."""
        self.on_trip_callbacks.append(callback)

    def add_on_reset_callback(self, callback: Callable[[], Coroutine[Any, Any, None]]): # Renomeado
        """Registra callback para quando circuit breaker é resetado."""
        self.on_reset_callbacks.append(callback)

    def add_on_half_open_callback(self, callback: Callable[[], Coroutine[Any, Any, None]]): # Novo
        """Registra callback para quando entra em modo HALF_OPEN."""
        self.on_half_open_callbacks.append(callback)


    def get_current_status(self) -> Dict[str, Any]: # Renomeado
        """Retorna status atual do circuit breaker."""
        status_dict: Dict[str, Any] = { # Renomeado
            'current_state': self.state.value,
            'is_trading_allowed': self.is_trading_allowed(),
            'is_in_half_open_test': self.is_in_half_open_test(),
            'overall_trip_count': self.trip_count,
            'last_trip_timestamp_utc': self.last_trip_time.isoformat() if self.last_trip_time else None,
            'last_trip_reason_value': self.last_trip_reason.value if self.last_trip_reason else None,
            'next_reset_attempt_utc': self.reset_scheduled_time.isoformat() if self.reset_scheduled_time else None
        }

        if self.state == CircuitBreakerState.HALF_OPEN:
            status_dict['half_open_test_progress'] = {
                'trades_completed_in_test': self._half_open_trades_done,
                'test_trades_target': self.half_open_test_trades_limit,
                'successful_test_trades': self._half_open_successful_trades,
                'current_test_success_rate': (
                    (self._half_open_successful_trades / self._half_open_trades_done)
                    if self._half_open_trades_done > 0 else 0.0
                ),
                'required_success_rate_for_reset': self.half_open_required_success_rate
            }
        return status_dict

    def get_trip_history_list(self, limit_records: int = 10) -> List[Dict[str, Any]]: # Renomeado e com limit
        """Retorna histórico de acionamentos (os mais recentes)."""
        # Converter datetimes para string no retorno para serialização JSON fácil
        history_to_return = []
        for record in self.trip_history[-limit_records:]:
            rec_copy = record.copy()
            if isinstance(rec_copy.get('timestamp'), datetime):
                rec_copy['timestamp'] = rec_copy['timestamp'].isoformat()
            history_to_return.append(rec_copy)
        return history_to_return

    async def shutdown(self):
        """Chamado ao desligar o bot para cancelar tarefas pendentes."""
        logger.info("Desligando CircuitBreaker...")
        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel()
            try:
                await self._reset_task
            except asyncio.CancelledError:
                logger.info("Tarefa de reset do CircuitBreaker cancelada durante o shutdown.")
        logger.info("CircuitBreaker desligado.")