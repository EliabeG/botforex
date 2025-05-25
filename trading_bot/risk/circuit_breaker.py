# risk/circuit_breaker.py
"""Sistema de circuit breaker para proteção de capital"""
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

from config.risk_config import RISK_LIMITS
from utils.logger import setup_logger

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
    RAPID_LOSSES = "rapid_losses"
    MARGIN_LEVEL = "margin_level"
    SYSTEM_ERROR = "system_error"
    MANUAL_TRIGGER = "manual_trigger"
    HIGH_VOLATILITY = "high_volatility"

class CircuitBreaker:
    """Circuit breaker para proteção contra perdas catastróficas"""
    
    def __init__(self):
        self.state = CircuitBreakerState.CLOSED
        self.trip_count = 0
        self.last_trip_time = None
        self.last_trip_reason = None
        self.trip_history = []
        self.reset_scheduled_time = None
        
        # Callbacks
        self.on_trip_callbacks = []
        self.on_reset_callbacks = []
        
        # Contadores
        self.consecutive_losses = 0
        self.rapid_loss_window = []
        self.daily_loss = 0.0
        self.current_drawdown = 0.0
        
        # Configurações
        self.pause_duration_hours = RISK_LIMITS.CIRCUIT_BREAKER_PAUSE_HOURS
        self.max_trips_per_week = 3
        self.half_open_test_trades = 3
        self.half_open_success_rate = 0.66
        
    async def check_conditions(self, 
                             account_status: Dict,
                             recent_trades: List[Dict]) -> bool:
        """
        Verifica condições para acionar circuit breaker
        
        Args:
            account_status: Status atual da conta
            recent_trades: Trades recentes
        
        Returns:
            True se circuit breaker deve ser acionado
        """
        if self.state == CircuitBreakerState.OPEN:
            return True
        
        # Verificar cada condição
        checks = [
            self._check_drawdown(account_status),
            self._check_daily_loss(account_status),
            self._check_consecutive_losses(recent_trades),
            self._check_rapid_losses(recent_trades),
            self._check_margin_level(account_status)
        ]
        
        # Executar verificações
        for check_result in checks:
            if check_result:
                reason, details = check_result
                await self.trip(reason, details)
                return True
        
        return False
    
    def _check_drawdown(self, account_status: Dict) -> Optional[tuple]:
        """Verifica drawdown máximo"""
        drawdown = account_status.get('current_drawdown', 0)
        
        if drawdown >= RISK_LIMITS.MAX_DRAWDOWN:
            return (
                TripReason.MAX_DRAWDOWN,
                {'drawdown': drawdown, 'limit': RISK_LIMITS.MAX_DRAWDOWN}
            )
        
        return None
    
    def _check_daily_loss(self, account_status: Dict) -> Optional[tuple]:
        """Verifica perda diária"""
        daily_pnl_pct = account_status.get('daily_pnl_pct', 0)
        
        if daily_pnl_pct <= -RISK_LIMITS.DAILY_LOSS_LIMIT:
            return (
                TripReason.DAILY_LOSS,
                {'daily_loss': daily_pnl_pct, 'limit': -RISK_LIMITS.DAILY_LOSS_LIMIT}
            )
        
        return None
    
    def _check_consecutive_losses(self, recent_trades: List[Dict]) -> Optional[tuple]:
        """Verifica perdas consecutivas"""
        if not recent_trades:
            return None
        
        consecutive = 0
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
        
        if consecutive >= RISK_LIMITS.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
            return (
                TripReason.CONSECUTIVE_LOSSES,
                {'consecutive': consecutive, 'limit': RISK_LIMITS.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES}
            )
        
        return None
    
    def _check_rapid_losses(self, recent_trades: List[Dict]) -> Optional[tuple]:
        """Verifica perdas rápidas em curto período"""
        if len(recent_trades) < 5:
            return None
        
        # Últimas 5 trades
        last_5_trades = recent_trades[-5:]
        total_pnl = sum(t.get('pnl', 0) for t in last_5_trades)
        
        # Assumir capital inicial de referência
        reference_capital = 10000  # Deve vir do account_status idealmente
        loss_pct = abs(total_pnl / reference_capital)
        
        if total_pnl < 0 and loss_pct >= RISK_LIMITS.CIRCUIT_BREAKER_LOSS_THRESHOLD:
            return (
                TripReason.RAPID_LOSSES,
                {
                    'trades': 5,
                    'total_loss': total_pnl,
                    'loss_pct': loss_pct,
                    'threshold': RISK_LIMITS.CIRCUIT_BREAKER_LOSS_THRESHOLD
                }
            )
        
        return None
    
    def _check_margin_level(self, account_status: Dict) -> Optional[tuple]:
        """Verifica nível de margem"""
        margin_level = account_status.get('margin_level', 1.0)
        
        if margin_level <= RISK_LIMITS.MARGIN_STOP_OUT:
            return (
                TripReason.MARGIN_LEVEL,
                {'margin_level': margin_level, 'stop_out': RISK_LIMITS.MARGIN_STOP_OUT}
            )
        
        return None
    
    async def trip(self, reason: TripReason, details: Dict = None):
        """
        Aciona o circuit breaker
        
        Args:
            reason: Razão do acionamento
            details: Detalhes adicionais
        """
        if self.state == CircuitBreakerState.OPEN:
            return  # Já está acionado
        
        self.state = CircuitBreakerState.OPEN
        self.last_trip_time = datetime.now()
        self.last_trip_reason = reason
        self.trip_count += 1
        
        # Registrar no histórico
        trip_record = {
            'timestamp': self.last_trip_time,
            'reason': reason.value,
            'details': details or {},
            'trip_number': self.trip_count
        }
        self.trip_history.append(trip_record)
        
        # Agendar reset
        self.reset_scheduled_time = self.last_trip_time + timedelta(hours=self.pause_duration_hours)
        
        logger.critical(f"CIRCUIT BREAKER ACIONADO! Razão: {reason.value}")
        logger.critical(f"Detalhes: {json.dumps(details, indent=2)}")
        logger.critical(f"Trading pausado até: {self.reset_scheduled_time}")
        
        # Executar callbacks
        for callback in self.on_trip_callbacks:
            try:
                await callback(reason, details)
            except Exception as e:
                logger.error(f"Erro em callback de trip: {e}")
        
        # Agendar reset automático
        asyncio.create_task(self._schedule_reset())
    
    async def _schedule_reset(self):
        """Agenda reset automático do circuit breaker"""
        wait_seconds = self.pause_duration_hours * 3600
        
        logger.info(f"Reset do circuit breaker agendado para {wait_seconds/3600:.1f} horas")
        
        await asyncio.sleep(wait_seconds)
        
        # Entrar em modo half-open para teste
        await self.enter_half_open()
    
    async def enter_half_open(self):
        """Entra em modo half-open para teste"""
        if self.state != CircuitBreakerState.OPEN:
            return
        
        self.state = CircuitBreakerState.HALF_OPEN
        logger.info("Circuit breaker em modo HALF-OPEN - testando recuperação")
        
        # Resetar contadores de teste
        self.half_open_test_count = 0
        self.half_open_success_count = 0
    
    async def test_half_open(self, trade_result: Dict) -> bool:
        """
        Testa operação em modo half-open
        
        Args:
            trade_result: Resultado do trade de teste
        
        Returns:
            True se deve continuar em half-open
        """
        if self.state != CircuitBreakerState.HALF_OPEN:
            return False
        
        self.half_open_test_count += 1
        
        if trade_result.get('pnl', 0) > 0:
            self.half_open_success_count += 1
        
        # Verificar se completou testes
        if self.half_open_test_count >= self.half_open_test_trades:
            success_rate = self.half_open_success_count / self.half_open_test_count
            
            if success_rate >= self.half_open_success_rate:
                # Sucesso - resetar circuit breaker
                await self.reset()
                return False
            else:
                # Falha - voltar para OPEN
                logger.warning(f"Teste half-open falhou. Success rate: {success_rate:.1%}")
                self.state = CircuitBreakerState.OPEN
                
                # Reagendar próximo teste
                await self._schedule_reset()
                return False
        
        return True  # Continuar testando
    
    async def reset(self):
        """Reseta o circuit breaker"""
        previous_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.reset_scheduled_time = None
        
        # Resetar contadores
        self.consecutive_losses = 0
        self.rapid_loss_window = []
        self.daily_loss = 0.0
        
        logger.info(f"Circuit breaker RESETADO. Estado anterior: {previous_state.value}")
        
        # Executar callbacks
        for callback in self.on_reset_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Erro em callback de reset: {e}")
    
    async def manual_trip(self, reason: str = "Manual intervention"):
        """Acionamento manual do circuit breaker"""
        await self.trip(
            TripReason.MANUAL_TRIGGER,
            {'reason': reason, 'triggered_by': 'user'}
        )
    
    def is_operational(self) -> bool:
        """Verifica se permite operações"""
        return self.state == CircuitBreakerState.CLOSED
    
    def is_testing(self) -> bool:
        """Verifica se está em modo de teste"""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def register_trip_callback(self, callback: Callable):
        """Registra callback para quando circuit breaker é acionado"""
        self.on_trip_callbacks.append(callback)
    
    def register_reset_callback(self, callback: Callable):
        """Registra callback para quando circuit breaker é resetado"""
        self.on_reset_callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """Retorna status atual do circuit breaker"""
        status = {
            'state': self.state.value,
            'operational': self.is_operational(),
            'trip_count': self.trip_count,
            'last_trip_time': self.last_trip_time.isoformat() if self.last_trip_time else None,
            'last_trip_reason': self.last_trip_reason.value if self.last_trip_reason else None,
            'reset_scheduled': self.reset_scheduled_time.isoformat() if self.reset_scheduled_time else None
        }
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            status['test_progress'] = {
                'tests_completed': self.half_open_test_count,
                'tests_required': self.half_open_test_trades,
                'success_count': self.half_open_success_count,
                'current_success_rate': (
                    self.half_open_success_count / self.half_open_test_count 
                    if self.half_open_test_count > 0 else 0
                )
            }
        
        return status
    
    def get_trip_history(self, limit: int = 10) -> List[Dict]:
        """Retorna histórico de acionamentos"""
        return self.trip_history[-limit:] if self.trip_history else []
    
    def check_weekly_limit(self) -> bool:
        """Verifica se atingiu limite semanal de trips"""
        if not self.trip_history:
            return False
        
        week_ago = datetime.now() - timedelta(days=7)
        recent_trips = [
            t for t in self.trip_history 
            if t['timestamp'] > week_ago
        ]
        
        return len(recent_trips) >= self.max_trips_per_week