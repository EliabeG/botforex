# risk/risk_manager.py
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from config.settings import CONFIG
from strategies.base_strategy import Signal, Position
from utils.logger import setup_logger

logger = setup_logger("risk_manager")

@dataclass
class RiskMetrics:
    """Métricas de risco atuais"""
    current_exposure: float
    open_positions: int
    daily_pnl: float
    daily_pnl_pct: float
    weekly_drawdown: float
    max_drawdown: float
    risk_available: float
    margin_used: float
    margin_available: float
    correlation_risk: float

class RiskManager:
    """Gerenciador central de risco"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.daily_trades = []
        self.risk_events = []
        self.correlation_matrix = {}
        self.max_correlation = 0.7  # Correlação máxima permitida
        self.high_water_mark = 0.0
        self.daily_start_balance = 0.0
        self.circuit_breaker_active = False
        self.last_circuit_breaker = None
        
    async def initialize(self, starting_balance: float):
        """Inicializa gerenciador de risco"""
        self.daily_start_balance = starting_balance
        self.high_water_mark = starting_balance
        logger.info(f"Risk Manager inicializado. Balanço inicial: {starting_balance}")
    
    async def can_open_position(self, signal: Signal) -> bool:
        """Verifica se pode abrir nova posição baseado em regras de risco"""
        try:
            # Circuit breaker ativo?
            if self.circuit_breaker_active:
                if await self._check_circuit_breaker_timeout():
                    self.circuit_breaker_active = False
                else:
                    logger.warning("Circuit breaker ativo - posição negada")
                    return False
            
            # Verificar limites básicos
            checks = [
                self._check_position_limit(),
                self._check_exposure_limit(),
                self._check_daily_loss_limit(),
                self._check_correlation_limit(signal),
                self._check_margin_requirements(signal),
                self._check_consecutive_losses(),
                self._check_time_restrictions()
            ]
            
            # Executar verificações em paralelo
            results = await asyncio.gather(*checks)
            
            # Todas as verificações devem passar
            can_trade = all(results)
            
            if not can_trade:
                logger.info(f"Posição negada para {signal.strategy_name}: "
                           f"Checks: {dict(zip(['positions', 'exposure', 'daily_loss', 
                                              'correlation', 'margin', 'consecutive', 'time'], results))}")
            
            return can_trade
            
        except Exception as e:
            logger.error(f"Erro ao verificar permissão de posição: {e}")
            return False  # Conservador - negar em caso de erro
    
    async def _check_position_limit(self) -> bool:
        """Verifica limite de posições simultâneas"""
        open_positions = len([p for p in self.positions.values() if p.size > 0])
        return open_positions < CONFIG.MAX_SIMULTANEOUS_ORDERS
    
    async def _check_exposure_limit(self) -> bool:
        """Verifica exposição total"""
        total_exposure = sum(p.size * p.entry_price for p in self.positions.values())
        max_exposure = self.daily_start_balance * 0.5  # Máximo 50% de exposição
        return total_exposure < max_exposure
    
    async def _check_daily_loss_limit(self) -> bool:
        """Verifica se limite de perda diária foi atingido"""
        daily_pnl = self._calculate_daily_pnl()
        daily_pnl_pct = daily_pnl / self.daily_start_balance
        
        if daily_pnl_pct <= -CONFIG.DAILY_LOSS_LIMIT:
            logger.warning(f"Limite de perda diária atingido: {daily_pnl_pct:.2%}")
            return False
        
        return True
    
    async def _check_correlation_limit(self, signal: Signal) -> bool:
        """Verifica correlação com posições existentes"""
        # Simplificado - em produção, calcular correlação real entre pares
        
        # Por enquanto, não permitir múltiplas posições na mesma direção
        same_direction_positions = [
            p for p in self.positions.values() 
            if p.side == signal.side and p.size > 0
        ]
        
        return len(same_direction_positions) < 2
    
    async def _check_margin_requirements(self, signal: Signal) -> bool:
        """Verifica requisitos de margem"""
        # Calcular margem necessária
        position_value = signal.position_size * signal.entry_price if signal.position_size else 100000
        required_margin = position_value / CONFIG.LEVERAGE
        
        # Margem já utilizada
        used_margin = sum(
            (p.size * p.entry_price) / CONFIG.LEVERAGE 
            for p in self.positions.values()
        )
        
        # Margem disponível
        available_margin = self.daily_start_balance - used_margin
        
        # Manter buffer de segurança (20% de margem livre)
        safety_buffer = self.daily_start_balance * 0.2
        
        return required_margin < (available_margin - safety_buffer)
    
    async def _check_consecutive_losses(self) -> bool:
        """Verifica perdas consecutivas"""
        if len(self.daily_trades) < 3:
            return True
        
        # Verificar últimas 3 trades
        last_trades = self.daily_trades[-3:]
        consecutive_losses = all(trade.get('pnl', 0) < 0 for trade in last_trades)
        
        if consecutive_losses:
            logger.warning("3 perdas consecutivas - aumentando cautela")
            # Poderia implementar pausa ou redução de tamanho
            return False
        
        return True
    
    async def _check_time_restrictions(self) -> bool:
        """Verifica restrições de horário"""
        now = datetime.now()
        
        # Não operar nos primeiros/últimos 5 minutos de cada sessão
        minute = now.minute
        hour = now.hour
        
        # Abertura Londres (8h)
        if hour == 8 and minute < 5:
            return False
        
        # Abertura NY (13h)
        if hour == 13 and minute < 5:
            return False
        
        # Fechamento Londres (16h)
        if hour == 15 and minute > 55:
            return False
        
        # Fechamento NY (21h)
        if hour == 20 and minute > 55:
            return False
        
        return True
    
    async def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Calcula tamanho da posição baseado em risco"""
        try:
            # Risco por trade
            risk_amount = account_balance * CONFIG.MAX_RISK_PER_TRADE
            
            # Ajustar risco baseado em performance recente
            risk_multiplier = await self._get_risk_multiplier()
            risk_amount *= risk_multiplier
            
            # Calcular distância do stop em pips
            if signal.entry_price and signal.stop_loss:
                stop_distance_pips = abs(signal.entry_price - signal.stop_loss) * 10000
            else:
                # Usar ATR médio como estimativa
                stop_distance_pips = 10  # Default conservador
            
            # Valor por pip (para EURUSD com conta em USD)
            pip_value = 10  # $10 por pip para lote padrão
            
            # Calcular tamanho do lote
            if stop_distance_pips > 0:
                lot_size = risk_amount / (stop_distance_pips * pip_value)
            else:
                lot_size = 0.01  # Mínimo
            
            # Aplicar limites
            lot_size = max(0.01, min(lot_size, self._get_max_lot_size()))
            
            # Arredondar para precisão do broker
            lot_size = round(lot_size, 2)
            
            logger.info(f"Tamanho calculado: {lot_size} lotes | "
                       f"Risco: ${risk_amount:.2f} | "
                       f"Stop: {stop_distance_pips:.1f} pips")
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Erro ao calcular tamanho da posição: {e}")
            return 0.01  # Retornar mínimo em caso de erro
    
    async def _get_risk_multiplier(self) -> float:
        """Obtém multiplicador de risco baseado em performance"""
        if not self.daily_trades:
            return 1.0
        
        # Calcular win rate recente (últimas 20 trades)
        recent_trades = self.daily_trades[-20:]
        wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0.5
        
        # Calcular Sharpe ratio simplificado
        returns = [t.get('pnl', 0) / self.daily_start_balance for t in recent_trades]
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Ajustar multiplicador
        multiplier = 1.0
        
        # Aumentar risco se performance boa
        if win_rate > 0.6 and sharpe > 1.5:
            multiplier = 1.25
        elif win_rate > 0.55 and sharpe > 1.0:
            multiplier = 1.1
        
        # Reduzir risco se performance ruim
        elif win_rate < 0.4 or sharpe < 0:
            multiplier = 0.5
        elif win_rate < 0.45 or sharpe < 0.5:
            multiplier = 0.75
        
        # Considerar drawdown atual
        current_dd = await self.get_current_drawdown()
        if current_dd > 0.1:  # DD > 10%
            multiplier *= 0.5
        elif current_dd > 0.05:  # DD > 5%
            multiplier *= 0.75
        
        return multiplier
    
    def _get_max_lot_size(self) -> float:
        """Retorna tamanho máximo de lote permitido"""
        # Baseado no balanço da conta
        if self.daily_start_balance < 10000:
            return 0.5
        elif self.daily_start_balance < 50000:
            return 1.0
        elif self.daily_start_balance < 100000:
            return 2.0
        else:
            return 5.0
    
    async def register_position(self, position: Position):
        """Registra nova posição aberta"""
        self.positions[position.id] = position
        logger.info(f"Posição registrada: {position.id} | "
                   f"{position.side} {position.size} @ {position.entry_price}")
    
    async def update_position(self, position_id: str, updates: Dict):
        """Atualiza informações da posição"""
        if position_id in self.positions:
            position = self.positions[position_id]
            for key, value in updates.items():
                setattr(position, key, value)
    
    async def close_position(self, position_id: str, exit_price: float, reason: str):
        """Registra fechamento de posição"""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # Calcular PnL
        if position.side == 'buy':
            pnl = (exit_price - position.entry_price) * position.size * 100000
        else:
            pnl = (position.entry_price - exit_price) * position.size * 100000
        
        # Registrar trade
        trade_record = {
            'position_id': position_id,
            'strategy': position.strategy_name,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'reason': reason,
            'duration': (datetime.now() - position.open_time).total_seconds(),
            'timestamp': datetime.now()
        }
        
        self.daily_trades.append(trade_record)
        
        # Atualizar high water mark
        current_balance = self.daily_start_balance + self._calculate_daily_pnl()
        if current_balance > self.high_water_mark:
            self.high_water_mark = current_balance
        
        # Remover posição
        del self.positions[position_id]
        
        logger.info(f"Posição fechada: {position_id} | PnL: ${pnl:.2f} | Razão: {reason}")
    
    def _calculate_daily_pnl(self) -> float:
        """Calcula PnL do dia"""
        return sum(trade.get('pnl', 0) for trade in self.daily_trades)
    
    async def get_current_drawdown(self) -> float:
        """Calcula drawdown atual"""
        current_balance = self.daily_start_balance + self._calculate_daily_pnl()
        
        # Adicionar PnL não realizado
        for position in self.positions.values():
            # Estimar PnL não realizado (simplificado)
            unrealized_pnl = position.pnl  # Assumindo que é atualizado externamente
            current_balance += unrealized_pnl
        
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - current_balance) / self.high_water_mark
            return max(0, drawdown)
        
        return 0
    
    async def check_circuit_breaker(self) -> bool:
        """Verifica se deve ativar circuit breaker"""
        # Drawdown máximo
        current_dd = await self.get_current_drawdown()
        if current_dd >= CONFIG.MAX_DRAWDOWN:
            await self._activate_circuit_breaker("Drawdown máximo atingido")
            return True
        
        # Perda diária
        daily_pnl_pct = self._calculate_daily_pnl() / self.daily_start_balance
        if daily_pnl_pct <= -CONFIG.DAILY_LOSS_LIMIT:
            await self._activate_circuit_breaker("Limite de perda diária")
            return True
        
        # Perdas consecutivas severas
        if len(self.daily_trades) >= 5:
            last_5_pnl = sum(t.get('pnl', 0) for t in self.daily_trades[-5:])
            if last_5_pnl < -self.daily_start_balance * 0.1:  # -10% em 5 trades
                await self._activate_circuit_breaker("Múltiplas perdas consecutivas")
                return True
        
        return False
    
    async def _activate_circuit_breaker(self, reason: str):
        """Ativa circuit breaker"""
        self.circuit_breaker_active = True
        self.last_circuit_breaker = datetime.now()
        
        logger.critical(f"CIRCUIT BREAKER ATIVADO: {reason}")
        
        # Registrar evento
        self.risk_events.append({
            'type': 'circuit_breaker',
            'reason': reason,
            'timestamp': datetime.now(),
            'drawdown': await self.get_current_drawdown(),
            'daily_pnl': self._calculate_daily_pnl()
        })
    
    async def _check_circuit_breaker_timeout(self) -> bool:
        """Verifica se circuit breaker pode ser desativado"""
        if not self.last_circuit_breaker:
            return True
        
        # 24 horas de pausa
        timeout = timedelta(hours=24)
        return datetime.now() - self.last_circuit_breaker > timeout
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Retorna métricas de risco atuais"""
        # Calcular exposição
        current_exposure = sum(
            p.size * p.entry_price for p in self.positions.values()
        )
        
        # Margem utilizada
        margin_used = current_exposure / CONFIG.LEVERAGE
        margin_available = self.daily_start_balance - margin_used
        
        # PnL diário
        daily_pnl = self._calculate_daily_pnl()
        daily_pnl_pct = daily_pnl / self.daily_start_balance
        
        # Drawdown
        current_dd = await self.get_current_drawdown()
        
        # Risco disponível
        risk_available = min(
            self.daily_start_balance * CONFIG.MAX_RISK_PER_TRADE,
            margin_available * 0.8  # 80% da margem disponível
        )
        
        # Risco de correlação (simplificado)
        correlation_risk = len(self.positions) / CONFIG.MAX_SIMULTANEOUS_ORDERS
        
        return RiskMetrics(
            current_exposure=current_exposure,
            open_positions=len(self.positions),
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_drawdown=current_dd,  # Simplificado
            max_drawdown=self.max_drawdown,
            risk_available=risk_available,
            margin_used=margin_used,
            margin_available=margin_available,
            correlation_risk=correlation_risk
        )
    
    async def end_of_day_reset(self):
        """Reset diário de métricas"""
        logger.info(f"Reset diário - PnL do dia: ${self._calculate_daily_pnl():.2f}")
        
        # Salvar histórico
        self.risk_events.append({
            'type': 'daily_summary',
            'date': datetime.now().date(),
            'pnl': self._calculate_daily_pnl(),
            'trades': len(self.daily_trades),
            'final_balance': self.daily_start_balance + self._calculate_daily_pnl()
        })
        
        # Reset
        self.daily_trades = []
        self.daily_start_balance = self.daily_start_balance + self._calculate_daily_pnl()
    
    async def get_available_risk(self) -> float:
        """Retorna quantidade de risco disponível para novas posições"""
        metrics = await self.get_risk_metrics()
        return metrics.risk_available