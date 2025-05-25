# core/execution_engine.py
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict

from config.settings import CONFIG
from api.ticktrader_ws import TickTraderTrade
from strategies.base_strategy import Position
from utils.logger import setup_logger

logger = setup_logger("execution_engine")

class OrderStatus(Enum):
    """Status de ordem"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Tipo de ordem"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class Order:
    """Estrutura de ordem"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.symbol = kwargs['symbol']
        self.side = kwargs['side']
        self.size = kwargs['size']
        self.type = kwargs.get('type', OrderType.MARKET)
        self.price = kwargs.get('price')
        self.stop_loss = kwargs.get('stop_loss')
        self.take_profit = kwargs.get('take_profit')
        self.strategy_name = kwargs.get('strategy_name')
        self.status = OrderStatus.PENDING
        self.created_at = datetime.now()
        self.submitted_at = None
        self.filled_at = None
        self.fill_price = None
        self.slippage = 0
        self.commission = 0
        self.metadata = kwargs.get('metadata', {})

class ExecutionEngine:
    """Motor de execução de ordens com otimização de latência"""
    
    def __init__(self, mode: str = "live"):
        self.mode = mode
        self.trade_client: Optional[TickTraderTrade] = None
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.execution_stats = defaultdict(lambda: {
            'total': 0,
            'filled': 0,
            'rejected': 0,
            'avg_latency': 0,
            'avg_slippage': 0
        })
        self.max_retries = 3
        self.retry_delay = 0.5
        
    async def initialize(self, trade_client: TickTraderTrade):
        """Inicializa motor de execução"""
        self.trade_client = trade_client
        
        # Registrar callbacks para eventos
        self.trade_client.register_callback('OrderUpdate', self._handle_order_update)
        self.trade_client.register_callback('PositionUpdate', self._handle_position_update)
        self.trade_client.register_callback('TradeUpdate', self._handle_trade_update)
        
        # Carregar posições existentes
        await self._load_existing_positions()
        
        logger.info(f"Motor de execução inicializado em modo {self.mode}")
    
    async def _load_existing_positions(self):
        """Carrega posições abertas existentes"""
        if self.mode == "live":
            positions = await self.trade_client.get_positions()
            for pos_data in positions:
                position = self._parse_position(pos_data)
                if position:
                    self.positions[position.id] = position
            
            logger.info(f"{len(self.positions)} posições existentes carregadas")
    
    async def create_order(self, symbol: str, side: str, size: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          strategy_name: str = None,
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None) -> Optional[Order]:
        """Cria e submete ordem"""
        try:
            # Criar objeto ordem
            order = Order(
                symbol=symbol,
                side=side,
                size=size,
                type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=strategy_name
            )
            
            # Validar ordem
            if not await self._validate_order(order):
                return None
            
            # Registrar ordem
            self.orders[order.id] = order
            
            # Submeter ordem
            success = await self._submit_order(order)
            
            if success:
                logger.info(f"Ordem criada: {order.id} | {side} {size} {symbol}")
                return order
            else:
                # Remover ordem se falhou
                del self.orders[order.id]
                return None
                
        except Exception as e:
            logger.error(f"Erro ao criar ordem: {e}")
            return None
    
    async def _validate_order(self, order: Order) -> bool:
        """Valida ordem antes de submeter"""
        # Verificar tamanho mínimo
        if order.size < 0.01:
            logger.error(f"Tamanho inválido: {order.size}")
            return False
        
        # Verificar side
        if order.side not in ['buy', 'sell']:
            logger.error(f"Side inválido: {order.side}")
            return False
        
        # Verificar stops para ordem market
        if order.type == OrderType.MARKET:
            if order.stop_loss and order.side == 'buy' and order.stop_loss >= order.price:
                logger.error("Stop loss inválido para compra")
                return False
            elif order.stop_loss and order.side == 'sell' and order.stop_loss <= order.price:
                logger.error("Stop loss inválido para venda")
                return False
        
        # Verificar símbolo
        if order.symbol != CONFIG.SYMBOL:
            logger.error(f"Símbolo não suportado: {order.symbol}")
            return False
        
        return True
    
    async def _submit_order(self, order: Order) -> bool:
        """Submete ordem ao broker com retry logic"""
        retries = 0
        
        while retries < self.max_retries:
            try:
                start_time = datetime.now()
                
                # Submeter baseado no tipo
                if order.type == OrderType.MARKET:
                    result = await self.trade_client.create_market_order(
                        symbol=order.symbol,
                        side=order.side,
                        volume=order.size
                    )
                else:  # LIMIT
                    result = await self.trade_client.create_limit_order(
                        symbol=order.symbol,
                        side=order.side,
                        volume=order.size,
                        price=order.price,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit
                    )
                
                # Calcular latência
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                if result:
                    order.status = OrderStatus.SUBMITTED
                    order.submitted_at = datetime.now()
                    
                    # Atualizar estatísticas
                    self._update_execution_stats(order.strategy_name, 'submitted', latency)
                    
                    logger.info(f"Ordem submetida com sucesso. Latência: {latency:.1f}ms")
                    return True
                else:
                    logger.warning(f"Falha ao submeter ordem (tentativa {retries + 1})")
                    
            except Exception as e:
                logger.error(f"Erro ao submeter ordem: {e}")
            
            retries += 1
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay * retries)
        
        # Falhou após todas tentativas
        order.status = OrderStatus.REJECTED
        self._update_execution_stats(order.strategy_name, 'rejected')
        return False
    
    async def modify_order(self, order_id: str, 
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> bool:
        """Modifica ordem existente"""
        if order_id not in self.orders:
            logger.error(f"Ordem não encontrada: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        try:
            success = await self.trade_client.modify_order(
                order_id=order_id,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if success:
                if stop_loss is not None:
                    order.stop_loss = stop_loss
                if take_profit is not None:
                    order.take_profit = take_profit
                
                logger.info(f"Ordem modificada: {order_id}")
                return True
                
        except Exception as e:
            logger.error(f"Erro ao modificar ordem: {e}")
        
        return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancela ordem pendente"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            logger.warning(f"Ordem {order_id} não pode ser cancelada (status: {order.status})")
            return False
        
        try:
            success = await self.trade_client.cancel_order(order_id)
            
            if success:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Ordem cancelada: {order_id}")
                return True
                
        except Exception as e:
            logger.error(f"Erro ao cancelar ordem: {e}")
        
        return False
    
    async def cancel_all_orders(self):
        """Cancela todas as ordens pendentes"""
        pending_orders = [
            o for o in self.orders.values() 
            if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
        ]
        
        results = await asyncio.gather(
            *[self.cancel_order(o.id) for o in pending_orders],
            return_exceptions=True
        )
        
        cancelled = sum(1 for r in results if r is True)
        logger.info(f"{cancelled} ordens canceladas")
    
    async def close_position(self, position_id: str, reason: str = "") -> bool:
        """Fecha posição específica"""
        if position_id not in self.positions:
            logger.error(f"Posição não encontrada: {position_id}")
            return False
        
        position = self.positions[position_id]
        
        try:
            success = await self.trade_client.close_position(position_id)
            
            if success:
                logger.info(f"Posição fechada: {position_id} | Razão: {reason}")
                del self.positions[position_id]
                return True
                
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")
        
        return False
    
    async def close_all_positions(self, reason: str = ""):
        """Fecha todas as posições abertas"""
        position_ids = list(self.positions.keys())
        
        results = await asyncio.gather(
            *[self.close_position(pid, reason) for pid in position_ids],
            return_exceptions=True
        )
        
        closed = sum(1 for r in results if r is True)
        logger.info(f"{closed} posições fechadas. Razão: {reason}")
    
    async def update_trailing_stop(self, position_id: str, current_price: float) -> bool:
        """Atualiza trailing stop de posição"""
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        
        # Calcular novo stop (simplificado - usar ATR em produção)
        distance = 0.0010  # 10 pips
        
        if position.side == 'buy':
            new_stop = current_price - distance
            if new_stop > position.stop_loss:
                return await self.modify_position_stops(
                    position_id,
                    stop_loss=new_stop
                )
        else:  # sell
            new_stop = current_price + distance
            if new_stop < position.stop_loss:
                return await self.modify_position_stops(
                    position_id,
                    stop_loss=new_stop
                )
        
        return False
    
    async def modify_position_stops(self, position_id: str,
                                   stop_loss: Optional[float] = None,
                                   take_profit: Optional[float] = None) -> bool:
        """Modifica stops de posição"""
        # Em TickTrader, modificamos a ordem associada
        # Implementação simplificada
        return await self.modify_order(position_id, stop_loss, take_profit)
    
    async def get_open_positions(self) -> List[Position]:
        """Retorna lista de posições abertas"""
        return list(self.positions.values())
    
    async def get_account_balance(self) -> float:
        """Obtém balanço atual da conta"""
        try:
            return await self.trade_client.get_account_balance()
        except Exception as e:
            logger.error(f"Erro ao obter balanço: {e}")
            return 0.0
    
    async def get_position_by_strategy(self, strategy_name: str) -> List[Position]:
        """Retorna posições de uma estratégia específica"""
        return [
            p for p in self.positions.values() 
            if p.strategy_name == strategy_name
        ]
    
    # Handlers de eventos
    
    async def _handle_order_update(self, data: Dict):
        """Processa atualização de ordem"""
        order_data = data.get('Order', {})
        order_id = order_data.get('Id')
        
        if order_id in self.orders:
            order = self.orders[order_id]
            
            # Atualizar status
            status = order_data.get('Status', '').lower()
            if status == 'filled':
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                order.fill_price = float(order_data.get('FilledPrice', 0))
                
                # Calcular slippage
                if order.type == OrderType.MARKET and order.price:
                    order.slippage = abs(order.fill_price - order.price) * 10000  # em pips
                
                # Criar posição
                await self._create_position_from_order(order, order_data)
                
                # Atualizar estatísticas
                latency = (order.filled_at - order.submitted_at).total_seconds() * 1000
                self._update_execution_stats(
                    order.strategy_name,
                    'filled',
                    latency,
                    order.slippage
                )
                
                logger.info(f"Ordem preenchida: {order_id} @ {order.fill_price} "
                           f"(slippage: {order.slippage:.1f} pips)")
                
            elif status == 'cancelled':
                order.status = OrderStatus.CANCELLED
            elif status == 'rejected':
                order.status = OrderStatus.REJECTED
                self._update_execution_stats(order.strategy_name, 'rejected')
    
    async def _handle_position_update(self, data: Dict):
        """Processa atualização de posição"""
        position_data = data.get('Position', {})
        position_id = position_data.get('Id')
        
        if position_id in self.positions:
            position = self.positions[position_id]
            
            # Atualizar PnL
            current_price = float(position_data.get('CurrentPrice', 0))
            if position.side == 'buy':
                position.pnl = (current_price - position.entry_price) * position.size * 100000
            else:
                position.pnl = (position.entry_price - current_price) * position.size * 100000
            
            # Verificar se posição foi fechada
            if position_data.get('Status') == 'Closed':
                del self.positions[position_id]
                logger.info(f"Posição fechada: {position_id} | PnL: ${position.pnl:.2f}")
    
    async def _handle_trade_update(self, data: Dict):
        """Processa execução de trade"""
        trade_data = data.get('Trade', {})
        order_id = trade_data.get('OrderId')
        
        if order_id in self.orders:
            order = self.orders[order_id]
            
            # Registrar detalhes da execução
            execution_price = float(trade_data.get('Price', 0))
            execution_size = float(trade_data.get('Volume', 0))
            
            logger.info(f"Trade executado: {execution_size} @ {execution_price} "
                       f"para ordem {order_id}")
    
    async def _create_position_from_order(self, order: Order, order_data: Dict):
        """Cria posição a partir de ordem preenchida"""
        position = Position(
            id=order.id,
            strategy_name=order.strategy_name,
            symbol=order.symbol,
            side=order.side,
            entry_price=order.fill_price,
            size=order.size,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            open_time=order.filled_at,
            metadata=order.metadata
        )
        
        self.positions[position.id] = position
        logger.info(f"Posição criada: {position.id}")
    
    def _parse_position(self, pos_data: Dict) -> Optional[Position]:
        """Converte dados do broker em objeto Position"""
        try:
            return Position(
                id=pos_data.get('Id'),
                strategy_name=pos_data.get('Comment', '').split('_')[0],
                symbol=pos_data.get('Symbol'),
                side='buy' if pos_data.get('Side') == 'Buy' else 'sell',
                entry_price=float(pos_data.get('Price', 0)),
                size=float(pos_data.get('Volume', 0)),
                stop_loss=float(pos_data.get('StopLoss', 0)),
                take_profit=float(pos_data.get('TakeProfit', 0)),
                open_time=datetime.fromtimestamp(pos_data.get('Created', 0) / 1000),
                pnl=float(pos_data.get('Profit', 0))
            )
        except Exception as e:
            logger.error(f"Erro ao parsear posição: {e}")
            return None
    
    def _update_execution_stats(self, strategy_name: str, event: str, 
                               latency: float = 0, slippage: float = 0):
        """Atualiza estatísticas de execução"""
        stats = self.execution_stats[strategy_name]
        
        stats['total'] += 1
        
        if event == 'filled':
            stats['filled'] += 1
            
            # Atualizar média de latência
            prev_avg_latency = stats['avg_latency']
            stats['avg_latency'] = (
                (prev_avg_latency * (stats['filled'] - 1) + latency) / 
                stats['filled']
            )
            
            # Atualizar média de slippage
            prev_avg_slippage = stats['avg_slippage']
            stats['avg_slippage'] = (
                (prev_avg_slippage * (stats['filled'] - 1) + slippage) / 
                stats['filled']
            )
            
        elif event == 'rejected':
            stats['rejected'] += 1
    
    def get_execution_stats(self, strategy_name: Optional[str] = None) -> Dict:
        """Retorna estatísticas de execução"""
        if strategy_name:
            return dict(self.execution_stats[strategy_name])
        
        # Agregar todas as estratégias
        total_stats = {
            'total': 0,
            'filled': 0,
            'rejected': 0,
            'avg_latency': 0,
            'avg_slippage': 0,
            'fill_rate': 0
        }
        
        for stats in self.execution_stats.values():
            total_stats['total'] += stats['total']
            total_stats['filled'] += stats['filled']
            total_stats['rejected'] += stats['rejected']
        
        if total_stats['filled'] > 0:
            # Média ponderada
            for strategy_name, stats in self.execution_stats.items():
                weight = stats['filled'] / total_stats['filled']
                total_stats['avg_latency'] += stats['avg_latency'] * weight
                total_stats['avg_slippage'] += stats['avg_slippage'] * weight
        
        if total_stats['total'] > 0:
            total_stats['fill_rate'] = total_stats['filled'] / total_stats['total']
        
        return total_stats
    
    async def check_order_timeout(self):
        """Verifica e cancela ordens com timeout"""
        current_time = datetime.now()
        timeout_duration = timedelta(milliseconds=CONFIG.ORDER_TIMEOUT_MS)
        
        for order in list(self.orders.values()):
            if (order.status == OrderStatus.SUBMITTED and 
                order.submitted_at and
                current_time - order.submitted_at > timeout_duration):
                
                logger.warning(f"Ordem {order.id} expirou por timeout")
                await self.cancel_order(order.id)
                order.status = OrderStatus.EXPIRED
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Retorna status de uma ordem"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    async def get_pending_orders(self) -> List[Order]:
        """Retorna ordens pendentes"""
        return [
            o for o in self.orders.values()
            if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
        ]
    
    def calculate_commission(self, size: float, price: float) -> float:
        """Calcula comissão estimada"""
        # TickTrader geralmente cobra por milhão negociado
        # Assumindo $35 por milhão
        volume = size * 100000  # Converter lotes para unidades
        commission_rate = 35 / 1000000  # $35 por milhão
        return volume * commission_rate
    
    async def emergency_close_all(self):
        """Fechamento de emergência de todas as posições"""
        logger.critical("FECHAMENTO DE EMERGÊNCIA ATIVADO")
        
        # Cancelar todas as ordens primeiro
        await self.cancel_all_orders()
        
        # Fechar todas as posições
        await self.close_all_positions("EMERGÊNCIA")
        
        logger.critical("Fechamento de emergência concluído")