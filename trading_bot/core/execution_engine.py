# core/execution_engine.py
import asyncio
from typing import Dict, List, Optional, Tuple, Any # Adicionado Any
from datetime import datetime, timedelta, timezone # Adicionado timezone
from enum import Enum
import uuid
from collections import defaultdict

from config.settings import CONFIG
from api.ticktrader_ws import TickTraderTrade # Supondo que esta é a interface correta
from strategies.base_strategy import Position # Supondo que a dataclass Position venha daqui
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
    CANCELLATION_PENDING = "cancellation_pending" # Adicionado
    MODIFICATION_PENDING = "modification_pending" # Adicionado


class OrderType(Enum):
    """Tipo de ordem"""
    MARKET = "Market" # Usar capitalização exata esperada pela API TickTrader
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit" # Adicionado


class Order:
    """Estrutura de ordem"""
    def __init__(self, **kwargs: Any): # Adicionada tipagem para kwargs
        self.id: str = kwargs.get('id', str(uuid.uuid4()))
        self.client_order_id: str = self.id # Manter um ID do cliente separado, se necessário
        self.broker_order_id: Optional[str] = kwargs.get('broker_order_id')

        self.symbol: str = kwargs['symbol']
        self.side: str = kwargs['side'] # 'Buy' ou 'Sell'
        self.size: float = float(kwargs['size']) # Garantir float
        self.type: OrderType = kwargs.get('type', OrderType.MARKET)

        self.price: Optional[float] = float(kwargs['price']) if kwargs.get('price') is not None else None
        self.stop_loss: Optional[float] = float(kwargs['stop_loss']) if kwargs.get('stop_loss') is not None else None
        self.take_profit: Optional[float] = float(kwargs['take_profit']) if kwargs.get('take_profit') is not None else None

        self.strategy_name: Optional[str] = kwargs.get('strategy_name')
        self.status: OrderStatus = kwargs.get('status', OrderStatus.PENDING) # Permitir definir status inicial
        self.created_at: datetime = kwargs.get('created_at', datetime.now(timezone.utc)) # Usar UTC

        self.submitted_at: Optional[datetime] = kwargs.get('submitted_at')
        self.filled_at: Optional[datetime] = kwargs.get('filled_at')
        self.cancelled_at: Optional[datetime] = kwargs.get('cancelled_at') # Adicionado

        self.fill_price: Optional[float] = float(kwargs.get('fill_price')) if kwargs.get('fill_price') is not None else None
        self.filled_size: float = float(kwargs.get('filled_size', 0.0)) # Adicionado
        self.remaining_size: float = self.size - self.filled_size # Adicionado

        self.slippage_pips: float = float(kwargs.get('slippage_pips', 0.0)) # Renomeado e float
        self.commission: float = float(kwargs.get('commission', 0.0))
        self.metadata: Dict[str, Any] = kwargs.get('metadata', {})
        self.error_message: Optional[str] = kwargs.get('error_message') # Para rejeições


class ExecutionEngine:
    """Motor de execução de ordens com otimização de latência"""

    def __init__(self, mode: str = "live"):
        self.mode = mode
        self.trade_client: Optional[TickTraderTrade] = None
        self.orders: Dict[str, Order] = {} # Armazena por client_order_id
        self.broker_to_client_order_map: Dict[str, str] = {} # Mapeia broker_order_id para client_order_id
        self.positions: Dict[str, Position] = {} # Armazena por position_id do broker
        self.execution_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: { # Tipagem mais precisa
            'total_requests': 0, # Renomeado de 'total'
            'submitted_ok': 0, # Renomeado de 'filled' para refletir submissão
            'rejected_by_broker': 0, # Renomeado de 'rejected'
            'filled_trades': 0, # Novo, para execuções reais
            'avg_submission_latency_ms': 0.0, # Renomeado
            'avg_fill_latency_ms': 0.0, # Novo
            'avg_slippage_pips': 0.0 # Renomeado
        })
        self.max_retries: int = CONFIG.ORDER_TIMEOUT_MS // 1000 // 2 if CONFIG.ORDER_TIMEOUT_MS > 2000 else 3 # Ajustado
        self.retry_delay_seconds: float = 0.5 # Renomeado

    async def initialize(self, trade_client: TickTraderTrade):
        """Inicializa motor de execução"""
        self.trade_client = trade_client
        if not self.trade_client:
            logger.error("Trade client não fornecido para ExecutionEngine.")
            raise ValueError("Trade client é obrigatório para ExecutionEngine.")


        # Registrar callbacks para eventos (os nomes dos eventos devem corresponder aos da API)
        # Ex: 'ExecutionReport', 'OrderCancelReject', etc.
        # A estrutura original usa 'OrderUpdate', 'PositionUpdate', 'TradeUpdate'
        self.trade_client.register_callback('OrderUpdate', self._handle_order_update_event) # Renomeado
        self.trade_client.register_callback('PositionUpdate', self._handle_position_update_event) # Renomeado
        # 'TradeUpdate' pode ser um evento de fill, que já seria coberto por ExecutionReport/OrderUpdate
        # self.trade_client.register_callback('TradeUpdate', self._handle_trade_update_event)


        await self._load_existing_positions_and_orders()
        logger.info(f"Motor de execução inicializado em modo {self.mode}")

    async def _load_existing_positions_and_orders(self):
        """Carrega posições e ordens abertas/pendentes existentes do broker."""
        if self.mode == "live" and self.trade_client and self.trade_client.is_connected():
            try:
                broker_positions = await self.trade_client.get_positions() # Assume que retorna lista de dicts
                for pos_data in broker_positions:
                    position = self._parse_broker_position_data(pos_data) # Método para converter
                    if position:
                        self.positions[position.id] = position # Assumindo que position.id é o ID do broker
                logger.info(f"{len(self.positions)} posições existentes carregadas do broker.")

                # Carregar ordens pendentes (se a API suportar GetOpenOrders)
                # broker_orders = await self.trade_client.get_open_orders()
                # for order_data in broker_orders:
                #     order = self._parse_broker_order_data(order_data)
                #     if order:
                #         self.orders[order.client_order_id] = order
                #         if order.broker_order_id:
                #             self.broker_to_client_order_map[order.broker_order_id] = order.client_order_id
                # logger.info(f"{len(self.orders)} ordens pendentes carregadas do broker.")

            except Exception as e:
                logger.exception("Erro ao carregar posições/ordens existentes do broker:")
        else:
            logger.info("Rodando em modo não-live ou cliente não conectado, não carregando posições/ordens do broker.")


    async def create_order(self, symbol: str, side: str, size: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          strategy_name: Optional[str] = None, # Tornar opcional
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None,
                          client_order_id: Optional[str] = None) -> Optional[Order]: # Adicionado client_order_id
        """Cria e submete ordem"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.error("Trade client não conectado. Não é possível criar ordem.")
            return None

        order_id_to_use = client_order_id or str(uuid.uuid4())
        order = Order(
            id=order_id_to_use, # Usar o ID do cliente
            symbol=symbol,
            side=side.capitalize(), # Garantir capitalização para API
            size=size,
            type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            created_at=datetime.now(timezone.utc)
        )

        if not await self._validate_order(order, price if order_type != OrderType.MARKET else await self._get_current_market_price(symbol, side)):
             order.status = OrderStatus.REJECTED
             order.error_message = "Validação local falhou."
             self.orders[order.id] = order # Registrar mesmo se rejeitada localmente
             return order # Retornar a ordem com status de rejeição


        self.orders[order.id] = order # Registrar antes de submeter

        success, broker_response = await self._submit_order_to_broker(order)

        if success and broker_response:
            order.submitted_at = datetime.now(timezone.utc)
            # O broker_order_id e o status inicial (ex: PENDING_NEW, NEW) virão da resposta.
            # _handle_order_update_event deve processar isso.
            # Se a submissão for síncrona e já retornar o broker_order_id:
            if broker_response.get('Id'): # Supondo que 'Id' é o broker_order_id
                order.broker_order_id = broker_response['Id']
                self.broker_to_client_order_map[order.broker_order_id] = order.id
            current_broker_status_str = broker_response.get('Status', 'Submitted').lower() # Exemplo
            try:
                order.status = OrderStatus(current_broker_status_str)
            except ValueError:
                logger.warning(f"Status de ordem desconhecido do broker '{current_broker_status_str}' para ordem {order.id}. Marcando como SUBMITTED.")
                order.status = OrderStatus.SUBMITTED

            logger.info(f"Ordem {order.id} (Broker: {order.broker_order_id}) submetida: {side} {size} {symbol} {order_type.value} @ {price if price else 'Market'}")
        else:
            order.status = OrderStatus.REJECTED
            order.error_message = broker_response.get('Error', {}).get('Message', "Falha na submissão ao broker.") if broker_response else "Falha na submissão ao broker."
            # Não remover de self.orders, deixar _handle_order_update_event (se houver) ou timeout tratar.
            logger.error(f"Falha ao submeter ordem {order.id} ao broker: {order.error_message}")


        return order


    async def _validate_order(self, order: Order, current_market_price: Optional[float]) -> bool:
        """Valida ordem antes de submeter"""
        if order.size < CONFIG.SYMBOL_SPECIFICATIONS.get(order.symbol, {}).get('min_lot', 0.01): # Usar de CONFIG
            order.error_message = f"Tamanho da ordem {order.size} é menor que o mínimo permitido."
            logger.error(f"Ordem {order.id}: {order.error_message}")
            return False
        if order.side.capitalize() not in ['Buy', 'Sell']:
            order.error_message = f"Lado da ordem inválido: {order.side}."
            logger.error(f"Ordem {order.id}: {order.error_message}")
            return False

        if order.type != OrderType.MARKET and order.price is None:
            order.error_message = f"Preço necessário para ordens {order.type.value}."
            logger.error(f"Ordem {order.id}: {order.error_message}")
            return False


        # Validação de SL/TP em relação ao preço de mercado/limite
        if current_market_price is not None: # current_market_price é o preço de referência
            min_stop_pips = CONFIG.RISK_PARAMS.MIN_STOP_DISTANCE_PIPS if hasattr(CONFIG, 'RISK_PARAMS') else 5.0
            pip_size = 0.0001 # Assumindo par XXX/YYY, ajustar para JPY
            min_stop_distance = min_stop_pips * pip_size


            if order.stop_loss is not None:
                if order.side.capitalize() == 'Buy':
                    if order.stop_loss >= current_market_price:
                        order.error_message = f"Stop loss ({order.stop_loss}) deve ser menor que o preço de entrada/mercado ({current_market_price}) para compra."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False
                    if current_market_price - order.stop_loss < min_stop_distance:
                        order.error_message = f"Distância do Stop Loss ({current_market_price - order.stop_loss:.5f}) muito curta. Mínimo: {min_stop_distance:.5f}."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False
                elif order.side.capitalize() == 'Sell':
                    if order.stop_loss <= current_market_price:
                        order.error_message = f"Stop loss ({order.stop_loss}) deve ser maior que o preço de entrada/mercado ({current_market_price}) para venda."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False
                    if order.stop_loss - current_market_price < min_stop_distance:
                        order.error_message = f"Distância do Stop Loss ({order.stop_loss - current_market_price:.5f}) muito curta. Mínimo: {min_stop_distance:.5f}."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False

            if order.take_profit is not None: # Validação similar para Take Profit
                min_tp_distance = min_stop_pips * pip_size # Assumindo mesma distância mínima para TP
                if order.side.capitalize() == 'Buy':
                    if order.take_profit <= current_market_price:
                        order.error_message = f"Take profit ({order.take_profit}) deve ser maior que o preço de entrada/mercado ({current_market_price}) para compra."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False
                    if order.take_profit - current_market_price < min_tp_distance:
                         order.error_message = f"Distância do Take Profit ({order.take_profit - current_market_price:.5f}) muito curta. Mínimo: {min_tp_distance:.5f}."
                         logger.error(f"Ordem {order.id}: {order.error_message}")
                         return False
                elif order.side.capitalize() == 'Sell':
                    if order.take_profit >= current_market_price:
                        order.error_message = f"Take profit ({order.take_profit}) deve ser menor que o preço de entrada/mercado ({current_market_price}) para venda."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False
                    if current_market_price - order.take_profit < min_tp_distance:
                        order.error_message = f"Distância do Take Profit ({current_market_price - order.take_profit:.5f}) muito curta. Mínimo: {min_tp_distance:.5f}."
                        logger.error(f"Ordem {order.id}: {order.error_message}")
                        return False

        if order.symbol != CONFIG.SYMBOL: # Permitir múltiplos símbolos se CONFIG for ajustado
            order.error_message = f"Símbolo {order.symbol} não é o símbolo configurado ({CONFIG.SYMBOL})."
            logger.error(f"Ordem {order.id}: {order.error_message}")
            return False

        return True

    async def _get_current_market_price(self, symbol: str, side: str) -> Optional[float]:
        """Obtém o preço de mercado relevante (ask para compra, bid para venda)."""
        # Idealmente, buscaria do DataManager ou FeedClient
        # Mock simples por agora:
        # last_tick = await data_manager.get_latest_tick(symbol) # Se DataManager estiver acessível
        # if last_tick:
        #     return last_tick.ask if side.capitalize() == 'Buy' else last_tick.bid
        logger.warning("_get_current_market_price não implementado para buscar preço real, usando placeholder.")
        return 1.08500 # Placeholder


    async def _submit_order_to_broker(self, order: Order) -> Tuple[bool, Optional[Dict[str,Any]]]:
        """Submete ordem ao broker com retry logic"""
        if not self.trade_client:
            logger.error("Trade client não está disponível para submeter ordem.")
            return False, {"Error": {"Message": "Trade client indisponível."}}

        retries = 0
        last_exception: Optional[Exception] = None

        while retries < self.max_retries:
            try:
                start_time_submission = datetime.now(timezone.utc) # Usar UTC
                broker_response: Optional[Dict[str, Any]] = None # Tipagem

                if order.type == OrderType.MARKET:
                    broker_response = await self.trade_client.create_market_order(
                        symbol=order.symbol,
                        side=order.side,
                        volume=order.size
                        # O strategy_comment pode ser passado aqui se a API suportar
                    )
                elif order.type == OrderType.LIMIT:
                    broker_response = await self.trade_client.create_limit_order(
                        symbol=order.symbol,
                        side=order.side,
                        volume=order.size,
                        price=order.price, # order.price não pode ser None para Limit
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit
                    )
                # Adicionar outros tipos de ordem (STOP, STOP_LIMIT) se necessário

                else:
                    logger.error(f"Tipo de ordem desconhecido: {order.type} para ordem {order.id}")
                    return False, {"Error": {"Message": f"Tipo de ordem desconhecido: {order.type}"}}


                latency_ms = (datetime.now(timezone.utc) - start_time_submission).total_seconds() * 1000

                # A resposta do broker para submissão pode variar.
                # Alguns confirmam apenas o recebimento, outros já dão o status.
                # Assumindo que `broker_response` contém o resultado da submissão.
                if broker_response and not broker_response.get('Error'): # Checar se há um campo de erro explícito
                    # O ID da ordem do broker pode vir aqui ou em um evento subsequente
                    broker_order_id = broker_response.get('Result', {}).get('Id')
                    if broker_order_id:
                        order.broker_order_id = str(broker_order_id)
                        self.broker_to_client_order_map[order.broker_order_id] = order.id
                    # O status também pode vir aqui.
                    # order.status = OrderStatus.SUBMITTED # Ou o status retornado pelo broker

                    self._update_execution_stats(order.strategy_name, 'submitted_ok', submission_latency_ms=latency_ms)
                    logger.info(f"Ordem {order.id} (Broker: {order.broker_order_id}) submetida. Latência de submissão: {latency_ms:.1f}ms")
                    return True, broker_response.get('Result') # Retornar o Result para mais detalhes
                else:
                    error_msg = broker_response.get('Error', {}).get('Message', 'Falha desconhecida na submissão') if broker_response else 'Sem resposta do broker'
                    logger.warning(f"Falha ao submeter ordem {order.id} (tentativa {retries + 1}/{self.max_retries}): {error_msg}")
                    # Não atualizar stats de rejeição aqui, pois pode ser um problema temporário.
                    # A rejeição final deve vir de um evento do broker.

            except ConnectionError as ce: # Exceção específica de conexão
                last_exception = ce
                logger.error(f"Erro de conexão ao submeter ordem {order.id}: {ce}")
            except asyncio.TimeoutError: # Se _send_and_wait tiver timeout
                last_exception = asyncio.TimeoutError("Timeout na submissão da ordem")
                logger.error(f"Timeout ao submeter ordem {order.id} (tentativa {retries + 1})")
            except Exception as e:
                last_exception = e
                logger.exception(f"Erro inesperado ao submeter ordem {order.id} (tentativa {retries + 1}):")


            retries += 1
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay_seconds * (2 ** (retries -1))) # Backoff exponencial

        # Falhou após todas as tentativas
        final_error_msg = f"Falha na submissão da ordem {order.id} após {self.max_retries} tentativas. Último erro: {last_exception or 'Desconhecido'}"
        logger.error(final_error_msg)
        order.status = OrderStatus.REJECTED # Marcar como rejeitada se todas as tentativas falharem
        order.error_message = final_error_msg
        self._update_execution_stats(order.strategy_name, 'rejected_by_broker')
        return False, {"Error": {"Message": final_error_msg}}


    async def modify_order(self, client_order_id: str, # Usar client_order_id
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          new_price: Optional[float] = None, # Para ordens limite
                          new_volume: Optional[float] = None) -> bool:
        """Modifica ordem existente"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.error("Trade client não conectado. Não é possível modificar ordem.")
            return False

        order_to_modify = self.orders.get(client_order_id)
        if not order_to_modify:
            logger.error(f"Ordem (Cliente ID: {client_order_id}) não encontrada para modificação.")
            return False

        if order_to_modify.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.warning(f"Ordem {client_order_id} não pode ser modificada (status: {order_to_modify.status.value})")
            return False

        # Broker ID é necessário para modificação
        broker_id_to_modify = order_to_modify.broker_order_id
        if not broker_id_to_modify:
            logger.error(f"Broker Order ID não encontrado para a ordem cliente {client_order_id}. Modificação não pode prosseguir.")
            return False

        # Salvar estado anterior para reverter em caso de falha na API, se necessário
        # original_status = order_to_modify.status
        # order_to_modify.status = OrderStatus.MODIFICATION_PENDING # Atualizar status local

        success = await self.trade_client.modify_order(
            order_id=broker_id_to_modify, # Enviar ID do broker
            new_price=new_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            new_volume=new_volume
        )

        if success:
            logger.info(f"Solicitação de modificação para ordem {client_order_id} (Broker: {broker_id_to_modify}) enviada.")
            # O status final da ordem e os valores atualizados virão via _handle_order_update_event
            return True
        else:
            # order_to_modify.status = original_status # Reverter status local se a submissão falhar
            logger.error(f"Falha ao enviar solicitação de modificação para ordem {client_order_id} (Broker: {broker_id_to_modify}).")
            return False


    async def cancel_order(self, client_order_id: str) -> bool: # Usar client_order_id
        """Cancela ordem pendente"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.error("Trade client não conectado. Não é possível cancelar ordem.")
            return False

        order_to_cancel = self.orders.get(client_order_id)
        if not order_to_cancel:
            logger.error(f"Ordem (Cliente ID: {client_order_id}) não encontrada para cancelamento.")
            return False

        if order_to_cancel.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.warning(f"Ordem {client_order_id} não pode ser cancelada (status atual: {order_to_cancel.status.value})")
            return False # Não é um erro, mas a ação não foi tomada

        broker_id_to_cancel = order_to_cancel.broker_order_id
        if not broker_id_to_cancel:
            # Se a ordem foi rejeitada localmente antes de ter um broker_id
            if order_to_cancel.status == OrderStatus.REJECTED:
                logger.info(f"Ordem {client_order_id} já estava rejeitada localmente.")
                order_to_cancel.status = OrderStatus.CANCELLED # Marcar como cancelada
                return True
            logger.error(f"Broker Order ID não encontrado para a ordem cliente {client_order_id}. Cancelamento não pode prosseguir.")
            return False

        # original_status = order_to_cancel.status
        # order_to_cancel.status = OrderStatus.CANCELLATION_PENDING # Atualizar status local

        success = await self.trade_client.cancel_order(broker_id_to_cancel) # Enviar ID do broker

        if success:
            logger.info(f"Solicitação de cancelamento para ordem {client_order_id} (Broker: {broker_id_to_cancel}) enviada.")
            # O status final virá via _handle_order_update_event
            return True
        else:
            # order_to_cancel.status = original_status # Reverter status local
            logger.error(f"Falha ao enviar solicitação de cancelamento para ordem {client_order_id} (Broker: {broker_id_to_cancel}).")
            return False

    async def cancel_all_orders(self):
        """Cancela todas as ordens pendentes (status PENDING, SUBMITTED)."""
        # Criar uma cópia da lista de IDs de ordens para evitar problemas de modificação durante a iteração
        order_ids_to_cancel = [
            o.id for o in self.orders.values()
            if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
        ]
        if not order_ids_to_cancel:
            logger.info("Nenhuma ordem pendente para cancelar.")
            return

        logger.info(f"Tentando cancelar {len(order_ids_to_cancel)} ordens pendentes...")
        # Usar asyncio.gather para enviar solicitações de cancelamento em paralelo
        results = await asyncio.gather(
            *[self.cancel_order(order_id) for order_id in order_ids_to_cancel],
            return_exceptions=True # Para não parar em caso de erro em uma delas
        )

        cancelled_count = sum(1 for r in results if r is True)
        failed_count = len(results) - cancelled_count
        logger.info(f"Cancelamento de todas as ordens: {cancelled_count} sucesso(s), {failed_count} falha(s).")

    async def close_position(self, position_id: str, reason: str = "", volume_to_close: Optional[float] = None) -> bool: # Usar position_id do broker
        """Fecha posição específica (total ou parcial)"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.error("Trade client não conectado. Não é possível fechar posição.")
            return False

        position_to_close = self.positions.get(position_id) # position_id aqui é o ID do broker
        if not position_to_close:
            logger.error(f"Posição (Broker ID: {position_id}) não encontrada no rastreamento local.")
            # Poderia tentar fechar mesmo assim se a API permitir fechar por ID sem ter o objeto local.
            # Mas é melhor ter o objeto local para referência.
            # Alternativamente, buscar a posição do broker primeiro se não estiver localmente.
            # broker_positions = await self.trade_client.get_positions()
            # found_on_broker = any(p.get('Id') == position_id for p in broker_positions)
            # if not found_on_broker:
            #     logger.error(f"Posição {position_id} também não encontrada no broker.")
            #     return False
            pass # Deixar a API do trade_client.close_position lidar com ID inválido


        # A API do TickTraderTrade.close_position pode precisar do volume para fechamento parcial
        success = await self.trade_client.close_position(position_id, volume=volume_to_close) # Passar volume opcional

        if success:
            action = "parcialmente" if volume_to_close and position_to_close and volume_to_close < position_to_close.size else ""
            logger.info(f"Solicitação de fechamento {action} para posição {position_id} enviada. Razão: {reason}")
            # A remoção/atualização da posição local deve ocorrer em _handle_position_update_event
            return True
        else:
            logger.error(f"Falha ao enviar solicitação de fechamento para posição {position_id}.")
            return False


    async def close_all_positions(self, reason: str = ""):
        """Fecha todas as posições abertas"""
        position_ids_to_close = list(self.positions.keys()) # IDs do broker
        if not position_ids_to_close:
            logger.info("Nenhuma posição aberta para fechar.")
            return

        logger.info(f"Tentando fechar {len(position_ids_to_close)} posições abertas. Razão: {reason}")
        results = await asyncio.gather(
            *[self.close_position(pid, reason) for pid in position_ids_to_close],
            return_exceptions=True
        )
        closed_count = sum(1 for r in results if r is True)
        failed_count = len(results) - closed_count
        logger.info(f"Fechamento de todas as posições: {closed_count} sucesso(s) de solicitação, {failed_count} falha(s).")


    async def update_trailing_stop(self, position_id: str, current_market_price: float) -> bool:
        """Atualiza trailing stop de posição"""
        # Esta lógica deve ser mais sofisticada, usando ATR e o estado da posição.
        # E o método modify_position_stops deve ser chamado.
        position = self.positions.get(position_id) # position_id é ID do broker
        if not position or not position.trailing_stop: # Supondo que Position tem um campo trailing_stop
            return False

        # Exemplo de lógica de trailing stop (precisa de ATR e outros dados da BaseStrategy.Position)
        atr = CONFIG.RISK_PARAMS.TRAILING_STOP_DISTANCE_PIPS * 0.0001 # Exemplo, pegar ATR real
        multiplier = 1.5 # Exemplo

        new_potential_stop: Optional[float] = None
        if position.side.capitalize() == 'Buy':
            new_potential_stop = current_market_price - (atr * multiplier)
            if new_potential_stop > position.stop_loss: # Só mover para cima
                return await self.modify_position_stops(position_id, stop_loss=new_potential_stop)
        elif position.side.capitalize() == 'Sell':
            new_potential_stop = current_market_price + (atr * multiplier)
            if new_potential_stop < position.stop_loss: # Só mover para baixo
                return await self.modify_position_stops(position_id, stop_loss=new_potential_stop)
        return False

    async def modify_position_stops(self, position_id: str, # ID da Posição do Broker
                                   stop_loss: Optional[float] = None,
                                   take_profit: Optional[float] = None) -> bool:
        """Modifica stops de uma posição aberta"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.error("Trade client não conectado. Não é possível modificar stops da posição.")
            return False

        # A API TickTrader pode ter um método específico para modificar posições
        # ou pode ser feito modificando a ordem original se a posição derivou de uma única ordem
        # e a API suportar isso. Assumindo que existe um método no trade_client:
        # success = await self.trade_client.modify_position(
        # position_id=position_id,
        # stop_loss=stop_loss,
        # take_profit=take_profit
        # )
        # Por enquanto, simulando que é uma modificação de "ordem" associada:
        # Isso é uma simplificação e depende da API real.
        # Se a posição não veio de uma ordem rastreada ou se a API é diferente, isso falhará.
        client_ord_id = self.broker_to_client_order_map.get(position_id)
        if not client_ord_id and position_id in self.orders: # Se o position_id for o mesmo que o client_order_id
            client_ord_id = position_id


        if client_ord_id:
            logger.info(f"Tentando modificar stops para posição {position_id} via ordem cliente {client_ord_id}.")
            # Chamar o self.modify_order (que espera o client_order_id)
            # Esta lógica é circular se modify_order chamar modify_position_stops.
            # O trade_client.modify_order deve ser o ponto de contato com a API de ordens.
            # O trade_client.modify_position (se existir) para posições.

            # A chamada correta aqui seria para a API do broker:
            success = await self.trade_client.modify_order( # Assumindo que modify_order no trade_client aceita ID da ordem do broker
                order_id=position_id, # Ou o order_id correto associado à posição
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            if success:
                logger.info(f"Solicitação de modificação de stops para posição {position_id} enviada.")
                # Atualizar localmente após confirmação do evento _handle_position_update_event
                return True
            else:
                logger.error(f"Falha ao enviar modificação de stops para posição {position_id}.")
                return False
        else:
            logger.warning(f"Não foi possível encontrar ordem cliente associada à posição {position_id} para modificar stops.")
            # Tentar modificar a posição diretamente se a API do trade_client suportar
            if hasattr(self.trade_client, 'modify_position_stops_api'): # Método hipotético
                 # success_pos_mod = await self.trade_client.modify_position_stops_api(position_id, stop_loss, take_profit)
                 # return success_pos_mod
                 pass # Implementar se houver tal método
            return False


    async def get_open_positions(self) -> List[Position]:
        """Retorna lista de posições abertas (convertidas para nosso objeto Position)"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.warning("Trade client não conectado, retornando posições locais.")
            return [p for p in self.positions.values() if isinstance(p, Position)]


        broker_pos_list = await self.trade_client.get_positions()
        # Atualizar o self.positions local com base no que o broker retorna
        current_broker_positions = {}
        parsed_positions = []
        for pos_data in broker_pos_list:
            parsed_pos = self._parse_broker_position_data(pos_data)
            if parsed_pos:
                current_broker_positions[parsed_pos.id] = parsed_pos
                parsed_positions.append(parsed_pos)
        self.positions = current_broker_positions # Sincronizar com o broker
        return parsed_positions


    async def get_account_balance(self) -> float:
        """Obtém balanço atual da conta"""
        if not self.trade_client or not self.trade_client.is_connected():
            logger.error("Trade client não conectado. Não é possível obter balanço.")
            # Tentar retornar último valor conhecido ou um default
            return float(self.trade_client.account_info.get('Balance', 0.0)) if self.trade_client else 0.0

        try:
            # O trade_client.get_account_balance() já deve fazer a chamada à API e atualizar
            # o account_info interno dele, ou retornar o valor diretamente.
            balance = await self.trade_client.get_account_balance() # Este método no TickTraderTrade já faz a chamada
            return balance if balance is not None else 0.0
        except Exception as e:
            logger.exception("Erro ao obter balanço da conta:")
            return 0.0

    async def get_positions_by_strategy(self, strategy_name: str) -> List[Position]: # Renomeado
        """Retorna posições de uma estratégia específica"""
        # As posições em self.positions são do tipo BaseStrategy.Position
        return [
            p for p in self.positions.values()
            if isinstance(p, Position) and p.strategy_name == strategy_name
        ]

    # Handlers de eventos (devem ser chamados pelo TickTraderTrade via callbacks)

    async def _handle_order_update_event(self, order_data_from_broker: Dict[str, Any]): # Renomeado e tipado
        """Processa atualização de ordem vinda do broker."""
        # order_data_from_broker é o payload JSON/dict do evento do broker
        broker_order_id = str(order_data_from_broker.get('Id', '')) # ID do Broker
        client_order_id = self.broker_to_client_order_map.get(broker_order_id)

        if not client_order_id and broker_order_id in self.orders: # Se o ID do broker for usado como chave local
            client_order_id = broker_order_id
        elif not client_order_id and order_data_from_broker.get('ClientOrderId') in self.orders: # Se API envia ClientOrderId
             client_order_id = order_data_from_broker.get('ClientOrderId')


        if not client_order_id:
            logger.warning(f"Atualização de ordem recebida para ID de broker desconhecido ou não mapeado: {broker_order_id}. Dados: {order_data_from_broker}")
            # Tentar parsear e adicionar se for uma nova ordem não rastreada
            # new_order = self._parse_broker_order_data(order_data_from_broker)
            # if new_order:
            #     self.orders[new_order.id] = new_order
            #     if new_order.broker_order_id:
            #         self.broker_to_client_order_map[new_order.broker_order_id] = new_order.id
            #     client_order_id = new_order.id
            # else:
            #     return
            return


        order = self.orders.get(client_order_id)
        if not order:
            logger.warning(f"Ordem com ID cliente {client_order_id} (Broker: {broker_order_id}) não encontrada localmente para atualização.")
            # Criar uma nova ordem local baseada nos dados do broker
            order = self._parse_broker_order_data(order_data_from_broker) # Precisa de um método para isso
            if order:
                self.orders[order.id] = order
                if order.broker_order_id and order.broker_order_id not in self.broker_to_client_order_map:
                     self.broker_to_client_order_map[order.broker_order_id] = order.id
            else:
                logger.error(f"Não foi possível parsear dados de ordem do broker para {broker_order_id}.")
                return


        logger.info(f"Atualização para ordem {order.id} (Broker: {broker_order_id}): Status {order_data_from_broker.get('Status')}")

        previous_status = order.status
        new_status_str = order_data_from_broker.get('Status', '').lower()
        try:
            order.status = OrderStatus(new_status_str)
        except ValueError:
            logger.error(f"Status de ordem desconhecido '{new_status_str}' recebido do broker para ordem {order.id}.")
            # Manter status anterior ou um estado de erro? Por enquanto, não alterar.
            order.status = previous_status # Reverter para status anterior conhecido

        order.broker_order_id = broker_order_id # Garantir que está atualizado
        if order_data_from_broker.get('Price'): order.price = float(order_data_from_broker['Price']) # Preço da ordem (limite)
        if order_data_from_broker.get('StopLoss'): order.stop_loss = float(order_data_from_broker['StopLoss'])
        if order_data_from_broker.get('TakeProfit'): order.take_profit = float(order_data_from_broker['TakeProfit'])
        if order_data_from_broker.get('FilledVolume'): order.filled_size = float(order_data_from_broker['FilledVolume'])
        order.remaining_size = order.size - order.filled_size

        if order.status == OrderStatus.FILLED or \
           (order.status == OrderStatus.PARTIALLY_FILLED and previous_status != OrderStatus.PARTIALLY_FILLED) or \
           (order.status == OrderStatus.PARTIALLY_FILLED and order_data_from_broker.get('LastFilledVolume', 0) > 0): # Se houve um novo preenchimento

            order.filled_at = datetime.now(timezone.utc) # Usar timestamp do evento se disponível, senão now()
            if order_data_from_broker.get('ExecutionTime'): # Exemplo de campo de timestamp
                try:
                    order.filled_at = datetime.fromtimestamp(int(order_data_from_broker['ExecutionTime'])/1000, tz=timezone.utc)
                except: pass # Manter now() se falhar

            # FilledPrice pode ser AvgFilledPrice ou LastFilledPrice dependendo da API e se é parcial/total
            order.fill_price = float(order_data_from_broker.get('AvgFilledPrice', order_data_from_broker.get('LastFilledPrice', order.fill_price or 0.0)))

            entry_price_for_slippage = order.price if order.type == OrderType.LIMIT else order.metadata.get('submission_market_price', order.fill_price)
            if entry_price_for_slippage and order.fill_price: # Garantir que ambos existam
                slippage_abs = abs(order.fill_price - entry_price_for_slippage)
                # Converter para pips (depende do par, aqui simplificado para EURUSD)
                pip_value_calc = 0.0001 # TODO: usar helper.get_pip_size(order.symbol)
                order.slippage_pips = slippage_abs / pip_value_calc
            else:
                order.slippage_pips = 0.0


            order.commission = float(order_data_from_broker.get('Commission', order.commission))

            # Se totalmente preenchida, criar/atualizar posição
            if order.status == OrderStatus.FILLED:
                await self._create_or_update_position_from_fill(order, order_data_from_broker)
                # Remover ordem do rastreamento ativo se totalmente preenchida
                # self.orders.pop(client_order_id, None)
                # self.broker_to_client_order_map.pop(broker_order_id, None)
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                 await self._create_or_update_position_from_fill(order, order_data_from_broker, is_partial=True)


            fill_latency_ms = 0
            if order.submitted_at and order.filled_at:
                fill_latency_ms = (order.filled_at - order.submitted_at).total_seconds() * 1000

            self._update_execution_stats(
                order.strategy_name,
                'filled_trades', # Métrica para fills reais
                fill_latency_ms=fill_latency_ms,
                slippage_pips=order.slippage_pips
            )
            logger.info(f"Ordem {order.id} (Broker: {broker_order_id}) preenchida/parcialmente preenchida. Preço Médio: {order.fill_price}, "
                       f"Preenchido: {order.filled_size}/{order.size}, Slippage: {order.slippage_pips:.1f} pips")

        elif order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            if order.status == OrderStatus.REJECTED:
                order.error_message = order_data_from_broker.get('RejectReason', order_data_from_broker.get('Text', 'Rejeitada pelo broker'))
                self._update_execution_stats(order.strategy_name, 'rejected_by_broker')
                logger.warning(f"Ordem {order.id} (Broker: {broker_order_id}) REJEITADA. Razão: {order.error_message}")
            elif order.status == OrderStatus.CANCELLED:
                order.cancelled_at = datetime.now(timezone.utc) # Usar timestamp do evento se disponível
                logger.info(f"Ordem {order.id} (Broker: {broker_order_id}) CANCELADA.")
            # Remover dos ativos
            # self.orders.pop(client_order_id, None)
            # self.broker_to_client_order_map.pop(broker_order_id, None)


    async def _handle_position_update_event(self, position_data_from_broker: Dict[str, Any]): # Renomeado
        """Processa atualização de posição vinda do broker."""
        # position_data_from_broker é o payload JSON/dict do evento do broker
        broker_pos_id = str(position_data_from_broker.get('Id', ''))
        if not broker_pos_id:
            logger.warning(f"Atualização de posição recebida sem ID: {position_data_from_broker}")
            return

        logger.debug(f"Atualização para posição (Broker ID: {broker_pos_id}): {position_data_from_broker}")


        local_position = self.positions.get(broker_pos_id)

        if position_data_from_broker.get('Status', '').lower() == 'closed':
            if local_position:
                # Calcular PnL final, comissões, etc., se a API não fornecer tudo.
                # A API geralmente envia um trade report/execution report para o fechamento.
                exit_price = float(position_data_from_broker.get('ClosePrice', local_position.entry_price)) # Usar entry_price se ClosePrice não vier
                closed_volume = float(position_data_from_broker.get('ClosedVolume', local_position.size))

                # Calcular PnL do fechamento
                pnl_this_close = 0
                if local_position.side.capitalize() == 'Buy':
                    pnl_this_close = (exit_price - local_position.entry_price) * closed_volume * CONFIG.SYMBOL_SPECIFICATIONS.get(local_position.symbol, {}).get('contract_size', 100000)
                else:
                    pnl_this_close = (local_position.entry_price - exit_price) * closed_volume * CONFIG.SYMBOL_SPECIFICATIONS.get(local_position.symbol, {}).get('contract_size', 100000)

                # Atualizar PnL total da posição (se ela teve múltiplos fechamentos parciais)
                local_position.pnl += pnl_this_close # Acumular PnL
                # Subtrair comissão aqui se aplicável ao fechamento e não contabilizada antes.

                # Reduzir tamanho da posição
                local_position.size -= closed_volume
                local_position.size = round(local_position.size, 2) # Arredondar

                if local_position.size <= 0.001: # Considerar fechada se tamanho muito pequeno
                    del self.positions[broker_pos_id]
                    logger.info(f"Posição (Broker ID: {broker_pos_id}) totalmente fechada. PnL total da posição: ${local_position.pnl:.2f}")
                    # Aqui você poderia chamar o DataManager para registrar o trade fechado.
                    # await data_manager.record_trade(local_position.to_dict_for_db()) # Método hipotético
                else:
                    logger.info(f"Posição (Broker ID: {broker_pos_id}) parcialmente fechada. "
                               f"Volume restante: {local_position.size}. PnL desta parte: ${pnl_this_close:.2f}")
            else:
                logger.warning(f"Recebido evento de fechamento para posição desconhecida (Broker ID: {broker_pos_id})")
            return # Posição fechada, não processar mais como aberta


        if not local_position:
            # Posição nova ou não rastreada, tentar criar/atualizar
            parsed_pos = self._parse_broker_position_data(position_data_from_broker)
            if parsed_pos:
                self.positions[parsed_pos.id] = parsed_pos
                local_position = parsed_pos
                logger.info(f"Nova posição (Broker ID: {broker_pos_id}) detectada e adicionada ao rastreamento.")
            else:
                logger.error(f"Não foi possível parsear dados de posição do broker para {broker_pos_id}")
                return


        # Atualizar dados da posição local com os do broker
        local_position.stop_loss = float(position_data_from_broker.get('StopLoss', local_position.stop_loss or 0.0))
        local_position.take_profit = float(position_data_from_broker.get('TakeProfit', local_position.take_profit or 0.0))
        local_position.pnl = float(position_data_from_broker.get('FloatingProfit', local_position.pnl)) # Usar FloatingProfit
        # Atualizar outros campos relevantes: Swap, MarginUsed, etc.

        # Se a API enviar CurrentPrice para a posição
        current_price_for_pos = float(position_data_from_broker.get('CurrentPrice', 0.0))
        if current_price_for_pos > 0 and local_position.entry_price > 0: # Recalcular PnL se CurrentPrice vier
            if local_position.side.capitalize() == 'Buy':
                local_position.pnl = (current_price_for_pos - local_position.entry_price) * local_position.size * CONFIG.SYMBOL_SPECIFICATIONS.get(local_position.symbol, {}).get('contract_size', 100000)
            else:
                local_position.pnl = (local_position.entry_price - current_price_for_pos) * local_position.size * CONFIG.SYMBOL_SPECIFICATIONS.get(local_position.symbol, {}).get('contract_size', 100000)
            # Subtrair comissões do PnL flutuante se não estiverem incluídas
            # local_position.pnl -= local_position.metadata.get('commission_total', 0.0)


    async def _create_or_update_position_from_fill(self, order: Order, fill_data: Dict[str, Any], is_partial: bool = False):
        """
        Cria uma nova posição ou atualiza uma existente baseada em um preenchimento de ordem.
        `fill_data` deve ser o payload do evento de preenchimento (ExecutionReport).
        """
        broker_pos_id_from_fill = str(fill_data.get('PositionId', order.broker_order_id or order.id)) # Broker pode retornar PositionId
        # Se PositionId não vier, podemos usar o BrokerOrderId como uma referência para a posição
        # (assumindo que uma ordem totalmente preenchida resulta em uma posição com ID relacionado).

        contract_size = CONFIG.SYMBOL_SPECIFICATIONS.get(order.symbol, {}).get('contract_size', 100000)
        filled_volume_this_fill = float(fill_data.get('LastFilledVolume', order.filled_size if not is_partial else 0.0)) # Volume deste preenchimento
        avg_fill_price_this_fill = float(fill_data.get('AvgFilledPrice', order.fill_price or 0.0)) # Preço deste preenchimento


        if broker_pos_id_from_fill in self.positions: # Atualizar posição existente (ex: scale-in, ou fill parcial anterior)
            pos = self.positions[broker_pos_id_from_fill]
            logger.info(f"Atualizando posição existente {pos.id} devido ao preenchimento da ordem {order.id}.")

            # Recalcular preço médio de entrada se estiver aumentando a posição (scale-in)
            if pos.side.capitalize() == order.side.capitalize(): # Aumentando a posição
                new_total_size = pos.size + filled_volume_this_fill
                pos.entry_price = ((pos.entry_price * pos.size) + (avg_fill_price_this_fill * filled_volume_this_fill)) / new_total_size
                pos.size = new_total_size
            else: # Reduzindo posição (não deveria acontecer aqui, isso é um fechamento)
                logger.warning(f"Preenchimento da ordem {order.id} parece estar reduzindo a posição {pos.id}. Isso deve ser tratado como um fechamento.")
                # Esta lógica de redução/fechamento é mais complexa e geralmente tratada por _handle_position_update_event
                # quando o Status da posição muda para 'Closed' ou o volume é reduzido.
                return # Não criar/atualizar posição aqui se for um fechamento.

            # Atualizar SL/TP se a ordem os tinha (a API pode fazer isso automaticamente)
            if order.stop_loss: pos.stop_loss = order.stop_loss
            if order.take_profit: pos.take_profit = order.take_profit
            pos.open_time = min(pos.open_time, order.filled_at or pos.open_time) # Usar o tempo de abertura mais antigo
            # Acumular comissão
            pos.metadata['commission_total'] = pos.metadata.get('commission_total', 0.0) + order.commission


        else: # Criar nova posição
            logger.info(f"Criando nova posição a partir do preenchimento da ordem {order.id}.")
            pos = Position( # Usando a dataclass Position da base_strategy
                id=broker_pos_id_from_fill, # Usar o ID da Posição retornado pelo broker
                strategy_name=order.strategy_name or "UnknownStrategy",
                symbol=order.symbol,
                side=order.side.capitalize(),
                entry_price=avg_fill_price_this_fill,
                size=filled_volume_this_fill, # O tamanho inicial da posição é o volume preenchido
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                open_time=order.filled_at or datetime.now(timezone.utc),
                metadata={
                    'client_order_id': order.id,
                    'broker_order_id': order.broker_order_id,
                    'initial_commission': order.commission,
                    'commission_total': order.commission,
                    **(order.metadata or {})
                }
            )
            self.positions[pos.id] = pos
        logger.info(f"Posição {pos.id} criada/atualizada: {pos.side} {pos.size} {pos.symbol} @ {pos.entry_price:.5f}")


    def _parse_broker_position_data(self, pos_data: Dict[str, Any]) -> Optional[Position]:
        """Converte dados de posição do broker (dict) para nosso objeto Position."""
        try:
            broker_id = str(pos_data.get('Id'))
            if not broker_id: return None

            # Determinar client_order_id associado se possível
            # (pode não estar disponível diretamente nos dados da posição do broker)
            client_order_id = self.broker_to_client_order_map.get(broker_id, broker_id) # Fallback para broker_id
            strategy_name_from_order = self.orders.get(client_order_id, {}).get('strategy_name', 'UnknownStrategy_Broker')

            open_timestamp_ms = pos_data.get('OpenTime', pos_data.get('Created', time.time() * 1000))


            return Position(
                id=broker_id,
                strategy_name=pos_data.get('Comment', strategy_name_from_order), # Usar Comment se disponível
                symbol=pos_data['Symbol'],
                side=pos_data['Side'].capitalize(),
                entry_price=float(pos_data['OpenPrice']), # Usar OpenPrice para posições
                size=float(pos_data['Volume']),
                stop_loss=float(pos_data.get('StopLoss', 0.0)) or None, # Converter 0.0 para None
                take_profit=float(pos_data.get('TakeProfit', 0.0)) or None, # Converter 0.0 para None
                open_time=datetime.fromtimestamp(open_timestamp_ms / 1000, tz=timezone.utc),
                pnl=float(pos_data.get('FloatingProfit', 0.0)), # PnL flutuante
                trailing_stop=pos_data.get('TrailingStopEnabled', False), # Exemplo
                metadata={
                    'broker_data': pos_data, # Guardar dados originais do broker
                    'swap': float(pos_data.get('Swap', 0.0)),
                    'margin': float(pos_data.get('Margin', 0.0)),
                    'commission_total': float(pos_data.get('Commission', 0.0)) # Comissão já paga pela posição
                }
            )
        except KeyError as ke:
            logger.error(f"Campo obrigatório ausente ao parsear dados da posição do broker: {ke}. Dados: {pos_data}")
            return None
        except Exception as e:
            logger.exception(f"Erro ao parsear dados da posição do broker:")
            return None

    def _parse_broker_order_data(self, order_data: Dict[str, Any]) -> Optional[Order]:
        """Converte dados de ordem do broker (dict) para nosso objeto Order."""
        try:
            broker_id = str(order_data.get('Id'))
            if not broker_id: return None

            client_id = str(order_data.get('ClientOrderId', broker_id)) # Preferir ClientOrderId se existir

            created_timestamp_ms = order_data.get('CreationTime', order_data.get('Time', time.time() * 1000))
            status_str = order_data.get('Status', 'unknown').lower()
            try:
                status_enum = OrderStatus(status_str)
            except ValueError:
                logger.warning(f"Status de ordem desconhecido '{status_str}' do broker. Usando PENDING.")
                status_enum = OrderStatus.PENDING


            order = Order(
                id=client_id,
                broker_order_id=broker_id,
                symbol=order_data['Symbol'],
                side=order_data['Side'].capitalize(),
                size=float(order_data['Volume']),
                type=OrderType(order_data.get('Type', 'Market').capitalize()), # Assumir Market se não especificado
                price=float(order_data['Price']) if order_data.get('Price') is not None else None,
                stop_loss=float(order_data.get('StopLoss', 0.0)) or None,
                take_profit=float(order_data.get('TakeProfit', 0.0)) or None,
                strategy_name=order_data.get('Comment', '').split('_')[0] if order_data.get('Comment') else None,
                status=status_enum,
                created_at=datetime.fromtimestamp(created_timestamp_ms / 1000, tz=timezone.utc),
                submitted_at=datetime.fromtimestamp(order_data['SubmissionTime'] / 1000, tz=timezone.utc) if order_data.get('SubmissionTime') else None,
                filled_at=datetime.fromtimestamp(order_data['ExecutionTime'] / 1000, tz=timezone.utc) if order_data.get('ExecutionTime') and status_enum in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED] else None,
                fill_price=float(order_data.get('AvgFilledPrice', 0.0)) or None if status_enum in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED] else None,
                filled_size=float(order_data.get('FilledVolume', 0.0)),
                commission=float(order_data.get('Commission', 0.0)),
                metadata={'broker_data': order_data},
                error_message=order_data.get('RejectReason') if status_enum == OrderStatus.REJECTED else None
            )
            order.remaining_size = order.size - order.filled_size
            return order

        except KeyError as ke:
            logger.error(f"Campo obrigatório ausente ao parsear dados da ordem do broker: {ke}. Dados: {order_data}")
            return None
        except Exception as e:
            logger.exception(f"Erro ao parsear dados da ordem do broker:")
            return None


    def _update_execution_stats(self, strategy_name: Optional[str], event_type: str,
                               submission_latency_ms: float = 0.0, # Renomeado
                               fill_latency_ms: float = 0.0, # Adicionado
                               slippage_pips: float = 0.0): # Renomeado
        """Atualiza estatísticas de execução"""
        # Se strategy_name for None, usar uma chave global ou "UNKNOWN"
        effective_strategy_name = strategy_name or "UNKNOWN_STRATEGY"
        stats = self.execution_stats[effective_strategy_name]

        if event_type == 'submitted_ok': # Ordem aceita pelo broker
            stats['total_requests'] += 1
            stats['submitted_ok'] += 1
            # Atualizar média de latência de submissão
            stats['avg_submission_latency_ms'] = \
                ((stats['avg_submission_latency_ms'] * (stats['submitted_ok'] -1)) + submission_latency_ms) / stats['submitted_ok'] \
                if stats['submitted_ok'] > 0 else submission_latency_ms

        elif event_type == 'rejected_by_broker': # Ordem rejeitada pelo broker
            stats['total_requests'] += 1 # Ainda foi uma tentativa
            stats['rejected_by_broker'] += 1

        elif event_type == 'filled_trades': # Um preenchimento (fill) ocorreu
            stats['filled_trades'] += 1
            # Atualizar média de latência de preenchimento
            stats['avg_fill_latency_ms'] = \
                ((stats['avg_fill_latency_ms'] * (stats['filled_trades'] -1)) + fill_latency_ms) / stats['filled_trades'] \
                if stats['filled_trades'] > 0 else fill_latency_ms
            # Atualizar média de slippage
            stats['avg_slippage_pips'] = \
                ((stats['avg_slippage_pips'] * (stats['filled_trades'] -1)) + slippage_pips) / stats['filled_trades'] \
                if stats['filled_trades'] > 0 else slippage_pips


    def get_execution_stats(self, strategy_name: Optional[str] = None) -> Dict[str, Any]: # Tipagem
        """Retorna estatísticas de execução"""
        if strategy_name:
            return dict(self.execution_stats.get(strategy_name, {})) # Usar get com default

        # Agregar todas as estratégias
        agg_stats: Dict[str, Any] = defaultdict(float) # Tipagem
        total_submitted_ok = 0
        total_filled_trades = 0

        for strat_name, stats_dict in self.execution_stats.items():
            agg_stats['total_requests'] += stats_dict.get('total_requests', 0)
            agg_stats['submitted_ok'] += stats_dict.get('submitted_ok', 0)
            agg_stats['rejected_by_broker'] += stats_dict.get('rejected_by_broker', 0)
            agg_stats['filled_trades'] += stats_dict.get('filled_trades', 0)

            current_submitted_ok = stats_dict.get('submitted_ok', 0)
            if current_submitted_ok > 0:
                agg_stats['weighted_avg_submission_latency_ms'] += stats_dict.get('avg_submission_latency_ms', 0.0) * current_submitted_ok
                total_submitted_ok += current_submitted_ok

            current_filled_trades = stats_dict.get('filled_trades', 0)
            if current_filled_trades > 0:
                agg_stats['weighted_avg_fill_latency_ms'] += stats_dict.get('avg_fill_latency_ms', 0.0) * current_filled_trades
                agg_stats['weighted_avg_slippage_pips'] += stats_dict.get('avg_slippage_pips', 0.0) * current_filled_trades
                total_filled_trades += current_filled_trades


        if total_submitted_ok > 0:
            agg_stats['avg_submission_latency_ms'] = agg_stats.pop('weighted_avg_submission_latency_ms', 0.0) / total_submitted_ok
        else:
            agg_stats['avg_submission_latency_ms'] = 0.0

        if total_filled_trades > 0:
            agg_stats['avg_fill_latency_ms'] = agg_stats.pop('weighted_avg_fill_latency_ms', 0.0) / total_filled_trades
            agg_stats['avg_slippage_pips'] = agg_stats.pop('weighted_avg_slippage_pips', 0.0) / total_filled_trades
        else:
            agg_stats['avg_fill_latency_ms'] = 0.0
            agg_stats['avg_slippage_pips'] = 0.0


        if agg_stats['total_requests'] > 0:
            agg_stats['overall_submission_success_rate'] = agg_stats['submitted_ok'] / agg_stats['total_requests']
            agg_stats['overall_fill_rate_vs_submitted'] = agg_stats['filled_trades'] / agg_stats['submitted_ok'] if agg_stats['submitted_ok'] > 0 else 0.0
        else:
            agg_stats['overall_submission_success_rate'] = 0.0
            agg_stats['overall_fill_rate_vs_submitted'] = 0.0


        return dict(agg_stats)


    async def check_order_timeouts(self): # Renomeado de check_order_timeout para plural
        """Verifica e cancela ordens com timeout de submissão ou que estão pendentes por muito tempo."""
        current_time_utc = datetime.now(timezone.utc)
        # Timeout para ordens que foram submetidas mas não receberam confirmação/fill/rejeição.
        submission_timeout_delta = timedelta(milliseconds=CONFIG.ORDER_TIMEOUT_MS)

        # Timeout mais longo para ordens limite/stop que estão no book mas não foram preenchidas.
        # pending_order_max_lifetime_delta = timedelta(hours=CONFIG.MAX_PENDING_ORDER_HOURS) # Exemplo de config

        order_ids_to_process = list(self.orders.keys()) # Copiar chaves para iterar

        for client_order_id in order_ids_to_process:
            order = self.orders.get(client_order_id)
            if not order: continue # Já foi removida

            timed_out = False
            reason = ""

            if order.status == OrderStatus.SUBMITTED and order.submitted_at:
                if current_time_utc - order.submitted_at > submission_timeout_delta:
                    timed_out = True
                    reason = f"Timeout de submissão ({CONFIG.ORDER_TIMEOUT_MS}ms) excedido."
            # elif order.status == OrderStatus.PENDING and order.type != OrderType.MARKET: # Para ordens limite/stop no book
                # if current_time_utc - order.created_at > pending_order_max_lifetime_delta:
                #     timed_out = True
                #     reason = f"Tempo máximo de vida para ordem pendente ({CONFIG.MAX_PENDING_ORDER_HOURS}h) excedido."

            if timed_out:
                logger.warning(f"Ordem {order.id} (Broker: {order.broker_order_id}) expirou por timeout. Razão: {reason}")
                cancel_success = await self.cancel_order(order.id)
                if cancel_success or not order.broker_order_id: # Se não tem broker_id, já foi rejeitada localmente ou nunca enviada
                    order.status = OrderStatus.EXPIRED
                    order.error_message = reason
                # Se cancelamento falhar e tiver broker_id, o status pode ficar CANCELLATION_PENDING


    async def get_order_status(self, client_order_id: str) -> Optional[OrderStatus]: # Usar client_order_id
        """Retorna status de uma ordem pelo ID do cliente"""
        order = self.orders.get(client_order_id)
        return order.status if order else None


    async def get_pending_orders(self) -> List[Order]:
        """Retorna ordens que ainda não foram finalizadas (filled, cancelled, rejected, expired)"""
        return [
            o for o in self.orders.values()
            if o.status not in [
                OrderStatus.FILLED, OrderStatus.CANCELLED,
                OrderStatus.REJECTED, OrderStatus.EXPIRED
            ]
        ]

    def calculate_commission(self, size_lots: float, symbol: str) -> float: # Renomeado e adicionado symbol
        """Calcula comissão estimada para um dado volume em lotes."""
        # Obter especificações do símbolo
        # Exemplo: specs = CONFIG.SYMBOL_SPECIFICATIONS.get(symbol, {})
        # contract_size = specs.get('contract_size', 100000)
        # commission_type = specs.get('commission_type', 'per_million')
        # commission_rate = specs.get('commission_value', 35) # ex: 35 USD per million USD
        # min_commission = specs.get('min_commission_per_trade', 0.01)

        # Simplificado, usando o valor de CONFIG e assumindo que é por lote
        # A config original tem COMMISSION_PER_LOT
        commission = size_lots * CONFIG.RISK_PARAMS.COMMISSION_PER_LOT if hasattr(CONFIG, 'RISK_PARAMS') else size_lots * 7.0
        return round(commission, 2)


    async def emergency_close_all(self, reason: str = "EMERGENCY_SYSTEM_CLOSE"): # Adicionado reason
        """Fechamento de emergência de todas as posições e cancelamento de ordens."""
        logger.critical(f"FECHAMENTO DE EMERGÊNCIA ATIVADO. Razão: {reason}")

        # Cancelar todas as ordens pendentes primeiro
        await self.cancel_all_orders()

        # Fechar todas as posições abertas
        await self.close_all_positions(reason)

        logger.critical("Fechamento de emergência concluído.")