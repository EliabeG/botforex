# api/fix_client.py
"""Cliente FIX 4.4 para backup de conexão"""
import quickfix as fix
import quickfix44 as fix44
from typing import Dict, Optional, Callable, List
from datetime import datetime
import uuid

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("fix_client")

class FIXApplication(fix.Application):
    """Aplicação FIX para gerenciar sessão"""
    
    def __init__(self):
        super().__init__()
        self.session_id = None
        self.connected = False
        self.orders = {}
        self.executions = {}
        
        # Callbacks
        self.on_execution = None
        self.on_order_update = None
        self.on_market_data = None
        
    def onCreate(self, sessionID):
        """Chamado quando sessão é criada"""
        self.session_id = sessionID
        logger.info(f"Sessão FIX criada: {sessionID}")
    
    def onLogon(self, sessionID):
        """Chamado quando logon é bem-sucedido"""
        self.connected = True
        logger.info(f"Logon FIX bem-sucedido: {sessionID}")
    
    def onLogout(self, sessionID):
        """Chamado quando logout ocorre"""
        self.connected = False
        logger.info(f"Logout FIX: {sessionID}")
    
    def toAdmin(self, message, sessionID):
        """Chamado para mensagens administrativas enviadas"""
        msg_type = message.getHeader().getField(fix.MsgType())
        
        # Adicionar credenciais no Logon
        if msg_type.getValue() == fix.MsgType_Logon:
            message.setField(fix.Username(CONFIG.LOGIN))
            message.setField(fix.Password(CONFIG.PASSWORD))
    
    def fromAdmin(self, message, sessionID):
        """Chamado para mensagens administrativas recebidas"""
        pass
    
    def toApp(self, message, sessionID):
        """Chamado para mensagens de aplicação enviadas"""
        logger.debug(f"Enviando: {message}")
    
    def fromApp(self, message, sessionID):
        """Chamado para mensagens de aplicação recebidas"""
        try:
            msg_type = message.getHeader().getField(fix.MsgType())
            
            if msg_type.getValue() == fix.MsgType_ExecutionReport:
                self._handle_execution_report(message)
            elif msg_type.getValue() == fix.MsgType_MarketDataSnapshotFullRefresh:
                self._handle_market_data(message)
            elif msg_type.getValue() == fix.MsgType_OrderCancelReject:
                self._handle_cancel_reject(message)
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
    
    def _handle_execution_report(self, message):
        """Processa relatório de execução"""
        try:
            order_id = message.getField(fix.ClOrdID())
            exec_type = message.getField(fix.ExecType())
            order_status = message.getField(fix.OrdStatus())
            
            exec_data = {
                'order_id': order_id,
                'exec_type': exec_type,
                'status': order_status,
                'symbol': message.getField(fix.Symbol()),
                'side': message.getField(fix.Side()),
                'quantity': message.getField(fix.OrderQty()),
                'price': message.getField(fix.Price()) if message.isSetField(fix.Price()) else None,
                'avg_price': message.getField(fix.AvgPx()) if message.isSetField(fix.AvgPx()) else None,
                'filled_qty': message.getField(fix.CumQty()) if message.isSetField(fix.CumQty()) else 0,
                'timestamp': datetime.now()
            }
            
            self.executions[order_id] = exec_data
            
            if self.on_execution:
                self.on_execution(exec_data)
                
            logger.info(f"Execução recebida: {order_id} - Status: {order_status}")
            
        except Exception as e:
            logger.error(f"Erro ao processar execution report: {e}")
    
    def _handle_market_data(self, message):
        """Processa dados de mercado"""
        try:
            symbol = message.getField(fix.Symbol())
            
            # Extrair bid/ask
            group = fix44.MarketDataSnapshotFullRefresh.NoMDEntries()
            num_entries = message.getField(fix.NoMDEntries())
            
            market_data = {'symbol': symbol, 'bids': [], 'asks': []}
            
            for i in range(int(num_entries)):
                message.getGroup(i + 1, group)
                
                entry_type = group.getField(fix.MDEntryType())
                price = float(group.getField(fix.MDEntryPx()))
                size = float(group.getField(fix.MDEntrySize()))
                
                if entry_type == fix.MDEntryType_BID:
                    market_data['bids'].append((price, size))
                elif entry_type == fix.MDEntryType_OFFER:
                    market_data['asks'].append((price, size))
            
            if self.on_market_data:
                self.on_market_data(market_data)
                
        except Exception as e:
            logger.error(f"Erro ao processar market data: {e}")
    
    def _handle_cancel_reject(self, message):
        """Processa rejeição de cancelamento"""
        try:
            order_id = message.getField(fix.ClOrdID())
            reason = message.getField(fix.Text()) if message.isSetField(fix.Text()) else "Unknown"
            
            logger.warning(f"Cancelamento rejeitado: {order_id} - Razão: {reason}")
            
        except Exception as e:
            logger.error(f"Erro ao processar cancel reject: {e}")

class FIXClient:
    """Cliente FIX 4.4 para trading"""
    
    def __init__(self):
        self.app = FIXApplication()
        self.initiator = None
        self.settings = None
        self._setup_settings()
        
    def _setup_settings(self):
        """Configura settings do FIX"""
        self.settings = fix.SessionSettings()
        
        # Configurações gerais
        self.settings.set(fix.SessionID(), "BeginString", "FIX.4.4")
        self.settings.set(fix.SessionID(), "SenderCompID", CONFIG.LOGIN)
        self.settings.set(fix.SessionID(), "TargetCompID", "TICKTRADER")
        self.settings.set(fix.SessionID(), "SocketConnectHost", CONFIG.SERVER)
        self.settings.set(fix.SessionID(), "SocketConnectPort", "5201")  # Porta FIX padrão
        self.settings.set(fix.SessionID(), "HeartBtInt", "30")
        self.settings.set(fix.SessionID(), "ReconnectInterval", "30")
        self.settings.set(fix.SessionID(), "FileStorePath", "fix_store")
        self.settings.set(fix.SessionID(), "FileLogPath", "fix_logs")
        self.settings.set(fix.SessionID(), "StartTime", "00:00:00")
        self.settings.set(fix.SessionID(), "EndTime", "00:00:00")
        self.settings.set(fix.SessionID(), "UseDataDictionary", "Y")
        self.settings.set(fix.SessionID(), "DataDictionary", "FIX44.xml")
    
    async def connect(self):
        """Conecta ao servidor FIX"""
        try:
            store_factory = fix.FileStoreFactory(self.settings)
            log_factory = fix.FileLogFactory(self.settings)
            
            self.initiator = fix.SocketInitiator(
                self.app,
                store_factory,
                self.settings,
                log_factory
            )
            
            self.initiator.start()
            
            # Aguardar conexão
            retry_count = 0
            while not self.app.connected and retry_count < 30:
                await asyncio.sleep(1)
                retry_count += 1
            
            if self.app.connected:
                logger.info("Cliente FIX conectado com sucesso")
                return True
            else:
                logger.error("Falha ao conectar cliente FIX")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao conectar FIX: {e}")
            return False
    
    async def disconnect(self):
        """Desconecta do servidor FIX"""
        if self.initiator:
            self.initiator.stop()
            logger.info("Cliente FIX desconectado")
    
    def create_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: Optional[float] = None,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> Optional[str]:
        """
        Cria nova ordem
        
        Args:
            symbol: Símbolo do instrumento
            side: 'BUY' ou 'SELL'
            quantity: Quantidade (lotes)
            order_type: 'MARKET' ou 'LIMIT'
            price: Preço limite (para ordem limit)
            stop_loss: Stop loss
            take_profit: Take profit
            
        Returns:
            ID da ordem ou None se falhar
        """
        if not self.app.connected:
            logger.error("FIX não conectado")
            return None
        
        try:
            # Criar mensagem de nova ordem
            order = fix44.NewOrderSingle()
            
            # Header
            order.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))
            
            # ID único da ordem
            cl_ord_id = str(uuid.uuid4())
            order.setField(fix.ClOrdID(cl_ord_id))
            
            # Detalhes da ordem
            order.setField(fix.Symbol(symbol))
            order.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
            order.setField(fix.OrderQty(quantity))
            order.setField(fix.OrdType(
                fix.OrdType_MARKET if order_type == "MARKET" else fix.OrdType_LIMIT
            ))
            
            if order_type == "LIMIT" and price:
                order.setField(fix.Price(price))
            
            # Time in force
            order.setField(fix.TimeInForce(fix.TimeInForce_DAY))
            
            # Timestamp
            order.setField(fix.TransactTime())
            
            # Adicionar stops se fornecidos (usando campos customizados)
            if stop_loss:
                order.setField(9001, str(stop_loss))  # Campo customizado para SL
            if take_profit:
                order.setField(9002, str(take_profit))  # Campo customizado para TP
            
            # Enviar ordem
            fix.Session.sendToTarget(order, self.app.session_id)
            
            logger.info(f"Ordem FIX enviada: {cl_ord_id}")
            return cl_ord_id
            
        except Exception as e:
            logger.error(f"Erro ao criar ordem FIX: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str, side: str) -> bool:
        """Cancela ordem existente"""
        if not self.app.connected:
            return False
        
        try:
            cancel = fix44.OrderCancelRequest()
            
            # IDs
            cancel.setField(fix.OrigClOrdID(order_id))
            cancel.setField(fix.ClOrdID(str(uuid.uuid4())))  # Novo ID para cancel
            
            # Detalhes originais da ordem
            cancel.setField(fix.Symbol(symbol))
            cancel.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
            
            # Timestamp
            cancel.setField(fix.TransactTime())
            
            # Enviar
            fix.Session.sendToTarget(cancel, self.app.session_id)
            
            logger.info(f"Cancelamento enviado para ordem: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao cancelar ordem FIX: {e}")
            return False
    
    def modify_order(self, order_id: str, symbol: str, side: str,
                    quantity: float, new_price: Optional[float] = None,
                    new_stop_loss: Optional[float] = None,
                    new_take_profit: Optional[float] = None) -> bool:
        """Modifica ordem existente"""
        if not self.app.connected:
            return False
        
        try:
            replace = fix44.OrderCancelReplaceRequest()
            
            # IDs
            replace.setField(fix.OrigClOrdID(order_id))
            new_id = str(uuid.uuid4())
            replace.setField(fix.ClOrdID(new_id))
            
            # Detalhes da ordem
            replace.setField(fix.Symbol(symbol))
            replace.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
            replace.setField(fix.OrderQty(quantity))
            
            if new_price:
                replace.setField(fix.Price(new_price))
            
            # Stops atualizados
            if new_stop_loss:
                replace.setField(9001, str(new_stop_loss))
            if new_take_profit:
                replace.setField(9002, str(new_take_profit))
            
            # Timestamp
            replace.setField(fix.TransactTime())
            
            # Enviar
            fix.Session.sendToTarget(replace, self.app.session_id)
            
            logger.info(f"Modificação enviada para ordem: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao modificar ordem FIX: {e}")
            return False
    
    def request_market_data(self, symbol: str) -> bool:
        """Solicita dados de mercado para símbolo"""
        if not self.app.connected:
            return False
        
        try:
            request = fix44.MarketDataRequest()
            
            # ID da requisição
            req_id = str(uuid.uuid4())
            request.setField(fix.MDReqID(req_id))
            
            # Tipo de subscrição
            request.setField(fix.SubscriptionRequestType(
                fix.SubscriptionRequestType_SNAPSHOT_UPDATES
            ))
            
            # Profundidade
            request.setField(fix.MarketDepth(10))
            
            # Tipos de entrada solicitados
            request.setField(fix.NoMDEntryTypes(2))
            
            # Bid
            group1 = fix44.MarketDataRequest.NoMDEntryTypes()
            group1.setField(fix.MDEntryType(fix.MDEntryType_BID))
            request.addGroup(group1)
            
            # Ask
            group2 = fix44.MarketDataRequest.NoMDEntryTypes()
            group2.setField(fix.MDEntryType(fix.MDEntryType_OFFER))
            request.addGroup(group2)
            
            # Símbolo
            request.setField(fix.NoRelatedSym(1))
            sym_group = fix44.MarketDataRequest.NoRelatedSym()
            sym_group.setField(fix.Symbol(symbol))
            request.addGroup(sym_group)
            
            # Enviar
            fix.Session.sendToTarget(request, self.app.session_id)
            
            logger.info(f"Solicitação de market data enviada: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao solicitar market data: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Obtém status de uma ordem"""
        if order_id in self.app.executions:
            return self.app.executions[order_id]
        return None
    
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self.app.connected
    
    def register_execution_callback(self, callback: Callable):
        """Registra callback para execuções"""
        self.app.on_execution = callback
    
    def register_market_data_callback(self, callback: Callable):
        """Registra callback para market data"""
        self.app.on_market_data = callback