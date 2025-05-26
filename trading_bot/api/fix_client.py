# api/fix_client.py
"""Cliente FIX 4.4 para backup de conexão"""
import quickfix as fix
import quickfix44 as fix44
from typing import Dict, Optional, Callable, List
from datetime import datetime
import uuid
import os # Adicionado para manipulação de caminhos
import asyncio # Adicionado para asyncio.sleep

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("fix_client")

class FIXApplication(fix.Application):
    """Aplicação FIX para gerenciar sessão"""

    def __init__(self):
        super().__init__()
        self.session_id = None
        self.connected = False
        self.orders = {} # Este dicionário não parece ser usado
        self.executions = {}

        # Callbacks
        self.on_execution: Optional[Callable] = None # Adicionada tipagem
        self.on_order_update: Optional[Callable] = None # Adicionada tipagem (embora não usada)
        self.on_market_data: Optional[Callable] = None # Adicionada tipagem

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
        msg_type_field = message.getHeader().getField(fix.MsgType()) # Renomeado para evitar conflito
        msg_type_value = msg_type_field.getString() # Obter valor como string

        # Adicionar credenciais no Logon
        if msg_type_value == fix.MsgType_Logon: # Comparar com string
            message.setField(fix.Username(CONFIG.LOGIN))
            message.setField(fix.Password(CONFIG.PASSWORD))

    def fromAdmin(self, message, sessionID):
        """Chamado para mensagens administrativas recebidas"""
        # logger.debug(f"Admin recebido: {message}") # Descomente para debug
        pass

    def toApp(self, message, sessionID):
        """Chamado para mensagens de aplicação enviadas"""
        logger.debug(f"Enviando App Msg: {message.toString().replace(chr(1), '|')}") # Usar toString e substituir SOH

    def fromApp(self, message, sessionID):
        """Chamado para mensagens de aplicação recebidas"""
        logger.debug(f"App Msg Recebida: {message.toString().replace(chr(1), '|')}") # Usar toString
        try:
            msg_type_field = message.getHeader().getField(fix.MsgType()) # Renomeado
            msg_type_value = msg_type_field.getString()

            if msg_type_value == fix.MsgType_ExecutionReport:
                self._handle_execution_report(message)
            elif msg_type_value == fix.MsgType_MarketDataSnapshotFullRefresh:
                self._handle_market_data(message)
            elif msg_type_value == fix.MsgType_OrderCancelReject:
                self._handle_cancel_reject(message)

        except Exception as e:
            logger.exception(f"Erro ao processar mensagem FIX fromApp:") # Usar logger.exception

    def _handle_execution_report(self, message):
        """Processa relatório de execução"""
        try:
            # Extrair campos de forma segura
            cl_ord_id_field = fix.ClOrdID()
            message.getField(cl_ord_id_field)
            order_id = cl_ord_id_field.getValue()

            exec_type_field = fix.ExecType()
            message.getField(exec_type_field)
            exec_type = exec_type_field.getValue()

            ord_status_field = fix.OrdStatus()
            message.getField(ord_status_field)
            order_status = ord_status_field.getValue()
            
            symbol_field = fix.Symbol()
            message.getField(symbol_field)
            symbol = symbol_field.getValue()

            side_field = fix.Side()
            message.getField(side_field)
            side = side_field.getValue()
            
            order_qty_field = fix.OrderQty()
            message.getField(order_qty_field)
            quantity = order_qty_field.getValue()


            price = None
            if message.isSetField(fix.Price().getField()):
                price_field = fix.Price()
                message.getField(price_field)
                price = price_field.getValue()

            avg_price = None
            if message.isSetField(fix.AvgPx().getField()):
                avg_px_field = fix.AvgPx()
                message.getField(avg_px_field)
                avg_price = avg_px_field.getValue()
            
            filled_qty = 0.0
            if message.isSetField(fix.CumQty().getField()):
                cum_qty_field = fix.CumQty()
                message.getField(cum_qty_field)
                filled_qty = cum_qty_field.getValue()


            exec_data = {
                'order_id': order_id,
                'exec_type': exec_type,
                'status': order_status,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'avg_price': avg_price,
                'filled_qty': filled_qty,
                'timestamp': datetime.now() # Usar timestamp do evento se disponível na mensagem
            }

            self.executions[order_id] = exec_data

            if self.on_execution:
                # Idealmente, on_execution seria uma corrotina se chamada de um contexto async
                # Se FIXApplication for usada em um loop asyncio, esta chamada deve ser agendada
                # asyncio.create_task(self.on_execution(exec_data)) ou similar
                self.on_execution(exec_data) # Mantendo síncrono por enquanto

            logger.info(f"Execução FIX recebida: {order_id} - Status: {order_status}")

        except Exception as e:
            logger.exception(f"Erro ao processar execution report FIX:")

    def _handle_market_data(self, message):
        """Processa dados de mercado"""
        try:
            symbol_field = fix.Symbol()
            message.getField(symbol_field)
            symbol = symbol_field.getValue()

            group = fix44.MarketDataSnapshotFullRefresh.NoMDEntries()
            num_entries_field = fix.NoMDEntries()
            message.getField(num_entries_field)
            num_entries = num_entries_field.getValue()


            market_data = {'symbol': symbol, 'bids': [], 'asks': [], 'timestamp': datetime.now()}

            for i in range(1, int(num_entries) + 1): # Loop de 1 até num_entries
                message.getGroup(i, group) # getGroup é 1-based

                md_entry_type_field = fix.MDEntryType()
                group.getField(md_entry_type_field)
                entry_type = md_entry_type_field.getValue()

                md_entry_px_field = fix.MDEntryPx()
                group.getField(md_entry_px_field)
                price = md_entry_px_field.getValue()
                
                md_entry_size_field = fix.MDEntrySize()
                group.getField(md_entry_size_field)
                size = md_entry_size_field.getValue()


                if entry_type == fix.MDEntryType_BID:
                    market_data['bids'].append((price, size))
                elif entry_type == fix.MDEntryType_OFFER:
                    market_data['asks'].append((price, size))

            if self.on_market_data:
                self.on_market_data(market_data) # Ver observação sobre async em _handle_execution_report

        except Exception as e:
            logger.exception(f"Erro ao processar market data FIX:")

    def _handle_cancel_reject(self, message):
        """Processa rejeição de cancelamento"""
        try:
            cl_ord_id_field = fix.ClOrdID()
            message.getField(cl_ord_id_field)
            order_id = cl_ord_id_field.getValue()
            
            reason = "Unknown"
            if message.isSetField(fix.Text().getField()):
                text_field = fix.Text()
                message.getField(text_field)
                reason = text_field.getValue()


            logger.warning(f"Cancelamento FIX rejeitado: {order_id} - Razão: {reason}")

        except Exception as e:
            logger.exception(f"Erro ao processar cancel reject FIX:")

class FIXClient:
    """Cliente FIX 4.4 para trading"""

    def __init__(self):
        self.app = FIXApplication()
        self.initiator: Optional[fix.SocketInitiator] = None # Adicionada tipagem
        self.settings: Optional[fix.SessionSettings] = None # Adicionada tipagem
        self._setup_settings()

    def _get_writable_path(self, relative_path: str) -> str:
        """Gera um caminho absoluto e garante que o diretório exista."""
        # Usar CONFIG.DATA_DIR ou CONFIG.LOG_DIR que já usam Path e mkdir
        if "store" in relative_path:
            base_dir = Path(CONFIG.DATA_DIR) / "fix"
        elif "logs" in relative_path:
            base_dir = Path(CONFIG.LOG_DIR) / "fix"
        else:
            base_dir = Path(CONFIG.DATA_DIR) / "fix_other" # Fallback

        final_path = base_dir / relative_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return str(final_path)


    def _setup_settings(self):
        """Configura settings do FIX"""
        self.settings = fix.SessionSettings()
        default_session_id = fix.SessionID("FIX.4.4", CONFIG.LOGIN, "TICKTRADER") # Default

        # Configurações gerais
        self.settings.set(default_session_id, "ConnectionType", "initiator")
        self.settings.set(default_session_id, "SocketConnectHost", CONFIG.SERVER.split(':')[0]) # Assume formato "host:port"
        self.settings.set(default_session_id, "SocketConnectPort", "5201")  # Porta FIX padrão, pode precisar ser configurável
        self.settings.set(default_session_id, "HeartBtInt", "30")
        self.settings.set(default_session_id, "ReconnectInterval", "30")
        self.settings.set(default_session_id, "FileStorePath", self._get_writable_path("store"))
        self.settings.set(default_session_id, "FileLogPath", self._get_writable_path("logs"))
        self.settings.set(default_session_id, "StartTime", "00:00:00")
        self.settings.set(default_session_id, "EndTime", "00:00:00")
        self.settings.set(default_session_id, "UseDataDictionary", "Y")
        # O DataDictionary precisa estar no caminho certo.
        # Se estiver na raiz do projeto, ou em um subdir específico:
        dd_path = Path(__file__).parent.parent / "config" / "FIX44.xml" # Exemplo de caminho
        if not dd_path.exists():
            logger.warning(f"Dicionário FIX44.xml não encontrado em {dd_path}. Verifique o caminho.")
            # Se não encontrar, pode tentar um caminho padrão ou falhar.
            # Por enquanto, vamos manter o nome do arquivo, quickfix pode ter seus locais de busca.
            self.settings.set(default_session_id, "DataDictionary", "FIX44.xml")
        else:
            self.settings.set(default_session_id, "DataDictionary", str(dd_path))

        # Configurações específicas da sessão (se você tiver múltiplas no arquivo de config)
        # sessionID = fix.SessionID("FIX.4.4", CONFIG.LOGIN, "TICKTRADER")
        # self.settings.set(sessionID, "BeginString", "FIX.4.4")
        # self.settings.set(sessionID, "SenderCompID", CONFIG.LOGIN)
        # self.settings.set(sessionID, "TargetCompID", "TICKTRADER")
        # ... (outras configs específicas da sessão se SessionSettings vier de um arquivo)


    async def connect(self) -> bool: # Adicionado tipo de retorno
        """Conecta ao servidor FIX"""
        try:
            # Garantir que os diretórios de log/store existam
            Path(self._get_writable_path("store")).mkdir(parents=True, exist_ok=True)
            Path(self._get_writable_path("logs")).mkdir(parents=True, exist_ok=True)

            store_factory = fix.FileStoreFactory(self.settings)
            log_factory = fix.FileLogFactory(self.settings)

            self.initiator = fix.SocketInitiator(
                self.app,
                store_factory,
                self.settings,
                log_factory
            )

            # Rodar start() em um executor para não bloquear o loop asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.initiator.start)


            # Aguardar conexão
            retry_count = 0
            while not self.app.connected and retry_count < 30:
                await asyncio.sleep(1)
                retry_count += 1

            if self.app.connected:
                logger.info("Cliente FIX conectado com sucesso")
                return True
            else:
                logger.error("Falha ao conectar cliente FIX após tentativas.")
                if self.initiator: # Tentar parar se a conexão falhou mas o initiator foi criado
                    await loop.run_in_executor(None, self.initiator.stop)
                return False

        except Exception as e:
            logger.exception(f"Erro ao conectar FIX:")
            return False

    async def disconnect(self):
        """Desconecta do servidor FIX"""
        if self.initiator:
            # Rodar stop() em um executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.initiator.stop)
            logger.info("Cliente FIX desconectado")
            self.app.connected = False # Marcar explicitamente como desconectado

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
        if not self.app.connected or not self.app.session_id: # Adicionado check de session_id
            logger.error("FIX não conectado ou ID da sessão não disponível")
            return None

        try:
            order = fix44.NewOrderSingle()

            # Header padrão já é definido pelo quickfix ao criar a mensagem com tipo.
            # Não é necessário: order.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))

            cl_ord_id = str(uuid.uuid4())
            order.setField(fix.ClOrdID(cl_ord_id))
            order.setField(fix.Symbol(symbol))
            order.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
            order.setField(fix.OrderQty(quantity)) # quickfix lida com conversão para float
            order.setField(fix.OrdType(
                fix.OrdType_MARKET if order_type.upper() == "MARKET" else fix.OrdType_LIMIT
            ))

            if order_type.upper() == "LIMIT" and price is not None:
                order.setField(fix.Price(price)) # quickfix lida com conversão

            order.setField(fix.TimeInForce(fix.TimeInForce_DAY)) # Exemplo, pode precisar ser outro
            order.setField(fix.TransactTime(datetime.utcnow())) # Usar UTC e deixar quickfix formatar

            # Adicionar stops se fornecidos (usando campos customizados ou padrão se disponíveis)
            # A especificação FIX 4.4 tem StopPx (Tag 99) para ordens Stop ou StopLimit.
            # Para SL/TP em ordens Market/Limit, é comum usar ordens separadas ou
            # funcionalidades específicas do broker (podem ser campos customizados).
            # Os campos 9001 e 9002 são exemplos de campos customizados.
            # Verifique a documentação do seu broker para os campos corretos de SL/TP.
            if stop_loss is not None:
                 # Se for um campo customizado:
                order.setField(fix.CustomTag(9001), str(stop_loss))
                # Se for um campo padrão para ordens stop (mas não para SL em market/limit):
                # order.setField(fix.StopPx(stop_loss))
            if take_profit is not None:
                # Se for um campo customizado:
                order.setField(fix.CustomTag(9002), str(take_profit))


            fix.Session.sendToTarget(order, self.app.session_id)
            logger.info(f"Ordem FIX enviada: {cl_ord_id}")
            return cl_ord_id

        except Exception as e:
            logger.exception(f"Erro ao criar ordem FIX:")
            return None

    def cancel_order(self, order_id: str, original_cl_ord_id: str, symbol: str, side: str) -> bool: # Adicionado original_cl_ord_id
        """Cancela ordem existente"""
        if not self.app.connected or not self.app.session_id:
            logger.error("FIX não conectado ou ID da sessão não disponível para cancelamento")
            return False

        try:
            cancel = fix44.OrderCancelRequest()

            cancel.setField(fix.OrigClOrdID(original_cl_ord_id)) # ID da ordem original a ser cancelada
            cancel.setField(fix.ClOrdID(str(uuid.uuid4())))  # Novo ID para a requisição de cancelamento
            cancel.setField(fix.Symbol(symbol))
            cancel.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
            cancel.setField(fix.TransactTime(datetime.utcnow()))
            # O OrderID (tag 37) do broker também pode ser necessário em alguns casos
            # Se você tem o OrderID do broker:
            # cancel.setField(fix.OrderID(order_id_do_broker))


            fix.Session.sendToTarget(cancel, self.app.session_id)
            logger.info(f"Cancelamento FIX enviado para ordem original: {original_cl_ord_id}")
            return True

        except Exception as e:
            logger.exception(f"Erro ao cancelar ordem FIX:")
            return False

    def modify_order(self, original_cl_ord_id: str, symbol: str, side: str,
                    new_quantity: Optional[float] = None, # Renomeado para clareza
                    new_price: Optional[float] = None,
                    new_stop_loss: Optional[float] = None,
                    new_take_profit: Optional[float] = None) -> bool:
        """Modifica ordem existente"""
        if not self.app.connected or not self.app.session_id:
            logger.error("FIX não conectado ou ID da sessão não disponível para modificação")
            return False

        try:
            replace = fix44.OrderCancelReplaceRequest()

            replace.setField(fix.OrigClOrdID(original_cl_ord_id))
            replace.setField(fix.ClOrdID(str(uuid.uuid4()))) # Novo ClOrdID para a requisição de replace
            replace.setField(fix.Symbol(symbol))
            replace.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
            
            # É mandatório enviar OrdType na OrderCancelReplaceRequest
            # Assumindo que a ordem original era LIMIT se new_price for fornecido, senão MARKET
            # Você precisa saber o tipo da ordem original ou ter uma lógica para determiná-lo
            # Aqui, estou assumindo que você quer manter o tipo ou alterá-lo se necessário.
            # FIX geralmente requer que você reenvie todos os campos da ordem, mesmo os não alterados.
            
            # Exemplo: Obter a ordem original para pegar seu tipo e quantidade se não for mudar a quantidade
            # original_order_details = self.app.executions.get(original_cl_ord_id) # Supondo que executions guarde detalhes
            # if not original_order_details:
            #     logger.error(f"Detalhes da ordem original {original_cl_ord_id} não encontrados para modificação.")
            #     return False

            # ord_type_original = original_order_details.get('type_fix_enum', fix.OrdType_MARKET) # Precisa do tipo original
            # replace.setField(fix.OrdType(ord_type_original))

            if new_quantity is not None:
                replace.setField(fix.OrderQty(new_quantity))
            # else:
                # Se não for mudar a quantidade, você ainda precisa enviar a quantidade original
                # replace.setField(fix.OrderQty(original_order_details.get('quantity')))


            if new_price is not None:
                replace.setField(fix.Price(new_price))
                replace.setField(fix.OrdType(fix.OrdType_LIMIT)) # Se mudar preço, geralmente é LIMIT
            # else if ord_type_original == fix.OrdType_LIMIT:
                #  replace.setField(fix.Price(original_order_details.get('price')))


            if new_stop_loss is not None:
                replace.setField(fix.CustomTag(9001), str(new_stop_loss))
            if new_take_profit is not None:
                replace.setField(fix.CustomTag(9002), str(new_take_profit))

            replace.setField(fix.TransactTime(datetime.utcnow()))

            fix.Session.sendToTarget(replace, self.app.session_id)
            logger.info(f"Modificação FIX enviada para ordem original: {original_cl_ord_id}")
            return True

        except Exception as e:
            logger.exception(f"Erro ao modificar ordem FIX:")
            return False

    def request_market_data(self, symbol: str, subscription_type: str = fix.SubscriptionRequestType_SNAPSHOT_UPDATES) -> bool:
        """Solicita dados de mercado para símbolo"""
        if not self.app.connected or not self.app.session_id:
            logger.error("FIX não conectado ou ID da sessão não disponível para MD request")
            return False

        try:
            request = fix44.MarketDataRequest()

            req_id = str(uuid.uuid4())
            request.setField(fix.MDReqID(req_id))
            request.setField(fix.SubscriptionRequestType(subscription_type))
            request.setField(fix.MarketDepth(10)) # Profundidade de 10 níveis

            # Tipos de entrada solicitados (Bid, Offer, Trade)
            request.setField(fix.NoMDEntryTypes(3)) # Número de tipos de entrada

            group_bid = fix44.MarketDataRequest.NoMDEntryTypes()
            group_bid.setField(fix.MDEntryType(fix.MDEntryType_BID))
            request.addGroup(group_bid)

            group_ask = fix44.MarketDataRequest.NoMDEntryTypes()
            group_ask.setField(fix.MDEntryType(fix.MDEntryType_OFFER))
            request.addGroup(group_ask)
            
            group_trade = fix44.MarketDataRequest.NoMDEntryTypes()
            group_trade.setField(fix.MDEntryType(fix.MDEntryType_TRADE)) # Para receber últimos trades
            request.addGroup(group_trade)


            # Símbolo
            request.setField(fix.NoRelatedSym(1)) # Número de símbolos
            sym_group = fix44.MarketDataRequest.NoRelatedSym()
            sym_group.setField(fix.Symbol(symbol))
            request.addGroup(sym_group)
            
            # MDUpdateType para subscrição incremental ou full refresh
            # request.setField(fix.MDUpdateType(fix.MDUpdateType_FULL_REFRESH)) # ou INCREMENTAL_REFRESH

            fix.Session.sendToTarget(request, self.app.session_id)
            logger.info(f"Solicitação de market data FIX enviada para: {symbol}")
            return True

        except Exception as e:
            logger.exception(f"Erro ao solicitar market data FIX:")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Obtém status de uma ordem"""
        return self.app.executions.get(order_id) # order_id aqui deve ser o ClOrdID

    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self.app.connected

    def register_execution_callback(self, callback: Callable):
        """Registra callback para execuções"""
        self.app.on_execution = callback

    def register_market_data_callback(self, callback: Callable):
        """Registra callback para market data"""
        self.app.on_market_data = callback