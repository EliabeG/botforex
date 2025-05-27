# core/orchestrator.py
import asyncio
from typing import Dict, List, Optional, Set, Any 
from datetime import datetime, timedelta, timezone 
import numpy as np 
from collections import defaultdict

from config.settings import CONFIG, REGIME_CONFIG 
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.data_manager import DataManager
from core.execution_engine import ExecutionEngine, Order as EngineOrder, OrderStatus as EngineOrderStatus, OrderType as EngineOrderType 
from strategies.base_strategy import BaseStrategy, Signal, Position as StrategyPosition, ExitSignal 
from optimization.scoring import StrategyScorer, PerformanceMetrics # Adicionado PerformanceMetrics
from risk.risk_manager import RiskManager
from api.ticktrader_ws import TickTraderFeed, TickTraderTrade, TickData, DOMSnapshot 
from utils.logger import setup_logger
# Adicionar import para CircuitBreakerState
from risk.circuit_breaker import CircuitBreakerState


logger = setup_logger("orchestrator")

class TradingOrchestrator:
    """Orquestrador central do sistema de trading"""

    def __init__(self, mode: str = "live"):
        self.mode = mode
        self.running = False

        # Componentes principais
        self.regime_detector = MarketRegimeDetector()
        self.data_manager = DataManager()
        self.execution_engine = ExecutionEngine(mode=mode)
        self.risk_manager = RiskManager()
        self.scorer = StrategyScorer()

        self.feed_client: Optional[TickTraderFeed] = None
        self.trade_client: Optional[TickTraderTrade] = None


        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: Set[str] = set()
        self.strategy_scores: Dict[str, float] = {}

        self.current_regime: Optional[str] = None 
        self.last_regime_update: datetime = datetime.now(timezone.utc) 
        self.last_score_update: datetime = datetime.now(timezone.utc) 
        self.tick_count: int = 0
        self.trade_count: int = 0 

        self.daily_pnl_pct: float = 0.0 
        self.session_start_balance: float = 0.0
        self.max_drawdown_pct: float = 0.0 

        self._main_loop_tasks: List[asyncio.Task] = [] 


    async def initialize(self):
        """Inicializa todos os componentes"""
        logger.info(f"Inicializando orquestrador em modo {self.mode}...")

        try:
            await self._connect_websockets()
            if not (self.feed_client and self.feed_client.is_connected() and \
                    self.trade_client and self.trade_client.is_connected()):
                logger.critical("Falha ao conectar WebSockets. Orquestrador nao pode continuar.")
                raise ConnectionError("Falha na conexao WebSocket inicial.")


            await self.data_manager.initialize() 

            if self.mode != "live":
                await self._load_historical_data_for_init() 

            await self.execution_engine.initialize(self.trade_client) # type: ignore 


            await self._load_strategies()
            await self._initialize_regime_detector()


            self.session_start_balance = await self.execution_engine.get_account_balance()
            if self.session_start_balance == 0.0 and self.mode == "live": 
                logger.warning("Balanco inicial da conta e 0. Verifique a conexao ou a conta.")
            await self.risk_manager.initialize(self.session_start_balance)


            logger.info("Orquestrador inicializado com sucesso")

        except Exception as e:
            logger.exception("Erro na inicializacao do orquestrador:") 
            raise

    async def _connect_websockets(self):
        """Conecta aos WebSockets do TickTrader"""
        logger.info("Conectando WebSockets...")
        try:
            self.feed_client = TickTraderFeed()
            self.trade_client = TickTraderTrade()

            await self.feed_client.connect() 
            if not self.feed_client.is_connected():
                raise ConnectionError("Falha ao conectar/autenticar no Feed WebSocket.")

            await self.feed_client.subscribe_symbol(CONFIG.SYMBOL)
            await self.feed_client.subscribe_dom(CONFIG.SYMBOL, CONFIG.DOM_LEVELS)

            await self.trade_client.connect() 
            if not self.trade_client.is_connected():
                raise ConnectionError("Falha ao conectar/autenticar no Trade WebSocket.")

            logger.info("WebSockets conectados e autenticados.")
        except Exception as e:
            logger.exception("Erro critico durante conexao WebSocket:")
            if self.feed_client: await self.feed_client.disconnect("Falha na conexao inicial")
            if self.trade_client: await self.trade_client.disconnect("Falha na conexao inicial")
            self.feed_client = None
            self.trade_client = None
            raise 

    async def _load_historical_data_for_init(self):
        """Carrega dados historicos para inicializacao (ex: backtest ou treino de modelo)."""
        logger.info("Carregando dados historicos para inicializacao (se aplicavel ao modo)...")
        pass 

    async def _load_strategies(self):
        """Carrega todas as estrategias disponiveis"""
        logger.info("Carregando estrategias...")
        from strategies import load_all_strategies 

        strategy_classes = load_all_strategies()

        for strategy_class in strategy_classes:
            try:
                strategy_instance = strategy_class() 
                self.strategies[strategy_instance.name] = strategy_instance
                await strategy_instance.initialize_strategy() 

                params = await self.data_manager.load_strategy_params(strategy_instance.name)
                if params:
                    strategy_instance.update_parameters(params)
                    logger.info(f"Parametros carregados para {strategy_instance.name}")

            except Exception as e:
                logger.exception(f"Erro ao carregar estrategia {strategy_class.__name__}:")

        logger.info(f"{len(self.strategies)} estrategias carregadas e inicializadas.")

    async def _initialize_regime_detector(self): 
        """Inicializa o detector de regime, treinando ou carregando modelo."""
        logger.info("Inicializando detector de regime...")
        try:
            await self.regime_detector.load_model() 
            if not self.regime_detector.is_trained:
                logger.info("Modelo de regime nao encontrado ou falhou ao carregar. Tentando treinar...")
                
                required_ticks_for_train = (REGIME_CONFIG.TREND_WINDOW or 250) + 50
                days_for_train = max(30, (required_ticks_for_train // (24*60)) + 2) 

                historical_data_df = await self.data_manager.get_historical_ticks(
                    symbol=CONFIG.SYMBOL,
                    days=days_for_train 
                )

                if historical_data_df is not None and not historical_data_df.empty:
                    await self.regime_detector.train(historical_data_df)
                    if self.regime_detector.is_trained:
                        logger.info("Detector de regime treinado com sucesso.")
                    else:
                        logger.warning("Treinamento do detector de regime falhou. Operando sem ML para regime.")
                else:
                    logger.warning("Sem dados historicos suficientes para treinar detector de regime. Operando sem ML para regime.")
            else:
                logger.info("Modelo de detector de regime carregado com sucesso.")

        except Exception as e:
            logger.exception("Erro ao inicializar detector de regime:")


    async def run(self):
        """Loop principal de execucao"""
        if not self.feed_client or not self.trade_client:
            logger.critical("Clientes WebSocket nao estao inicializados. Encerrando o Orchestrator.")
            return

        self.running = True
        logger.info("Iniciando loop principal do orquestrador")

        self._main_loop_tasks = [ 
            asyncio.create_task(self._process_market_data(), name="ProcessMarketData"),
            asyncio.create_task(self._update_regime_and_strategies(), name="UpdateRegimeStrategies"),
            asyncio.create_task(self._monitor_open_positions(), name="MonitorPositions"), 
            asyncio.create_task(self._perform_risk_checks(), name="RiskChecks") 
        ]

        try:
            await asyncio.gather(*self._main_loop_tasks)
        except asyncio.CancelledError:
            logger.info("Loop principal do orquestrador cancelado.")
        except Exception as e:
            logger.exception("Erro critico no loop principal do orquestrador:") 
            await self.shutdown()
        finally:
            logger.info("Loop principal do orquestrador finalizado.")


    async def _process_market_data(self):
        """Processa dados de mercado em tempo real"""
        if not self.feed_client: return 

        while self.running:
            try:
                tick: Optional[TickData] = await self.feed_client.get_tick() 
                if not tick:
                    await asyncio.sleep(0.001)  
                    continue

                self.tick_count += 1
                await self.data_manager.store_tick(tick)
                market_context = await self._build_market_context(tick)

                active_strategy_names = list(self.active_strategies) 
                if not active_strategy_names:
                    await asyncio.sleep(0.01) 
                    continue

                for strategy_name in active_strategy_names:
                    if strategy_name not in self.strategies:
                        logger.warning(f"Estrategia '{strategy_name}' esta ativa mas nao encontrada. Removendo.")
                        self.active_strategies.discard(strategy_name)
                        continue

                    strategy = self.strategies[strategy_name]
                    if not strategy.active: 
                        logger.debug(f"Estrategia {strategy_name} nao esta ativa, pulando processamento de tick.")
                        continue

                    signal: Optional[Signal] = await strategy.on_tick(market_context) 

                    if signal and signal.is_valid(): 
                        logger.info(f"Sinal gerado por {strategy_name}: {signal.side} {CONFIG.SYMBOL} @ {signal.entry_price or 'Market'}")
                        if await self.risk_manager.can_open_new_position( 
                            signal,
                            await self.execution_engine.get_account_balance(),
                            await self.execution_engine.get_open_positions(),
                            margin_level_pct=(await self.execution_engine.trade_client.get_account_info()).get('MarginLevel') if self.execution_engine.trade_client else None, # type: ignore
                            recent_trades_for_cb=await self.data_manager.get_recent_closed_trades(count=20) # type: ignore
                            ):
                            await self._execute_signal(strategy_name, signal, market_context)
                        else:
                            logger.info(f"Sinal de {strategy_name} nao permitido pela gestao de risco.")
            except asyncio.CancelledError:
                logger.info("Tarefa _process_market_data cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao processar dados de mercado:") 
                await asyncio.sleep(1)  

    async def _update_regime_and_strategies(self): 
        """Atualiza deteccao de regime e scores/selecao de estrategias periodicamente."""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                if (now - self.last_regime_update).total_seconds() >= (CONFIG.REGIME_UPDATE_MS / 1000.0):
                    recent_ticks_df = await self.data_manager.get_historical_ticks( 
                        symbol=CONFIG.SYMBOL,
                        days=2 
                    )
                    if recent_ticks_df is not None and not recent_ticks_df.empty:
                        new_regime, confidence = await self.regime_detector.detect_regime(recent_ticks_df)
                        if confidence >= CONFIG.REGIME_CONFIDENCE_THRESHOLD:
                            if new_regime != self.current_regime:
                                logger.info(f"Mudanca de regime: {self.current_regime} -> {new_regime} (confianca: {confidence:.2%})")
                                self.current_regime = new_regime
                        else:
                            logger.debug(f"Regime detectado {new_regime} com baixa confianca ({confidence:.2%}). Mantendo regime atual: {self.current_regime}")
                        self.last_regime_update = now
                    else:
                        logger.warning("Nao foi possivel obter ticks recentes para atualizacao de regime.")


                if (now - self.last_score_update).total_seconds() / 60.0 >= CONFIG.SCORE_UPDATE_MINUTES or \
                   (self.trade_count > 0 and CONFIG.SCORE_UPDATE_TRADES > 0 and self.trade_count % CONFIG.SCORE_UPDATE_TRADES == 0): 
                    logger.info("Atualizando scores e selecionando estrategias ativas...")
                    await self._update_strategy_scores_and_select()
                    self.last_score_update = now

                await asyncio.sleep(30)  

            except asyncio.CancelledError:
                logger.info("Tarefa _update_regime_and_strategies cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao atualizar regime e estrategias:")
                await asyncio.sleep(60) 


    async def _update_strategy_scores_and_select(self): 
        """Calcula scores e entao seleciona estrategias ativas."""
        for strategy_name, strategy_instance in self.strategies.items(): 
            try:
                performance_metrics_dict = await self.data_manager.get_strategy_performance(
                    strategy_name,
                    days=30 
                )
                if performance_metrics_dict and performance_metrics_dict.get('total_trades', 0) > 5: 
                    # Construir objeto PerformanceMetrics a partir do dicionario
                    # Os campos em performance_metrics_dict devem corresponder aos de PerformanceMetrics
                    # Se nao corresponderem, mapear manualmente ou ajustar get_strategy_performance
                    try:
                        perf_obj = PerformanceMetrics(**performance_metrics_dict)
                        score = self.scorer.calculate_final_score(perf_obj)
                    except TypeError as te:
                        logger.error(f"Erro ao criar PerformanceMetrics para {strategy_name} com dados: {performance_metrics_dict}. Erro: {te}")
                        score = 0.0 # Ou score anterior

                    self.strategy_scores[strategy_name] = score
                    logger.debug(f"Score para {strategy_name}: {score:.4f} (Trades: {performance_metrics_dict.get('total_trades')})")
                else:
                    if strategy_name not in self.strategy_scores:
                        self.strategy_scores[strategy_name] = 0.0 
                    logger.debug(f"Performance insuficiente para calcular novo score para {strategy_name}. Mantendo score: {self.strategy_scores[strategy_name]:.4f}")
            except Exception as e:
                logger.error(f"Erro ao calcular score para {strategy_name}: {e}")
                if strategy_name not in self.strategy_scores:
                     self.strategy_scores[strategy_name] = 0.0

        await self._select_active_strategies()


    async def _select_active_strategies(self):
        """Seleciona as melhores estrategias para o regime atual baseado nos scores."""
        if not self.current_regime:
            logger.warning("Regime de mercado atual nao definido. Nenhuma estrategia sera ativada.")
            return

        logger.info(f"Selecionando estrategias para regime: {self.current_regime}")
        suitable_strategies = []
        for name, strategy_instance in self.strategies.items(): 
            if self.current_regime in strategy_instance.suitable_regimes:
                score = self.strategy_scores.get(name, 0.0) 
                if score > (getattr(CONFIG, 'MIN_STRATEGY_SCORE_TO_ACTIVATE', 0.3)): 
                    suitable_strategies.append((name, score))

        sorted_strategies = sorted(
            suitable_strategies,
            key=lambda x: x[1],
            reverse=True
        )[:CONFIG.MAX_ACTIVE_STRATEGIES]

        newly_active_strategy_names = {name for name, _ in sorted_strategies} 

        strategies_to_deactivate = self.active_strategies - newly_active_strategy_names
        for strategy_name in strategies_to_deactivate:
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].deactivate_strategy() 
                logger.info(f"Estrategia {strategy_name} desativada.")

        strategies_to_activate = newly_active_strategy_names - self.active_strategies
        for strategy_name in strategies_to_activate:
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].activate_strategy() 
                logger.info(f"Estrategia {strategy_name} ativada (Score: {self.strategy_scores.get(strategy_name, 0.0):.4f}, Regime: {self.current_regime}).")

        self.active_strategies = newly_active_strategy_names
        logger.info(f"Estrategias ativas: {self.active_strategies if self.active_strategies else 'Nenhuma'}")


    async def _monitor_open_positions(self): 
        """Monitora posicoes abertas para saidas e trailing stops."""
        while self.running:
            try:
                open_positions_list = await self.execution_engine.get_open_positions() 
                if not open_positions_list:
                    await asyncio.sleep(1) 
                    continue
                
                last_tick_obj_list = await self.data_manager.get_recent_ticks(CONFIG.SYMBOL, 1) 
                if not last_tick_obj_list:
                    logger.warning("Nao foi possivel obter ultimo tick para monitorar posicoes.")
                    await asyncio.sleep(5)
                    continue
                
                market_context_for_exit = await self._build_market_context(last_tick_obj_list[0])


                for position_item in open_positions_list: 
                    strategy_pos_obj = StrategyPosition(
                        id=str(position_item.id), 
                        strategy_name=str(position_item.strategy_name or "Unknown"), 
                        symbol=str(position_item.symbol), 
                        side=str(position_item.side), 
                        entry_price=float(position_item.entry_price), 
                        size=float(position_item.size), 
                        stop_loss=float(position_item.stop_loss) if position_item.stop_loss is not None else None, 
                        take_profit=float(position_item.take_profit) if position_item.take_profit is not None else None, 
                        open_time=position_item.open_time, 
                        unrealized_pnl=float(position_item.pnl) if hasattr(position_item, 'pnl') else 0.0, 
                        metadata=position_item.metadata or {} 
                    )

                    strategy_name = strategy_pos_obj.strategy_name
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]

                        if not strategy.active and strategy_name not in self.active_strategies:
                            logger.info(f"Estrategia {strategy_name} nao esta mais ativa. Fechando posicao {strategy_pos_obj.id}.")
                            await self.execution_engine.close_position(strategy_pos_obj.id, reason="Estrategia desativada")
                            continue 

                        exit_signal: Optional[ExitSignal] = await strategy.evaluate_exit_conditions( 
                            strategy_pos_obj, 
                            market_context_for_exit 
                        )
                        if exit_signal:
                            logger.info(f"Sinal de SAIDA de {strategy_name} para posicao {strategy_pos_obj.id}: {exit_signal.reason}")
                            await self.execution_engine.close_position(
                                exit_signal.position_id_to_close, 
                                reason=exit_signal.reason,
                                volume_to_close=exit_signal.exit_size_lots
                            )
                            continue 

                        if strategy_pos_obj.metadata.get('trailing_stop_active', False): 
                            pass

                    else:
                        logger.warning(f"Estrategia '{strategy_name}' para a posicao {strategy_pos_obj.id} nao encontrada.")

                await asyncio.sleep(CONFIG.ORDER_TIMEOUT_MS / 10000.0 if CONFIG.ORDER_TIMEOUT_MS > 0 else 1.0) 

            except asyncio.CancelledError:
                logger.info("Tarefa _monitor_open_positions cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao monitorar posicoes abertas:")
                await asyncio.sleep(5)


    async def _perform_risk_checks(self): 
        """Verifica limites de risco globais continuamente."""
        while self.running:
            try:
                current_balance = await self.execution_engine.get_account_balance()
                if self.session_start_balance > 0: 
                    self.daily_pnl_pct = (current_balance - self.session_start_balance) / self.session_start_balance
                else:
                    self.daily_pnl_pct = 0.0

                if self.daily_pnl_pct <= -CONFIG.DAILY_LOSS_LIMIT:
                    logger.critical(f"LIMITE DE PERDA DIARIA ATINGIDO: {self.daily_pnl_pct:.2%}. Parando trading para hoje.")
                    await self._stop_trading_session("Limite de perda diaria atingido") 
                    return 

                elif self.daily_pnl_pct >= CONFIG.TARGET_DAILY_PROFIT:
                    logger.info(f"META DE LUCRO DIARIO ATINGIDA: {self.daily_pnl_pct:.2%}. Parando trading para hoje.")
                    await self._stop_trading_session("Meta de lucro diario atingida")
                    return 
                
                if self.risk_manager.circuit_breaker.state == CircuitBreakerState.OPEN:
                     logger.info("RiskManager reportou circuit breaker ativo. Nenhuma nova trade.")
                     pass 

                await asyncio.sleep(10)  

            except asyncio.CancelledError:
                logger.info("Tarefa _perform_risk_checks cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao verificar limites de risco globais:")
                await asyncio.sleep(30)

    async def _build_market_context(self, current_tick: TickData) -> Dict[str, Any]: 
        """Constroi contexto de mercado para as estrategias."""
        dom_snapshot: Optional[DOMSnapshot] = None
        if self.feed_client: 
            dom_snapshot = await self.feed_client.get_dom_snapshot(CONFIG.SYMBOL)

        recent_ticks_list = await self.data_manager.get_recent_ticks(CONFIG.SYMBOL, count=1000)
        current_volatility = await self.data_manager.calculate_volatility(CONFIG.SYMBOL, period=20)
        account_balance = await self.execution_engine.get_account_balance() # Obter balanco aqui

        return {
            "tick": current_tick, 
            "regime": self.current_regime,
            "dom": dom_snapshot,
            "recent_ticks": recent_ticks_list, 
            "volatility": current_volatility, 
            "spread": current_tick.spread if current_tick else 0.0,
            "session": self._get_trading_session(),
            "risk_available": await self.risk_manager.get_available_risk_for_next_trade(account_balance), 
            "timestamp": current_tick.timestamp if current_tick else datetime.now(timezone.utc) 
        }

    async def _execute_signal(self, strategy_name: str, signal: Signal, market_context: Dict[str, Any]): 
        """Executa sinal de trading."""
        try:
            account_balance_for_sizing = await self.execution_engine.get_account_balance()
            position_size_result = await self.risk_manager.calculate_position_size(
                signal, 
                account_balance_for_sizing,
                market_conditions_for_sizer={'volatility': market_context.get('volatility')} 
            )

            if position_size_result and position_size_result.lot_size > 0:
                entry_price_for_order = None 
                order_type_to_send = EngineOrderType.MARKET 
                
                current_tick_from_context = market_context.get('tick')
                if not current_tick_from_context:
                    logger.error(f"Nao foi possivel obter tick atual do contexto para executar sinal de {strategy_name}")
                    return

                if signal.order_type.upper() == "LIMIT" and signal.entry_price is not None:
                     order_type_to_send = EngineOrderType.LIMIT
                     entry_price_for_order = signal.entry_price
                
                order: Optional[EngineOrder] = await self.execution_engine.create_order( 
                    symbol=CONFIG.SYMBOL,
                    side=signal.side.capitalize(), 
                    size=position_size_result.lot_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_name=strategy_name, 
                    order_type=order_type_to_send, 
                    price=entry_price_for_order,
                    metadata={ 
                        'signal_confidence': signal.confidence,
                        'signal_reason': signal.reason,
                        'risk_amount_calc': position_size_result.risk_amount_currency, 
                        'risk_percent_calc': position_size_result.risk_percent_of_balance 
                    }
                )

                if order and order.broker_order_id: 
                    self.trade_count += 1 
                    logger.info(f"Ordem {order.id} (BrokerID: {order.broker_order_id}) submetida por {strategy_name}. Tamanho: {order.size}")
                elif order: 
                    logger.warning(f"Ordem {order.id} criada por {strategy_name} mas falhou na submissao ou nao tem BrokerID.")
                else:
                    logger.warning(f"Falha ao criar ordem para sinal de {strategy_name}.")

            else:
                logger.info(f"Tamanho da posicao calculado como zero ou invalido para sinal de {strategy_name}. Nenhuma ordem criada.")

        except Exception as e:
            logger.exception(f"Erro ao executar sinal da estrategia {strategy_name}:")


    async def _stop_trading_session(self, reason: str): 
        """Para operacoes de trading para a sessao atual (ex: dia)."""
        logger.warning(f"Parando operacoes de trading para a sessao/dia. Razao: {reason}")

        for strategy_name in list(self.active_strategies): 
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].deactivate_strategy() 
                logger.info(f"Estrategia {strategy_name} desativada devido a parada da sessao.")
        
        logger.info("Fechando todas as posicoes abertas devido a parada da sessao...")
        await self.execution_engine.close_all_positions(reason)

        logger.info("Cancelando todas as ordens pendentes devido a parada da sessao...")
        await self.execution_engine.cancel_all_orders()


    async def _schedule_restart_next_day(self): 
        """Agenda reinicio das operacoes para o proximo dia de trading."""
        now = datetime.now(timezone.utc)
        
        london_start_hour = CONFIG.SESSION_CONFIG.get('LONDON', {}).get('start_hour', 7) 
        next_trading_day_start = (now + timedelta(days=1)).replace(hour=london_start_hour-1, minute=0, second=0, microsecond=0) 
        
        wait_seconds = (next_trading_day_start - now).total_seconds()
        if wait_seconds <=0: 
            next_trading_day_start = (now + timedelta(days=2)).replace(hour=london_start_hour-1, minute=0, second=0, microsecond=0)
            wait_seconds = (next_trading_day_start - now).total_seconds()

        logger.info(f"Trading pausado. Reinicio das operacoes agendado para {next_trading_day_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if wait_seconds > 0 :
            await asyncio.sleep(wait_seconds)

        logger.info("Reiniciando operacoes de trading apos pausa programada.")
        self.session_start_balance = await self.execution_engine.get_account_balance() 
        self.daily_pnl_pct = 0.0
        self.max_drawdown_pct = 0.0
        await self.risk_manager.daily_session_reset(self.session_start_balance) 


    def _get_trading_session(self) -> str:
        """Retorna sessao de trading atual (UTC)."""
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        
        overlap_cfg = CONFIG.SESSION_CONFIG.get('OVERLAP') 
        if overlap_cfg and overlap_cfg.get('start_hour', -1) <= current_hour < overlap_cfg.get('end_hour', -1):
            return "Overlap"

        for session_name, cfg in CONFIG.SESSION_CONFIG.items():
            if session_name == 'OVERLAP': continue 
            start = cfg.get('start_hour', -1)
            end = cfg.get('end_hour', -1)
            if start == -1 or end == -1: continue 

            if start > end:  
                if current_hour >= start or current_hour < end:
                    return session_name.capitalize()
            else:
                if start <= current_hour < end:
                    return session_name.capitalize()
        return "Transition" 


    async def check_feed_connection(self) -> bool:
        """Verifica conexao do feed."""
        return self.feed_client.is_connected() if self.feed_client else False

    async def check_trade_connection(self) -> bool:
        """Verifica conexao de trading."""
        return self.trade_client.is_connected() if self.trade_client else False

    async def get_latency(self) -> float:
        """Obtem latencia atual do feed em ms."""
        return await self.feed_client.get_latency() if self.feed_client else -1.0

    async def get_current_drawdown(self) -> float: 
        """Obtem drawdown atual em percentual."""
        current_balance = await self.execution_engine.get_account_balance()
        return await self.risk_manager.get_current_session_drawdown_pct(current_balance)


    async def shutdown(self):
        """Desliga o orquestrador e seus componentes de forma graciosa."""
        if not self.running: 
            return
        logger.info("Desligando orquestrador...")
        self.running = False 

        for task in self._main_loop_tasks:
            if task and not task.done():
                task.cancel()
        if self._main_loop_tasks:
            await asyncio.gather(*self._main_loop_tasks, return_exceptions=True) 


        await self._stop_trading_session("Desligamento do sistema")

        if self.feed_client:
            await self.feed_client.disconnect("Desligamento do orquestrador")
        if self.trade_client:
            await self.trade_client.disconnect("Desligamento do orquestrador")


        if self.data_manager:
            await self.data_manager.save_state()
        
        if self.risk_manager:
            await self.risk_manager.shutdown()

        logger.info("Orquestrador desligado com sucesso.")