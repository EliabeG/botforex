# core/orchestrator.py
import asyncio
from typing import Dict, List, Optional, Set, Any # Adicionado Any
from datetime import datetime, timedelta, timezone # Adicionado timezone
import numpy as np # Embora não usado diretamente, pode ser usado por subcomponentes
from collections import defaultdict

from config.settings import CONFIG
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.data_manager import DataManager
from core.execution_engine import ExecutionEngine, Order as EngineOrder, OrderStatus as EngineOrderStatus # Renomeado para evitar conflito
from strategies.base_strategy import BaseStrategy, Signal, Position as StrategyPosition, ExitSignal # Renomeado para evitar conflito
from optimization.scoring import StrategyScorer
from risk.risk_manager import RiskManager
from api.ticktrader_ws import TickTraderFeed, TickTraderTrade, TickData, DOMSnapshot # Adicionado DOMSnapshot
from utils.logger import setup_logger

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

        # Conexões WebSocket
        # Estes serão inicializados em _connect_websockets
        self.feed_client: Optional[TickTraderFeed] = None
        self.trade_client: Optional[TickTraderTrade] = None


        # Estratégias
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: Set[str] = set()
        self.strategy_scores: Dict[str, float] = {}

        # Estado
        self.current_regime: Optional[str] = None # Adicionada tipagem
        self.last_regime_update: datetime = datetime.now(timezone.utc) # Usar UTC
        self.last_score_update: datetime = datetime.now(timezone.utc) # Usar UTC
        self.tick_count: int = 0
        self.trade_count: int = 0 # Contador de ordens executadas (ou trades fechados)

        # Métricas
        self.daily_pnl_pct: float = 0.0 # Renomeado de daily_pnl para clareza que é percentual
        self.session_start_balance: float = 0.0
        self.max_drawdown_pct: float = 0.0 # Renomeado de max_drawdown para clareza

        self._main_loop_tasks: List[asyncio.Task] = [] # Para rastrear tarefas principais


    async def initialize(self):
        """Inicializa todos os componentes"""
        logger.info(f"Inicializando orquestrador em modo {self.mode}...")

        try:
            # Conectar WebSockets PRIMEIRO, pois outros componentes podem depender deles
            await self._connect_websockets()
            if not (self.feed_client and self.feed_client.is_connected() and \
                    self.trade_client and self.trade_client.is_connected()):
                logger.critical("Falha ao conectar WebSockets. Orquestrador não pode continuar.")
                raise ConnectionError("Falha na conexão WebSocket inicial.")


            # Inicializar gerenciador de dados
            await self.data_manager.initialize() # DataManager agora pode usar CONFIG para caminhos

            # Carregar histórico se necessário (principalmente para backtest ou treino inicial)
            if self.mode != "live":
                await self._load_historical_data_for_init() # Renomeado para clareza

            # Inicializar motor de execução
            # O trade_client já deve estar conectado aqui
            await self.execution_engine.initialize(self.trade_client)


            # Carregar e inicializar estratégias
            await self._load_strategies()

            # Treinar detector de regime (ou carregar modelo treinado)
            await self._initialize_regime_detector()


            # Obter balanço inicial e inicializar RiskManager
            self.session_start_balance = await self.execution_engine.get_account_balance()
            if self.session_start_balance == 0.0 and self.mode == "live": # Checagem crítica
                logger.warning("Balanço inicial da conta é 0. Verifique a conexão ou a conta.")
                # Poderia levantar um erro aqui se for crítico
            await self.risk_manager.initialize(self.session_start_balance)


            logger.info("Orquestrador inicializado com sucesso")

        except Exception as e:
            logger.exception("Erro na inicialização do orquestrador:") # Usar logger.exception
            raise

    async def _connect_websockets(self):
        """Conecta aos WebSockets do TickTrader"""
        logger.info("Conectando WebSockets...")
        try:
            self.feed_client = TickTraderFeed()
            self.trade_client = TickTraderTrade()

            await self.feed_client.connect() # connect já autentica e inicia _process_messages
            if not self.feed_client.is_connected():
                raise ConnectionError("Falha ao conectar/autenticar no Feed WebSocket.")

            # Inscrever no símbolo após conexão e autenticação do feed
            await self.feed_client.subscribe_symbol(CONFIG.SYMBOL)
            await self.feed_client.subscribe_dom(CONFIG.SYMBOL, CONFIG.DOM_LEVELS)

            await self.trade_client.connect() # connect já autentica e inicia _process_messages
            if not self.trade_client.is_connected():
                raise ConnectionError("Falha ao conectar/autenticar no Trade WebSocket.")

            logger.info("WebSockets conectados e autenticados.")
        except Exception as e:
            logger.exception("Erro crítico durante conexão WebSocket:")
            # Se a conexão falhar aqui, o bot não pode operar.
            # Limpar clientes para evitar uso de instâncias não conectadas.
            if self.feed_client: await self.feed_client.disconnect("Falha na conexão inicial")
            if self.trade_client: await self.trade_client.disconnect("Falha na conexão inicial")
            self.feed_client = None
            self.trade_client = None
            raise # Relançar para que a inicialização falhe


    async def _load_historical_data_for_init(self):
        """Carrega dados históricos para inicialização (ex: backtest ou treino de modelo)."""
        # Esta função seria chamada se, por exemplo, o modo de backtest precisasse
        # preencher o DataManager com um conjunto de dados específico.
        # Para treino de regime, é chamado separadamente.
        logger.info("Carregando dados históricos para inicialização (se aplicável ao modo)...")
        # Exemplo:
        # if self.mode == "backtest":
        #     start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        #     end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)
        #     await self.data_manager.download_and_store_historical_range(CONFIG.SYMBOL, start_date, end_date)
        pass # Implementar conforme necessidade do modo


    async def _load_strategies(self):
        """Carrega todas as estratégias disponíveis"""
        logger.info("Carregando estratégias...")
        from strategies import load_all_strategies # Importação local para evitar dependência circular no topo

        strategy_classes = load_all_strategies()

        for strategy_class in strategy_classes:
            try:
                strategy_instance = strategy_class() # Renomeado para evitar conflito
                self.strategies[strategy_instance.name] = strategy_instance
                await strategy_instance.initialize() # Chamada async

                params = await self.data_manager.load_strategy_params(strategy_instance.name)
                if params:
                    strategy_instance.update_parameters(params)
                    logger.info(f"Parâmetros carregados para {strategy_instance.name}")

            except Exception as e:
                logger.exception(f"Erro ao carregar estratégia {strategy_class.__name__}:")

        logger.info(f"{len(self.strategies)} estratégias carregadas e inicializadas.")

    async def _initialize_regime_detector(self): # Renomeado de _train_regime_detector
        """Inicializa o detector de regime, treinando ou carregando modelo."""
        logger.info("Inicializando detector de regime...")
        try:
            await self.regime_detector.load_model() # Tentar carregar primeiro
            if not self.regime_detector.is_trained:
                logger.info("Modelo de regime não encontrado ou falhou ao carregar. Tentando treinar...")
                # Obter dados históricos suficientes para treino
                # REGIME_CONFIG.TREND_WINDOW + buffer (ex: 50)
                required_ticks_for_train = (REGIME_CONFIG.TREND_WINDOW or 250) + 50
                days_for_train = max(30, (required_ticks_for_train // (24*60)) + 2) # Estimativa de dias (assumindo 1 tick/min)

                historical_data_df = await self.data_manager.get_historical_ticks(
                    symbol=CONFIG.SYMBOL,
                    days=days_for_train # Ajustar conforme necessidade de dados para features
                )

                if historical_data_df is not None and not historical_data_df.empty:
                    await self.regime_detector.train(historical_data_df)
                    if self.regime_detector.is_trained:
                        logger.info("Detector de regime treinado com sucesso.")
                    else:
                        logger.warning("Treinamento do detector de regime falhou. Operando sem ML para regime.")
                else:
                    logger.warning("Sem dados históricos suficientes para treinar detector de regime. Operando sem ML para regime.")
            else:
                logger.info("Modelo de detector de regime carregado com sucesso.")

        except Exception as e:
            logger.exception("Erro ao inicializar detector de regime:")


    async def run(self):
        """Loop principal de execução"""
        if not self.feed_client or not self.trade_client:
            logger.critical("Clientes WebSocket não estão inicializados. Encerrando o Orchestrator.")
            return

        self.running = True
        logger.info("Iniciando loop principal do orquestrador")

        self._main_loop_tasks = [ # Armazenar tarefas para cancelamento gracioso
            asyncio.create_task(self._process_market_data(), name="ProcessMarketData"),
            asyncio.create_task(self._update_regime_and_strategies(), name="UpdateRegimeStrategies"),
            # _update_scores foi integrado em _update_regime_and_strategies
            asyncio.create_task(self._monitor_open_positions(), name="MonitorPositions"), # Renomeado
            asyncio.create_task(self._perform_risk_checks(), name="RiskChecks") # Renomeado
        ]

        try:
            await asyncio.gather(*self._main_loop_tasks)
        except asyncio.CancelledError:
            logger.info("Loop principal do orquestrador cancelado.")
        except Exception as e:
            logger.exception("Erro crítico no loop principal do orquestrador:") # Usar logger.exception
            # Considerar um shutdown gracioso aqui também
            await self.shutdown()
        finally:
            logger.info("Loop principal do orquestrador finalizado.")


    async def _process_market_data(self):
        """Processa dados de mercado em tempo real"""
        if not self.feed_client: return # Safety check

        while self.running:
            try:
                tick: Optional[TickData] = await self.feed_client.get_tick() # get_tick já é async
                if not tick:
                    await asyncio.sleep(0.001)  # Pequena pausa se não houver ticks
                    continue

                self.tick_count += 1
                await self.data_manager.store_tick(tick)
                market_context = await self._build_market_context(tick)

                active_strategy_names = list(self.active_strategies) # Copiar para evitar problemas de modificação durante iteração
                if not active_strategy_names:
                    # logger.debug("Nenhuma estratégia ativa para processar o tick.")
                    await asyncio.sleep(0.01) # Pausa maior se não há estratégias
                    continue

                # Processar sinais de forma concorrente se houver muitas estratégias
                # Por enquanto, sequencial para simplicidade se o número de estratégias ativas for pequeno
                for strategy_name in active_strategy_names:
                    if strategy_name not in self.strategies:
                        logger.warning(f"Estratégia '{strategy_name}' está ativa mas não encontrada. Removendo.")
                        self.active_strategies.discard(strategy_name)
                        continue

                    strategy = self.strategies[strategy_name]
                    if not strategy.active: # Dupla checagem
                        logger.debug(f"Estratégia {strategy_name} não está ativa, pulando processamento de tick.")
                        continue


                    signal: Optional[Signal] = await strategy.process_tick(market_context)

                    if signal and signal.is_valid(): # is_valid() já está na classe Signal
                        # Adicionar lógica para verificar se já existe uma posição para esta estratégia
                        # ou se o sinal é para fechar/modificar uma posição existente.
                        # A lógica atual foca em abrir novas posições.
                        logger.info(f"Sinal gerado por {strategy_name}: {signal.side} {CONFIG.SYMBOL} @ {signal.entry_price or 'Market'}")
                        if await self.risk_manager.can_open_position(signal):
                            await self._execute_signal(strategy_name, signal, market_context)
                        else:
                            logger.info(f"Sinal de {strategy_name} não permitido pela gestão de risco.")


            except asyncio.CancelledError:
                logger.info("Tarefa _process_market_data cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao processar dados de mercado:") # Usar logger.exception
                await asyncio.sleep(1)  # Pausa para evitar loop de erro rápido

    async def _update_regime_and_strategies(self): # Combinado _update_regime e _update_scores
        """Atualiza detecção de regime e scores/seleção de estratégias periodicamente."""
        while self.running:
            try:
                # Atualizar Regime
                now = datetime.now(timezone.utc)
                if (now - self.last_regime_update).total_seconds() >= (CONFIG.REGIME_UPDATE_MS / 1000.0):
                    recent_ticks_df = await self.data_manager.get_historical_ticks( # get_historical_ticks retorna DataFrame
                        symbol=CONFIG.SYMBOL,
                        days=2 # Dias suficientes para calcular features de regime (ajustar)
                    )
                    if recent_ticks_df is not None and not recent_ticks_df.empty:
                        new_regime, confidence = await self.regime_detector.detect_regime(recent_ticks_df)
                        if confidence >= CONFIG.REGIME_CONFIDENCE_THRESHOLD:
                            if new_regime != self.current_regime:
                                logger.info(f"Mudança de regime: {self.current_regime} → {new_regime} (confiança: {confidence:.2%})")
                                self.current_regime = new_regime
                                # Resselecionar estratégias é feito após atualização de scores
                        else:
                            logger.debug(f"Regime detectado {new_regime} com baixa confiança ({confidence:.2%}). Mantendo regime atual: {self.current_regime}")
                        self.last_regime_update = now
                    else:
                        logger.warning("Não foi possível obter ticks recentes para atualização de regime.")


                # Atualizar Scores e Selecionar Estratégias
                if (now - self.last_score_update).total_seconds() / 60.0 >= CONFIG.SCORE_UPDATE_MINUTES or \
                   (self.trade_count > 0 and self.trade_count % CONFIG.SCORE_UPDATE_TRADES == 0): # Evitar divisão por zero
                    logger.info("Atualizando scores e selecionando estratégias ativas...")
                    await self._update_strategy_scores_and_select()
                    self.last_score_update = now

                await asyncio.sleep(30)  # Verificar condições de atualização a cada 30s

            except asyncio.CancelledError:
                logger.info("Tarefa _update_regime_and_strategies cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao atualizar regime e estratégias:")
                await asyncio.sleep(60) # Pausa maior em caso de erro


    async def _update_strategy_scores_and_select(self): # Novo método combinado
        """Calcula scores e então seleciona estratégias ativas."""
        for strategy_name, strategy_instance in self.strategies.items(): # Renomeado strategy para strategy_instance
            try:
                # Obter performance dos últimos 30 dias para scoring
                # A performance pode ser uma dataclass ou dict
                performance_metrics_dict = await self.data_manager.get_strategy_performance(
                    strategy_name,
                    days=30 # Usar um período padrão para scoring
                )
                if performance_metrics_dict and performance_metrics_dict.get('total_trades', 0) > 5: # Min trades para score
                    score = self.scorer.calculate_score(performance_metrics_dict)
                    self.strategy_scores[strategy_name] = score
                    logger.debug(f"Score para {strategy_name}: {score:.4f} (Trades: {performance_metrics_dict.get('total_trades')})")
                else:
                    # Se não houver performance suficiente, manter score antigo ou default
                    if strategy_name not in self.strategy_scores:
                        self.strategy_scores[strategy_name] = 0.0 # Default score
                    logger.debug(f"Performance insuficiente para calcular novo score para {strategy_name}. Mantendo score: {self.strategy_scores[strategy_name]:.4f}")
            except Exception as e:
                logger.error(f"Erro ao calcular score para {strategy_name}: {e}")
                if strategy_name not in self.strategy_scores:
                     self.strategy_scores[strategy_name] = 0.0

        await self._select_active_strategies()


    async def _select_active_strategies(self):
        """Seleciona as melhores estratégias para o regime atual baseado nos scores."""
        if not self.current_regime:
            logger.warning("Regime de mercado atual não definido. Nenhuma estratégia será ativada.")
            # Desativar todas se o regime for desconhecido?
            # for strategy_name in list(self.active_strategies):
            #     await self.strategies[strategy_name].deactivate()
            #     logger.info(f"Estratégia {strategy_name} desativada devido a regime indefinido.")
            # self.active_strategies.clear()
            return

        logger.info(f"Selecionando estratégias para regime: {self.current_regime}")
        suitable_strategies = []
        for name, strategy_instance in self.strategies.items(): # Renomeado strategy para strategy_instance
            if self.current_regime in strategy_instance.suitable_regimes:
                score = self.strategy_scores.get(name, 0.0) # Usar 0.0 se não houver score
                # Adicionar filtro de score mínimo para considerar uma estratégia
                if score > (getattr(CONFIG, 'MIN_STRATEGY_SCORE_TO_ACTIVATE', 0.3)): # Ex: score mínimo 0.3
                    suitable_strategies.append((name, score))

        # Ordenar por score e selecionar TOP N
        sorted_strategies = sorted(
            suitable_strategies,
            key=lambda x: x[1],
            reverse=True
        )[:CONFIG.MAX_ACTIVE_STRATEGIES]

        newly_active_strategy_names = {name for name, _ in sorted_strategies} # Renomeado

        # Desativar estratégias que não estão mais na lista de ativas
        strategies_to_deactivate = self.active_strategies - newly_active_strategy_names
        for strategy_name in strategies_to_deactivate:
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].deactivate()
                logger.info(f"Estratégia {strategy_name} desativada.")

        # Ativar novas estratégias
        strategies_to_activate = newly_active_strategy_names - self.active_strategies
        for strategy_name in strategies_to_activate:
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].activate()
                logger.info(f"Estratégia {strategy_name} ativada (Score: {self.strategy_scores.get(strategy_name, 0.0):.4f}, Regime: {self.current_regime}).")

        self.active_strategies = newly_active_strategy_names
        logger.info(f"Estratégias ativas: {self.active_strategies if self.active_strategies else 'Nenhuma'}")


    async def _monitor_open_positions(self): # Renomeado de _monitor_positions
        """Monitora posições abertas para saídas e trailing stops."""
        while self.running:
            try:
                open_positions_list = await self.execution_engine.get_open_positions() # Retorna List[Position] do ExecutionEngine
                if not open_positions_list:
                    await asyncio.sleep(1) # Pausa curta se não houver posições
                    continue

                # Obter preço de mercado atual uma vez para todas as posições (se for o mesmo símbolo)
                # Se múltiplos símbolos, obter dentro do loop
                current_market_price = await self.data_manager.get_current_price(CONFIG.SYMBOL)
                if current_market_price == 0.0: # Se não conseguir preço, pular este ciclo de monitoramento
                    logger.warning("Não foi possível obter preço de mercado atual para monitorar posições.")
                    await asyncio.sleep(5) # Pausa maior
                    continue


                for position in open_positions_list: # position é do tipo StrategyPosition aqui
                    strategy_name = position.strategy_name
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]

                        # Verificar se a estratégia ainda está ativa ou se deve fechar posições de estratégias inativas
                        if not strategy.active and strategy_name not in self.active_strategies:
                            logger.info(f"Estratégia {strategy_name} não está mais ativa. Fechando posição {position.id}.")
                            await self.execution_engine.close_position(position.id, reason="Estratégia desativada")
                            continue # Próxima posição


                        # Verificar condições de saída da estratégia
                        exit_signal: Optional[ExitSignal] = await strategy.check_exit_conditions(
                            position,
                            current_market_price
                        )
                        if exit_signal:
                            logger.info(f"Sinal de SAÍDA de {strategy_name} para posição {position.id}: {exit_signal.reason}")
                            await self.execution_engine.close_position(
                                exit_signal.position_id, # Usar position_id do exit_signal
                                reason=exit_signal.reason,
                                partial_volume=position.size * exit_signal.partial_exit if exit_signal.partial_exit < 1.0 else None
                            )
                            continue # Posição foi (ou está sendo) fechada

                        # Atualizar trailing stop se a estratégia o utiliza e se a posição o tem ativo
                        # A classe Position da estratégia deve ter um atributo 'trailing_stop_active' ou similar
                        if hasattr(position, 'trailing_stop_active') and position.trailing_stop_active:
                             # O método update_trailing_stop no ExecutionEngine já pega a posição e faz a lógica
                            await self.execution_engine.update_trailing_stop(position.id, current_market_price)


                    else:
                        logger.warning(f"Estratégia '{strategy_name}' para a posição {position.id} não encontrada. A posição não será gerenciada pela estratégia.")
                        # Considerar uma política de fechamento para posições órfãs.


                await asyncio.sleep(CONFIG.ORDER_TIMEOUT_MS / 10000.0 or 1.0)  # Verificar em intervalos curtos, ex: 1s

            except asyncio.CancelledError:
                logger.info("Tarefa _monitor_open_positions cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao monitorar posições abertas:")
                await asyncio.sleep(5)


    async def _perform_risk_checks(self): # Renomeado de _check_risk_limits
        """Verifica limites de risco globais continuamente."""
        while self.running:
            try:
                current_balance = await self.execution_engine.get_account_balance()
                if self.session_start_balance > 0: # Evitar divisão por zero se saldo inicial for 0
                    self.daily_pnl_pct = (current_balance - self.session_start_balance) / self.session_start_balance
                else:
                    self.daily_pnl_pct = 0.0


                # Verificar limite de perda diária
                if self.daily_pnl_pct <= -CONFIG.DAILY_LOSS_LIMIT:
                    logger.critical(f"LIMITE DE PERDA DIÁRIA ATINGIDO: {self.daily_pnl_pct:.2%}. Parando trading para hoje.")
                    await self._stop_trading_session("Limite de perda diária atingido") # Renomeado
                    # Considerar não reiniciar automaticamente no mesmo dia
                    # await self._schedule_restart_next_day()
                    return # Sair da tarefa se o trading for parado para o dia


                # Verificar meta de lucro diário
                elif self.daily_pnl_pct >= CONFIG.TARGET_DAILY_PROFIT:
                    logger.info(f"META DE LUCRO DIÁRIO ATINGIDA: {self.daily_pnl_pct:.2%}. Parando trading para hoje.")
                    await self._stop_trading_session("Meta de lucro diário atingida")
                    # await self._schedule_restart_next_day()
                    return # Sair da tarefa


                # Calcular drawdown
                # RiskManager poderia expor um método para obter drawdown atual
                # Simulando aqui para ilustração, mas idealmente viria do RiskManager
                # hwm = await self.data_manager.get_high_water_mark() # Supondo que DataManager rastreia HWM da conta
                # current_dd_pct = (hwm - current_balance) / hwm if hwm > 0 else 0.0
                # if current_dd_pct > self.max_drawdown_pct:
                #     self.max_drawdown_pct = current_dd_pct

                # Usar o RiskManager para checar Circuit Breaker
                # Isso requer que RiskManager tenha acesso ao status da conta
                account_status_for_rm = {
                    'current_balance': current_balance,
                    'daily_pnl_pct': self.daily_pnl_pct,
                    # 'current_drawdown': self.max_drawdown_pct, # ou o DD atual do RiskManager
                    # ... outras métricas que CircuitBreaker precise
                }
                # if self.risk_manager.circuit_breaker.state == CircuitBreakerState.OPEN: # Exemplo de acesso
                #     if await self.risk_manager.circuit_breaker.check_conditions_to_reset_or_half_open():
                #         pass
                # elif await self.risk_manager.check_circuit_breaker_conditions(account_status_for_rm):
                #     logger.critical("CIRCUIT BREAKER ATIVADO PELO RISK MANAGER. Parando todas as operações.")
                #     await self._stop_trading_session("Circuit Breaker Ativado")
                #     # A lógica de pausa/reagendamento seria tratada pelo CircuitBreaker ou RiskManager
                #     return # Sair da tarefa

                # O RiskManager também pode ter seu próprio loop para circuit breaker.
                # Simplificando:
                if self.risk_manager.circuit_breaker_active: # Se o RiskManager tiver sua flag
                     logger.info("RiskManager reportou circuit breaker ativo. Nenhuma nova trade.")
                     if await self.risk_manager._check_circuit_breaker_timeout(): # Se pode resetar
                         self.risk_manager.circuit_breaker_active = False
                         logger.info("Circuit breaker do RiskManager resetado após timeout.")
                     else:
                         await self._stop_trading_session("Circuit Breaker (RiskManager) Ativo")
                         # A pausa longa já estaria implícita ou gerenciada pelo RiskManager
                         return


                await asyncio.sleep(10)  # Verificar a cada 10 segundos

            except asyncio.CancelledError:
                logger.info("Tarefa _perform_risk_checks cancelada.")
                break
            except Exception as e:
                logger.exception("Erro ao verificar limites de risco globais:")
                await asyncio.sleep(30)

    async def _build_market_context(self, current_tick: TickData) -> Dict[str, Any]: # Renomeado tick para current_tick
        """Constrói contexto de mercado para as estratégias."""
        # Obter DOM apenas se houver estratégias que o utilizem
        dom_snapshot: Optional[DOMSnapshot] = None
        if any(hasattr(s, 'uses_dom') and s.uses_dom for s in self.strategies.values() if s.active): # Exemplo
            if self.feed_client:
                dom_snapshot = await self.feed_client.get_dom_snapshot(CONFIG.SYMBOL)

        # Obter ticks recentes (ex: últimos 1000 para cálculo de indicadores)
        # O número de ticks deve ser suficiente para o maior período de lookback das estratégias.
        recent_ticks_list = await self.data_manager.get_recent_ticks(CONFIG.SYMBOL, count=1000)

        # Volatilidade (ex: ATR ou std dev de retornos)
        # Poderia ser calculado no DataManager ou aqui se for simples.
        # Por ora, vamos assumir que DataManager pode fornecer isso.
        current_volatility = await self.data_manager.calculate_volatility(CONFIG.SYMBOL, period=20)


        return {
            "tick": current_tick, # O tick mais recente
            "regime": self.current_regime,
            "dom": dom_snapshot,
            "recent_ticks": recent_ticks_list, # Lista de objetos TickData
            "volatility": current_volatility, # Ex: ATR(14) ou std dev normalizado
            "spread": current_tick.spread if current_tick else 0.0,
            "session": self._get_trading_session(),
            "risk_available": await self.risk_manager.get_available_risk(),
            "timestamp": current_tick.timestamp if current_tick else datetime.now(timezone.utc) # Adicionar timestamp ao contexto
        }

    async def _execute_signal(self, strategy_name: str, signal: Signal, market_context: Dict[str, Any]): # Adicionado market_context
        """Executa sinal de trading."""
        try:
            # O RiskManager agora retorna um objeto PositionSizeResult
            # Passar mais contexto para o PositionSizer se necessário
            position_size_result = await self.risk_manager.calculate_position_size(
                signal, # Signal já contém entry_price (se aplicável) e stop_loss
                await self.execution_engine.get_account_balance()
                # Adicionar outros dados relevantes para position sizing:
                # symbol=CONFIG.SYMBOL,
                # leverage=CONFIG.LEVERAGE,
                # market_conditions={'volatility': market_context.get('volatility')}
            )

            if position_size_result and position_size_result.lot_size > 0:
                # Criar ordem com o tamanho calculado
                # O preço de entrada para ordens a mercado é o preço atual de mercado, não signal.entry_price
                # signal.entry_price é mais uma referência ou para ordens limite.
                entry_price_for_order = None # Para Market Order
                order_type_to_send = EngineOrderType.MARKET
                if signal.entry_price and abs(signal.entry_price - market_context['tick'].mid) < (CONFIG.MAX_SPREAD_PIPS / 10000.0 * 5): # Se próximo ao mercado
                    # Poderia ser uma ordem limite se signal.entry_price for específico
                    # order_type_to_send = EngineOrderType.LIMIT
                    # entry_price_for_order = signal.entry_price
                    pass # Mantendo Market por enquanto


                order: Optional[EngineOrder] = await self.execution_engine.create_order( # Retorna EngineOrder
                    symbol=CONFIG.SYMBOL,
                    side=signal.side.capitalize(), # 'Buy' ou 'Sell'
                    size=position_size_result.lot_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_name=strategy_name, # Passar nome da estratégia para rastreamento
                    order_type=order_type_to_send,
                    price=entry_price_for_order,
                    metadata={ # Adicionar metadados relevantes
                        'signal_confidence': signal.confidence,
                        'signal_reason': signal.reason,
                        'risk_amount_calc': position_size_result.risk_amount,
                        'risk_percent_calc': position_size_result.risk_percent
                    }
                )

                if order and order.broker_order_id: # Se a ordem foi submetida e tem ID do broker
                    self.trade_count += 1 # Incrementar contador de trades (ou ordens submetidas)
                    logger.info(f"Ordem {order.id} (BrokerID: {order.broker_order_id}) submetida por {strategy_name}. Tamanho: {order.size}")
                    # O registro do trade no DataManager deve ocorrer APÓS o preenchimento (fill).
                    # O ExecutionEngine lidará com isso através dos callbacks de _handle_order_update_event.
                elif order: # Ordem criada mas não submetida ou falhou
                    logger.warning(f"Ordem {order.id} criada por {strategy_name} mas falhou na submissão ou não tem BrokerID.")
                else:
                    logger.warning(f"Falha ao criar ordem para sinal de {strategy_name}.")

            else:
                logger.info(f"Tamanho da posição calculado como zero ou inválido para sinal de {strategy_name}. Nenhuma ordem criada.")


        except Exception as e:
            logger.exception(f"Erro ao executar sinal da estratégia {strategy_name}:")


    async def _stop_trading_session(self, reason: str): # Renomeado de _stop_trading
        """Para operações de trading para a sessão atual (ex: dia)."""
        logger.warning(f"Parando operações de trading para a sessão/dia. Razão: {reason}")

        # Desativar todas as estratégias ativas temporariamente
        # As estratégias podem ser reativadas no próximo ciclo de seleção se as condições permitirem.
        for strategy_name in list(self.active_strategies): # Iterar sobre cópia
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].deactivate()
                logger.info(f"Estratégia {strategy_name} desativada devido à parada da sessão.")
        # self.active_strategies.clear() # Não limpar, deixar _select_active_strategies gerenciar

        # Fechar todas as posições abertas
        logger.info("Fechando todas as posições abertas devido à parada da sessão...")
        await self.execution_engine.close_all_positions(reason)

        # Cancelar todas as ordens pendentes
        logger.info("Cancelando todas as ordens pendentes devido à parada da sessão...")
        await self.execution_engine.cancel_all_orders()

        # Aqui, o bot não necessariamente "desliga", mas para de tomar novas decisões de trading
        # até que a condição de parada seja revertida (ex: novo dia, reset do circuit breaker).


    async def _schedule_restart_next_day(self): # Novo método
        """Agenda reinício das operações para o próximo dia de trading."""
        now = datetime.now(timezone.utc)
        next_trading_day_start = (now + timedelta(days=1)).replace(hour=CONFIG.SESSION_CONFIG['LONDON']['start_hour']-1, minute=0, second=0, microsecond=0) # Ex: 1h antes da abertura de Londres
        # Adicionar lógica para pular fins de semana

        wait_seconds = (next_trading_day_start - now).total_seconds()
        if wait_seconds <=0: # Se já passou do horário de reinício, agendar para o dia seguinte
            next_trading_day_start = (now + timedelta(days=2)).replace(hour=CONFIG.SESSION_CONFIG['LONDON']['start_hour']-1, minute=0, second=0, microsecond=0)
            wait_seconds = (next_trading_day_start - now).total_seconds()


        logger.info(f"Trading pausado. Reinício das operações agendado para {next_trading_day_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if wait_seconds > 0 :
            await asyncio.sleep(wait_seconds)

        logger.info("Reiniciando operações de trading após pausa programada.")
        self.session_start_balance = await self.execution_engine.get_account_balance() # Resetar balanço inicial
        self.daily_pnl_pct = 0.0
        self.max_drawdown_pct = 0.0
        # RiskManager pode precisar ser resetado para o dia.
        # await self.risk_manager.daily_reset(self.session_start_balance)
        # Estratégias serão reavaliadas e reativadas pelo loop _update_regime_and_strategies


    def _get_trading_session(self) -> str:
        """Retorna sessão de trading atual (UTC)."""
        # Usar config para horários de sessão
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour

        # Verificar OVERLAP primeiro, pois é mais específico
        overlap_cfg = CONFIG.SESSION_CONFIG['OVERLAP']
        if overlap_cfg['start_hour'] <= current_hour < overlap_cfg['end_hour']:
            return "Overlap"

        for session_name, cfg in CONFIG.SESSION_CONFIG.items():
            if session_name == 'OVERLAP': continue # Já checado
            start = cfg['start_hour']
            end = cfg['end_hour']
            if start > end:  # Sessão overnight (ex: Ásia)
                if current_hour >= start or current_hour < end:
                    return session_name.capitalize()
            else:
                if start <= current_hour < end:
                    return session_name.capitalize()
        return "Transition" # Fora dos horários principais


    async def check_feed_connection(self) -> bool:
        """Verifica conexão do feed."""
        return self.feed_client.is_connected() if self.feed_client else False

    async def check_trade_connection(self) -> bool:
        """Verifica conexão de trading."""
        return self.trade_client.is_connected() if self.trade_client else False

    async def get_latency(self) -> float:
        """Obtém latência atual do feed em ms."""
        return await self.feed_client.get_latency() if self.feed_client else -1.0

    async def get_current_drawdown(self) -> float: # Agora retorna o DD percentual
        """Obtém drawdown atual em percentual."""
        # Este método idealmente calcularia o DD com base na equity atual vs HWM.
        # O RiskManager seria o local mais apropriado para esta lógica detalhada.
        # Por enquanto, retorna o max_drawdown_pct rastreado pelo orchestrator.
        # Para um DD "real" do dia, seria:
        # current_balance = await self.execution_engine.get_account_balance()
        # hwm_today = max(self.session_start_balance, current_balance_ao_longo_do_dia) # Precisa rastrear HWM intraday
        # dd = (hwm_today - current_balance) / hwm_today if hwm_today > 0 else 0.0
        # return dd
        return self.max_drawdown_pct # Retorna o DD máximo observado na sessão atual


    async def shutdown(self):
        """Desliga o orquestrador e seus componentes de forma graciosa."""
        if not self.running: # Evitar múltiplas chamadas de shutdown
            return
        logger.info("Desligando orquestrador...")
        self.running = False # Sinalizar para todos os loops pararem

        # Cancelar tarefas principais do loop
        for task in self._main_loop_tasks:
            if task and not task.done():
                task.cancel()
        if self._main_loop_tasks:
            await asyncio.gather(*self._main_loop_tasks, return_exceptions=True) # Esperar que as tarefas finalizem


        # Parar trading (fecha posições, cancela ordens)
        await self._stop_trading_session("Desligamento do sistema")

        # Desconectar WebSockets
        if self.feed_client:
            await self.feed_client.disconnect("Desligamento do orquestrador")
        if self.trade_client:
            await self.trade_client.disconnect("Desligamento do orquestrador")


        # Salvar estado final do DataManager (ex: flush final de ticks)
        if self.data_manager:
            await self.data_manager.save_state()

        logger.info("Orquestrador desligado com sucesso.")