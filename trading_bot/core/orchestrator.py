# core/orchestrator.py
import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from config.settings import CONFIG
from core.market_regime import MarketRegimeDetector
from core.data_manager import DataManager
from core.execution_engine import ExecutionEngine
from strategies.base_strategy import BaseStrategy
from optimization.scoring import StrategyScorer
from risk.risk_manager import RiskManager
from api.ticktrader_ws import TickTraderFeed, TickTraderTrade
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
        self.feed_client = TickTraderFeed()
        self.trade_client = TickTraderTrade()
        
        # Estratégias
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: Set[str] = set()
        self.strategy_scores: Dict[str, float] = {}
        
        # Estado
        self.current_regime = None
        self.last_regime_update = datetime.now()
        self.last_score_update = datetime.now()
        self.tick_count = 0
        self.trade_count = 0
        
        # Métricas
        self.daily_pnl = 0.0
        self.session_start_balance = 0.0
        self.max_drawdown = 0.0
        
    async def initialize(self):
        """Inicializa todos os componentes"""
        logger.info("Inicializando orquestrador...")
        
        try:
            # Conectar WebSockets
            await self._connect_websockets()
            
            # Inicializar gerenciador de dados
            await self.data_manager.initialize()
            
            # Carregar histórico se necessário
            if self.mode != "live":
                await self._load_historical_data()
            
            # Inicializar motor de execução
            await self.execution_engine.initialize(self.trade_client)
            
            # Carregar e inicializar estratégias
            await self._load_strategies()
            
            # Treinar detector de regime
            await self._train_regime_detector()
            
            # Obter balanço inicial
            self.session_start_balance = await self.execution_engine.get_account_balance()
            
            logger.info("Orquestrador inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na inicialização do orquestrador: {e}")
            raise
    
    async def _connect_websockets(self):
        """Conecta aos WebSockets do TickTrader"""
        logger.info("Conectando WebSockets...")
        
        # Conectar feed de dados
        await self.feed_client.connect()
        
        # Inscrever no símbolo
        await self.feed_client.subscribe_symbol(CONFIG.SYMBOL)
        await self.feed_client.subscribe_dom(CONFIG.SYMBOL, CONFIG.DOM_LEVELS)
        
        # Conectar cliente de trading
        await self.trade_client.connect()
        await self.trade_client.authenticate()
        
        logger.info("WebSockets conectados")
    
    async def _load_strategies(self):
        """Carrega todas as estratégias disponíveis"""
        logger.info("Carregando estratégias...")
        
        # Importar dinamicamente todas as estratégias
        from strategies import load_all_strategies
        
        strategy_classes = load_all_strategies()
        
        for strategy_class in strategy_classes:
            try:
                strategy = strategy_class()
                self.strategies[strategy.name] = strategy
                await strategy.initialize()
                
                # Carregar parâmetros otimizados se existirem
                params = await self.data_manager.load_strategy_params(strategy.name)
                if params:
                    strategy.update_parameters(params)
                
            except Exception as e:
                logger.error(f"Erro ao carregar estratégia {strategy_class.__name__}: {e}")
        
        logger.info(f"{len(self.strategies)} estratégias carregadas")
    
    async def _train_regime_detector(self):
        """Treina o detector de regime de mercado"""
        logger.info("Treinando detector de regime...")
        
        # Obter dados históricos
        historical_data = await self.data_manager.get_historical_ticks(
            symbol=CONFIG.SYMBOL,
            days=30
        )
        
        if historical_data:
            await self.regime_detector.train(historical_data)
            logger.info("Detector de regime treinado")
        else:
            logger.warning("Sem dados históricos para treinar detector de regime")
    
    async def run(self):
        """Loop principal de execução"""
        self.running = True
        logger.info("Iniciando loop principal do orquestrador")
        
        # Tarefas assíncronas
        tasks = [
            asyncio.create_task(self._process_market_data()),
            asyncio.create_task(self._update_regime()),
            asyncio.create_task(self._update_scores()),
            asyncio.create_task(self._monitor_positions()),
            asyncio.create_task(self._check_risk_limits())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Erro no loop principal: {e}")
            raise
    
    async def _process_market_data(self):
        """Processa dados de mercado em tempo real"""
        while self.running:
            try:
                # Receber tick
                tick = await self.feed_client.get_tick()
                if not tick:
                    await asyncio.sleep(0.001)
                    continue
                
                self.tick_count += 1
                
                # Armazenar tick
                await self.data_manager.store_tick(tick)
                
                # Atualizar contexto de mercado
                market_context = await self._build_market_context(tick)
                
                # Processar em estratégias ativas
                for strategy_name in self.active_strategies:
                    strategy = self.strategies[strategy_name]
                    
                    # Gerar sinal
                    signal = await strategy.process_tick(market_context)
                    
                    if signal and signal.is_valid():
                        # Verificar risco
                        if await self.risk_manager.can_open_position(signal):
                            # Executar ordem
                            await self._execute_signal(strategy_name, signal)
                
            except Exception as e:
                logger.error(f"Erro ao processar dados de mercado: {e}")
                await asyncio.sleep(1)
    
    async def _update_regime(self):
        """Atualiza detecção de regime periodicamente"""
        while self.running:
            try:
                # Aguardar intervalo
                await asyncio.sleep(CONFIG.REGIME_UPDATE_MS / 1000)
                
                # Obter dados recentes
                recent_ticks = await self.data_manager.get_recent_ticks(
                    symbol=CONFIG.SYMBOL,
                    count=1000
                )
                
                if recent_ticks:
                    # Detectar regime
                    new_regime, confidence = await self.regime_detector.detect_regime(recent_ticks)
                    
                    # Atualizar se confiança suficiente
                    if confidence >= CONFIG.REGIME_CONFIDENCE_THRESHOLD:
                        if new_regime != self.current_regime:
                            logger.info(f"Mudança de regime: {self.current_regime} → {new_regime} (confiança: {confidence:.2%})")
                            self.current_regime = new_regime
                            
                            # Reselecionar estratégias
                            await self._select_active_strategies()
                    
                    self.last_regime_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Erro ao atualizar regime: {e}")
                await asyncio.sleep(5)
    
    async def _update_scores(self):
        """Atualiza scores das estratégias periodicamente"""
        while self.running:
            try:
                # Verificar condições de atualização
                time_elapsed = (datetime.now() - self.last_score_update).total_seconds() / 60
                trades_since_update = self.trade_count % CONFIG.SCORE_UPDATE_TRADES
                
                if time_elapsed >= CONFIG.SCORE_UPDATE_MINUTES or trades_since_update == 0:
                    logger.info("Atualizando scores das estratégias")
                    
                    # Calcular scores para todas as estratégias
                    for strategy_name, strategy in self.strategies.items():
                        performance = await self.data_manager.get_strategy_performance(
                            strategy_name,
                            days=30
                        )
                        
                        if performance:
                            score = self.scorer.calculate_score(performance)
                            self.strategy_scores[strategy_name] = score
                    
                    # Reselecionar estratégias ativas
                    await self._select_active_strategies()
                    
                    self.last_score_update = datetime.now()
                
                await asyncio.sleep(60)  # Verificar a cada minuto
                
            except Exception as e:
                logger.error(f"Erro ao atualizar scores: {e}")
                await asyncio.sleep(60)
    
    async def _select_active_strategies(self):
        """Seleciona as melhores estratégias para o regime atual"""
        if not self.current_regime:
            return
        
        logger.info(f"Selecionando estratégias para regime {self.current_regime}")
        
        # Filtrar estratégias adequadas ao regime
        regime_strategies = {}
        for name, strategy in self.strategies.items():
            if self.current_regime in strategy.suitable_regimes:
                score = self.strategy_scores.get(name, 0)
                if score > 0:
                    regime_strategies[name] = score
        
        # Selecionar TOP N
        sorted_strategies = sorted(
            regime_strategies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:CONFIG.MAX_ACTIVE_STRATEGIES]
        
        # Atualizar estratégias ativas
        new_active = set(name for name, _ in sorted_strategies)
        
        # Desativar estratégias removidas
        for strategy_name in self.active_strategies - new_active:
            await self.strategies[strategy_name].deactivate()
            logger.info(f"Estratégia {strategy_name} desativada")
        
        # Ativar novas estratégias
        for strategy_name in new_active - self.active_strategies:
            await self.strategies[strategy_name].activate()
            logger.info(f"Estratégia {strategy_name} ativada (score: {self.strategy_scores[strategy_name]:.4f})")
        
        self.active_strategies = new_active
    
    async def _monitor_positions(self):
        """Monitora posições abertas"""
        while self.running:
            try:
                # Obter posições abertas
                positions = await self.execution_engine.get_open_positions()
                
                for position in positions:
                    # Verificar stop loss e take profit
                    current_price = await self.data_manager.get_current_price(CONFIG.SYMBOL)
                    
                    # Atualizar trailing stop se necessário
                    if position.trailing_stop:
                        await self.execution_engine.update_trailing_stop(
                            position.id,
                            current_price
                        )
                    
                    # Verificar sinais de saída das estratégias
                    strategy_name = position.strategy_name
                    if strategy_name in self.active_strategies:
                        strategy = self.strategies[strategy_name]
                        
                        exit_signal = await strategy.check_exit_conditions(
                            position,
                            current_price
                        )
                        
                        if exit_signal:
                            await self.execution_engine.close_position(
                                position.id,
                                reason=exit_signal.reason
                            )
                
                await asyncio.sleep(1)  # Verificar a cada segundo
                
            except Exception as e:
                logger.error(f"Erro ao monitorar posições: {e}")
                await asyncio.sleep(5)
    
    async def _check_risk_limits(self):
        """Verifica limites de risco continuamente"""
        while self.running:
            try:
                # Calcular PnL diário
                current_balance = await self.execution_engine.get_account_balance()
                self.daily_pnl = (current_balance - self.session_start_balance) / self.session_start_balance
                
                # Verificar limite de perda diária
                if self.daily_pnl <= -CONFIG.DAILY_LOSS_LIMIT:
                    logger.warning(f"Limite de perda diária atingido: {self.daily_pnl:.2%}")
                    await self._stop_trading("Limite de perda diária")
                
                # Verificar meta de lucro diário
                elif self.daily_pnl >= CONFIG.TARGET_DAILY_PROFIT:
                    logger.info(f"Meta diária atingida: {self.daily_pnl:.2%}")
                    await self._stop_trading("Meta diária atingida")
                
                # Calcular drawdown
                high_water_mark = await self.data_manager.get_high_water_mark()
                current_dd = (high_water_mark - current_balance) / high_water_mark
                
                if current_dd > self.max_drawdown:
                    self.max_drawdown = current_dd
                
                # Circuit breaker
                if current_dd >= CONFIG.MAX_DRAWDOWN:
                    logger.critical(f"Circuit breaker ativado! Drawdown: {current_dd:.2%}")
                    await self._stop_trading("Circuit breaker - Drawdown máximo")
                    
                    # Pausar por 24 horas
                    await self._schedule_restart(hours=24)
                
                await asyncio.sleep(10)  # Verificar a cada 10 segundos
                
            except Exception as e:
                logger.error(f"Erro ao verificar limites de risco: {e}")
                await asyncio.sleep(30)
    
    async def _build_market_context(self, tick) -> Dict:
        """Constrói contexto de mercado para as estratégias"""
        return {
            "tick": tick,
            "regime": self.current_regime,
            "dom": await self.feed_client.get_dom_snapshot(CONFIG.SYMBOL),
            "recent_ticks": await self.data_manager.get_recent_ticks(CONFIG.SYMBOL, 100),
            "volatility": await self.data_manager.calculate_volatility(CONFIG.SYMBOL),
            "spread": tick.ask - tick.bid,
            "session": self._get_trading_session(),
            "risk_available": await self.risk_manager.get_available_risk()
        }
    
    async def _execute_signal(self, strategy_name: str, signal):
        """Executa sinal de trading"""
        try:
            # Calcular tamanho da posição
            position_size = await self.risk_manager.calculate_position_size(
                signal,
                await self.execution_engine.get_account_balance()
            )
            
            if position_size > 0:
                # Criar ordem
                order = await self.execution_engine.create_order(
                    symbol=CONFIG.SYMBOL,
                    side=signal.side,
                    size=position_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_name=strategy_name
                )
                
                if order:
                    self.trade_count += 1
                    logger.info(f"Ordem executada: {order.id} | Estratégia: {strategy_name}")
                    
                    # Registrar trade
                    await self.data_manager.record_trade(order)
                    
        except Exception as e:
            logger.error(f"Erro ao executar sinal: {e}")
    
    async def _stop_trading(self, reason: str):
        """Para operações de trading"""
        logger.info(f"Parando operações: {reason}")
        
        # Desativar todas as estratégias
        for strategy_name in self.active_strategies:
            await self.strategies[strategy_name].deactivate()
        
        self.active_strategies.clear()
        
        # Fechar todas as posições
        await self.execution_engine.close_all_positions(reason)
        
        # Cancelar ordens pendentes
        await self.execution_engine.cancel_all_orders()
    
    async def _schedule_restart(self, hours: int):
        """Agenda reinício após período"""
        restart_time = datetime.now() + timedelta(hours=hours)
        logger.info(f"Reinício agendado para {restart_time}")
        
        # Aguardar período
        await asyncio.sleep(hours * 3600)
        
        # Reinicializar
        await self.initialize()
    
    def _get_trading_session(self) -> str:
        """Retorna sessão de trading atual"""
        hour = datetime.now().hour
        
        if 7 <= hour < 16:  # Horário de Londres
            return "London"
        elif 13 <= hour < 22:  # Horário de NY
            return "NewYork"
        elif 23 <= hour or hour < 8:  # Horário da Ásia
            return "Asia"
        else:
            return "Transition"
    
    async def check_feed_connection(self) -> bool:
        """Verifica conexão do feed"""
        return self.feed_client.is_connected()
    
    async def check_trade_connection(self) -> bool:
        """Verifica conexão de trading"""
        return self.trade_client.is_connected()
    
    async def get_latency(self) -> float:
        """Obtém latência atual em ms"""
        return await self.feed_client.get_latency()
    
    async def get_current_drawdown(self) -> float:
        """Obtém drawdown atual"""
        return self.max_drawdown
    
    async def shutdown(self):
        """Desliga o orquestrador"""
        logger.info("Desligando orquestrador...")
        
        self.running = False
        
        # Parar trading
        await self._stop_trading("Desligamento")
        
        # Desconectar WebSockets
        await self.feed_client.disconnect()
        await self.trade_client.disconnect()
        
        # Salvar estado
        await self.data_manager.save_state()
        
        logger.info("Orquestrador desligado")