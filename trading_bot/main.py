# main.py
import asyncio
import signal
import sys
import logging
from datetime import datetime
import argparse

from config.settings import CONFIG
from core.orchestrator import TradingOrchestrator
from utils.logger import setup_logger
from utils.ntp_sync import NTPSynchronizer
from monitoring.metrics_collector import MetricsCollector

# Configurar logger principal
logger = setup_logger("main", CONFIG.LOG_LEVEL)

class TradingBot:
    """Bot de trading principal"""
    
    def __init__(self, mode: str = "live"):
        self.mode = mode
        self.orchestrator = None
        self.ntp_sync = NTPSynchronizer()
        self.metrics = MetricsCollector(port=CONFIG.METRICS_PORT)
        self.running = False
        
    async def initialize(self):
        """Inicializa todos os componentes do bot"""
        try:
            logger.info(f"Inicializando Trading Bot em modo {self.mode}")
            logger.info(f"Conta: {CONFIG.LOGIN} | Servidor: {CONFIG.SERVER}")
            
            # Sincronizar relógio
            await self.ntp_sync.sync()
            
            # Iniciar coletor de métricas
            self.metrics.start()
            
            # Criar e inicializar orquestrador
            self.orchestrator = TradingOrchestrator(mode=self.mode)
            await self.orchestrator.initialize()
            
            logger.info("Trading Bot inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na inicialização: {e}")
            raise
    
    async def start(self):
        """Inicia operação do bot"""
        try:
            self.running = True
            logger.info("Iniciando operações de trading")
            
            # Tarefas assíncronas principais
            tasks = [
                asyncio.create_task(self.orchestrator.run()),
                asyncio.create_task(self.ntp_sync.periodic_sync(CONFIG.NTP_SYNC_MINUTES * 60)),
                asyncio.create_task(self.monitor_health())
            ]
            
            # Aguardar até parada
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Erro durante execução: {e}")
            raise
    
    async def monitor_health(self):
        """Monitora saúde do sistema"""
        while self.running:
            try:
                # Verificar conexões
                ws_feed_status = await self.orchestrator.check_feed_connection()
                ws_trade_status = await self.orchestrator.check_trade_connection()
                
                # Atualizar métricas
                self.metrics.update_connection_status(
                    feed=ws_feed_status,
                    trade=ws_trade_status
                )
                
                # Verificar latência
                latency = await self.orchestrator.get_latency()
                if latency > CONFIG.ALERT_LATENCY_MS:
                    logger.warning(f"Latência alta detectada: {latency}ms")
                
                # Verificar drawdown
                drawdown = await self.orchestrator.get_current_drawdown()
                if drawdown > CONFIG.ALERT_DD_WEEKLY:
                    logger.warning(f"Drawdown semanal alto: {drawdown:.2%}")
                
                await asyncio.sleep(30)  # Verificar a cada 30 segundos
                
            except Exception as e:
                logger.error(f"Erro no monitoramento de saúde: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Para o bot de forma segura"""
        logger.info("Parando Trading Bot...")
        self.running = False
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        self.metrics.stop()
        logger.info("Trading Bot parado com sucesso")
    
    def handle_signal(self, signame):
        """Manipula sinais do sistema"""
        logger.info(f"Recebido sinal {signame}")
        asyncio.create_task(self.stop())

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="Bot de Trading Automatizado EURUSD")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest"],
        default="live",
        help="Modo de operação do bot"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Caminho para arquivo de configuração customizado"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativar modo debug"
    )
    return parser.parse_args()

async def main():
    """Função principal"""
    args = parse_arguments()
    
    # Ajustar nível de log se debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Criar instância do bot
    bot = TradingBot(mode=args.mode)
    
    # Configurar manipuladores de sinal
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: bot.handle_signal(s))
    
    try:
        # Inicializar
        await bot.initialize()
        
        # Executar
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
    finally:
        await bot.stop()

if __name__ == "__main__":
    # Configurar event loop com política adequada
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Executar bot
    asyncio.run(main())