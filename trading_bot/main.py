# main.py
import asyncio
import signal
import sys
import logging
from datetime import datetime
import argparse

from config.settings import CONFIG # CONFIG já carrega as settings e cria os diretórios
from core.orchestrator import TradingOrchestrator
from utils.logger import setup_logger # setup_logger é importado aqui
from utils.ntp_sync import NTPSynchronizer
from monitoring.metrics_collector import MetricsCollector

# Configurar logger principal - CONFIG.LOG_LEVEL já está disponível
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
            if not await self.ntp_sync.sync(): # Adicionado 'if not' para logar falha se ocorrer
                logger.warning("Falha inicial na sincronização NTP. O bot continuará, mas o tempo pode não ser preciso.")


            # Iniciar coletor de métricas
            self.metrics.start()

            # Criar e inicializar orquestrador
            self.orchestrator = TradingOrchestrator(mode=self.mode)
            await self.orchestrator.initialize()

            logger.info("Trading Bot inicializado com sucesso")

        except Exception as e:
            logger.exception("Erro na inicialização:") # Usar logger.exception para incluir traceback
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
            logger.exception("Erro durante execução:") # Usar logger.exception
            # Não relançar aqui necessariamente, a menos que queira parar tudo do main.
            # O finally no main global cuidará da parada.
            # Se relançar, o bot pode parar abruptamente sem o cleanup do finally global.
            # No entanto, se o erro for crítico no start, talvez seja melhor parar.
            # Por ora, vamos manter o raise para que o erro seja visível no nível superior.
            raise


    async def monitor_health(self):
        """Monitora saúde do sistema"""
        while self.running:
            try:
                if self.orchestrator: # Adicionado para evitar erro se orchestrator não inicializou
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
                    if latency > CONFIG.ALERT_LATENCY_MS: # ALERT_LATENCY_MS é int
                        logger.warning(f"Latência alta detectada: {latency:.2f}ms") # Adicionado .2f para formatação

                    # Verificar drawdown
                    drawdown = await self.orchestrator.get_current_drawdown()
                    if drawdown > CONFIG.ALERT_DD_WEEKLY:
                        logger.warning(f"Drawdown semanal alto: {drawdown:.2%}")
                else:
                    logger.warning("Orquestrador não está disponível para monitoramento de saúde.")


                await asyncio.sleep(30)  # Verificar a cada 30 segundos

            except Exception as e:
                logger.error(f"Erro no monitoramento de saúde: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        """Para o bot de forma segura"""
        if not self.running: # Evitar múltiplas chamadas
            return
        logger.info("Parando Trading Bot...")
        self.running = False

        if self.orchestrator:
            await self.orchestrator.shutdown()

        # Parar ntp_sync se ele tiver uma tarefa de background que precisa ser cancelada
        # (O código original não mostra um método stop explícito para ntp_sync, mas a task dele será cancelada pelo gather)

        self.metrics.stop()
        logger.info("Trading Bot parado com sucesso")

    def handle_signal(self, signum, frame): # Adicionado frame conforme assinatura padrão
        """Manipula sinais do sistema"""
        # Para evitar chamar o logger depois que ele pode ter sido desligado durante o shutdown:
        print(f"Recebido sinal {signum}, iniciando parada...") # Usar print para emergência
        if self.running: # Apenas criar task se estiver rodando para evitar erro
             asyncio.create_task(self.stop())


def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="Bot de Trading Automatizado EURUSD")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest"],
        default="live", # Ou o valor de CONFIG.TRADING_MODE
        help="Modo de operação do bot"
    )
    parser.add_argument(
        "--config", # Este argumento não parece estar sendo usado para carregar um arquivo JSON de config
        type=str,
        help="Caminho para arquivo de configuração customizado (não implementado no settings.py atual)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativar modo debug"
    )
    return parser.parse_args()

async def main_async_logic(): # Renomeado para clareza
    """Função principal com lógica assíncrona"""
    args = parse_arguments()

    if args.debug:
        # Certifique-se que o logger raiz e outros loggers importantes são ajustados
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG) # Ajusta o logger específico do main.py
        logger.info("Modo DEBUG ativado.")


    # Criar instância do bot
    # O modo pode vir de args ou da CONFIG, priorizar args se fornecido.
    effective_mode = args.mode if args.mode else CONFIG.TRADING_MODE
    bot = TradingBot(mode=effective_mode)


    # Configurar manipuladores de sinal
    # loop = asyncio.get_event_loop() # Não é mais recomendado pegar o loop assim diretamente
    # for sig in (signal.SIGTERM, signal.SIGINT):
    #     loop.add_signal_handler(sig, lambda s=sig: bot.handle_signal(s, None))

    # Uma forma mais robusta de lidar com signals em asyncio moderno:
    # signal.signal(s, lambda s, f: bot.handle_signal(s)) ainda pode ter problemas com asyncio
    # A melhor abordagem é deixar o KeyboardInterrupt ser pego e usar SIGHUP/SIGTERM
    # para tarefas de cleanup mais complexas se o loop estiver ocupado.
    # Para simplicidade, o método original de signal.signal pode ser mantido,
    # mas asyncio.create_task(bot.stop()) é a forma correta de chamar uma corrotina de um handler.

    # Tentar capturar sinais para shutdown gracioso
    # Esta é uma forma mais compatível com asyncio
    shutdown_signals = (signal.SIGTERM, signal.SIGINT)
    loop = asyncio.get_running_loop()

    for s in shutdown_signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(bot.stop())
        )


    try:
        # Inicializar
        await bot.initialize()

        # Executar
        await bot.start()

    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário (KeyboardInterrupt)")
    except asyncio.CancelledError:
        logger.info("Tarefa principal cancelada.")
    except Exception as e:
        logger.exception("Erro fatal no main_async_logic:") # Usar logger.exception
    finally:
        logger.info("Iniciando processo de parada final do bot...")
        await bot.stop() # Garantir que stop seja chamado


if __name__ == "__main__":
    # Configurar event loop com política adequada
    if sys.platform == "win32" and sys.version_info >= (3, 8): # Adicionado check de versão Python
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Executar bot
    asyncio.run(main_async_logic())