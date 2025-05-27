import asyncio
import sys
import os
from datetime import datetime, timezone # Adicionado para criar Signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar um logger basico para o proprio script de teste
import logging
test_script_logger = logging.getLogger("component_tester")
test_script_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
test_script_logger.addHandler(handler)


async def test_components():
    test_script_logger.info("?? Testing Trading Bot Components...\n")
    test_passed_count = 0
    test_failed_count = 0

    # --- Test 1: Config ---
    test_script_logger.info("1?? Testing Configuration...")
    try:
        from config.settings import CONFIG
        assert CONFIG.SYMBOL == "EURUSD", f"CONFIG.SYMBOL e {CONFIG.SYMBOL}, esperado EURUSD"
        assert CONFIG.LEVERAGE > 0, f"CONFIG.LEVERAGE e {CONFIG.LEVERAGE}, esperado > 0"
        test_script_logger.info(f"    ? Config OK - Symbol: {CONFIG.SYMBOL}, Leverage: {CONFIG.LEVERAGE}")
        test_passed_count += 1
    except Exception as e:
        test_script_logger.error(f"    ? Config Error: {e}", exc_info=True)
        test_failed_count += 1
        # Nao retornar, tentar outros testes

    # --- Test 2: Logger ---
    test_script_logger.info("\n2?? Testing Logger...")
    try:
        from utils.logger import setup_logger
        logger_module_test = setup_logger("test_module_logger") # Nome diferente para evitar conflito
        logger_module_test.info("Test message from module logger")
        test_script_logger.info("    ? Logger OK")
        test_passed_count += 1
    except Exception as e:
        test_script_logger.error(f"    ? Logger Error: {e}", exc_info=True)
        test_failed_count += 1

    # --- Test 3: Redis Cache ---
    test_script_logger.info("\n3?? Testing Redis Cache...")
    redis_cache_instance = None # Para o finally
    try:
        from data.redis_cache import RedisCache # Movido import para dentro do try
        redis_cache_instance = RedisCache()
        test_script_logger.info("    ? Attempting Redis connection...")
        await redis_cache_instance.connect()
        assert redis_cache_instance.connected, "Falha ao conectar ao Redis"
        
        # Testar store e get
        test_key = "test_redis_component:mykey"
        test_value = {"data": "test_data_value", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        # Limpar chave antes do teste para garantir
        if redis_cache_instance.redis: # Checar se self.redis nao e None
            await redis_cache_instance.redis.delete(test_key)

        await redis_cache_instance.store_indicator(CONFIG.SYMBOL, "test_indicator", test_value)
        retrieved_value = await redis_cache_instance.get_indicator(CONFIG.SYMBOL, "test_indicator")
        
        assert retrieved_value is not None, "Valor nao recuperado do Redis"
        assert retrieved_value.get("data") == test_value["data"], "Valor recuperado do Redis nao confere"
        test_script_logger.info(f"    Redis store/get test value: {retrieved_value}")

        test_script_logger.info("    ? Redis Cache OK (Connection, Store, Get)")
        test_passed_count += 1
    except NameError as ne: # Capturar NameError especificamente
        test_script_logger.error(f"    ? Redis Cache NameError: {ne} - Verifique a importacao de 'date' em redis_cache.py e reconstrua a imagem Docker com --no-cache.", exc_info=True)
        test_failed_count += 1
    except Exception as e:
        test_script_logger.error(f"    ? Redis Cache Error: {e}", exc_info=True)
        test_failed_count += 1
    finally:
        if redis_cache_instance and redis_cache_instance.connected:
            await redis_cache_instance.disconnect()
            test_script_logger.info("    Redis connection closed for test.")


    # --- Test 4: Risk Manager ---
    test_script_logger.info("\n4?? Testing Risk Manager...")
    try:
        from risk.risk_manager import RiskManager
        risk_manager = RiskManager()
        await risk_manager.initialize(starting_balance=10000)
        test_script_logger.info(f"    Risk Manager daily_start_balance: {risk_manager.daily_start_balance_rm}")
        assert risk_manager.daily_start_balance_rm == 10000
        test_script_logger.info("    ? Risk Manager initialized OK")
        test_passed_count += 1
    except Exception as e:
        test_script_logger.error(f"    ? Risk Manager Error: {e}", exc_info=True)
        test_failed_count += 1

    # --- Test 5: Base Strategy & Signal/Position ---
    test_script_logger.info("\n5?? Testing Base Strategy...")
    try:
        from strategies.base_strategy import BaseStrategy, Signal, Position
        
        # Testar instanciacao de Signal com argumentos obrigatorios
        current_time = datetime.now(timezone.utc)
        test_signal = Signal(
            strategy_name="TestStrategy", 
            symbol="EURUSD", # Argumento obrigatorio
            side="buy", 
            confidence=0.75,
            timestamp=current_time # Fornecer timestamp
        )
        assert test_signal.symbol == "EURUSD"
        assert test_signal.strategy_name == "TestStrategy"
        assert test_signal.is_valid()
        test_script_logger.info(f"    Signal instance: {test_signal}")

        # Testar instanciacao de Position
        test_position = Position(
            id="test_pos_123",
            strategy_name="TestStrategy",
            symbol="EURUSD",
            side="buy",
            entry_price=1.1000,
            size=0.1,
            open_time=current_time
        )
        assert test_position.size == 0.1
        test_script_logger.info(f"    Position instance: {test_position}")
        
        # Criar uma subclasse dummy de BaseStrategy para teste
        class DummyStrategy(BaseStrategy):
            def get_default_parameters(self): return {}
            async def calculate_indicators(self, market_context): self.current_indicators = {"test_indicator": 1}
            async def generate_signal(self, market_context): return None
            async def evaluate_exit_conditions(self, open_position, market_context): return None

        dummy_strategy = DummyStrategy(name="DummyTest")
        await dummy_strategy.initialize_strategy()
        assert dummy_strategy.name == "DummyTest"
        test_script_logger.info("    ? Base Strategy (Signal, Position, Dummy Strategy init) OK")
        test_passed_count += 1
    except Exception as e:
        test_script_logger.error(f"    ? Base Strategy Error: {e}", exc_info=True)
        test_failed_count += 1

    # --- Test 6: NTP Sync ---
    test_script_logger.info("\n6?? Testing NTP Sync...")
    ntp_sync_instance = None
    try:
        from utils.ntp_sync import NTPSynchronizer
        ntp_sync_instance = NTPSynchronizer()
        # O teste de conexao real pode ser demorado, testar apenas inicializacao
        assert ntp_sync_instance.ntp_client is not None
        test_script_logger.info("    ? NTP Synchronizer initialized OK")
        test_passed_count += 1
    except Exception as e:
        test_script_logger.error(f"    ? NTP Sync Error: {e}", exc_info=True)
        test_failed_count += 1
    
    # Adicionar mais testes para outros componentes (DataManager, ExecutionEngine, etc.) aqui

    test_script_logger.info("\n? Component tests completed!")
    test_script_logger.info(f"PASSED: {test_passed_count}, FAILED: {test_failed_count}")

if __name__ == "__main__":
    # Garantir que o script e executado no diretorio /app se estiver dentro do Docker
    if os.getcwd() != '/app' and os.path.exists('/app'):
        os.chdir('/app')
        test_script_logger.info(f"Changed CWD to /app for test execution.")
    
    test_script_logger.info(f"Current Working Directory: {os.getcwd()}")
    test_script_logger.info(f"Python sys.path: {sys.path}")

    asyncio.run(test_components())