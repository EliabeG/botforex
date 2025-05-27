# test_components.py
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_components():
    print("?? Testing Trading Bot Components...\n")
    
    # Test 1: Config
    print("1?? Testing Configuration...")
    try:
        from config.settings import CONFIG
        assert CONFIG.SYMBOL == "EURUSD"
        assert CONFIG.LEVERAGE > 0
        print(f"   ? Config OK - Symbol: {CONFIG.SYMBOL}, Leverage: {CONFIG.LEVERAGE}")
    except Exception as e:
        print(f"   ? Config Error: {e}")
        return
    
    # Test 2: Logger
    print("\n2?? Testing Logger...")
    try:
        from utils.logger import setup_logger
        logger = setup_logger("test")
        logger.info("Test message")
        print("   ? Logger OK")
    except Exception as e:
        print(f"   ? Logger Error: {e}")
    
    # Test 3: Redis Cache - ISOLATED
    print("\n3?? Testing Redis Cache...")
    try:
        # First test the imports directly
        from datetime import datetime, timedelta, timezone, date
        test_date = date.today()
        print(f"   ? datetime.date import OK: {test_date}")
        
        # Now import RedisCache
        from data.redis_cache import RedisCache
        cache = RedisCache()
        print("   ? Redis Cache initialized")
    except Exception as e:
        print(f"   ? Redis Cache Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Risk Manager
    print("\n4?? Testing Risk Manager...")
    try:
        from risk.risk_manager import RiskManager
        risk_mgr = RiskManager()
        print("   ? Risk Manager initialized")
    except Exception as e:
        print(f"   ? Risk Manager Error: {e}")
    
    # Test 5: Base Strategy
    print("\n5?? Testing Base Strategy...")
    try:
        from strategies.base_strategy import BaseStrategy, Signal
        from datetime import datetime, timezone
        signal = Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name="TestStrategy",
            symbol="EURUSD",
            side="buy",
            confidence=0.8,
            stop_loss=1.0950,
            take_profit=1.1100
        )
        assert signal.is_valid()
        print("   ? Base Strategy OK")
    except Exception as e:
        print(f"   ? Base Strategy Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: NTP Sync
    print("\n6?? Testing NTP Sync...")
    try:
        from utils.ntp_sync import NTPSynchronizer
        ntp = NTPSynchronizer()
        print("   ? NTP Synchronizer initialized")
    except Exception as e:
        print(f"   ? NTP Sync Error: {e}")
    
    print("\n? Component tests completed!")

if __name__ == "__main__":
    asyncio.run(test_components())