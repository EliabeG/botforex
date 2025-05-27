# test_redis_isolated.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Redis import...")

try:
    # Test datetime imports
    from datetime import datetime, timedelta, timezone, date
    print("? datetime imports OK")
    
    # Test if we can create date object
    test_date = date.today()
    print(f"? date object creation OK: {test_date}")
    
    # Now try importing RedisCache
    from data.redis_cache import RedisCache
    print("? RedisCache import OK")
    
    # Try creating instance
    cache = RedisCache()
    print("? RedisCache instance creation OK")
    
except Exception as e:
    print(f"? Error: {e}")
    import traceback
    traceback.print_exc()