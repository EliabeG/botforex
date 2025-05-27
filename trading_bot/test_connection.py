import asyncio
import aiohttp
from config.settings import CONFIG

async def test_api_connection():
    print(f"🔌 Testing connection to {CONFIG.REST_API_URL}...\n")
    
    timeout = aiohttp.ClientTimeout(total=10)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test REST API
            async with session.get(f"{CONFIG.REST_API_URL}/TickTrader") as response:
                print(f"REST API Status: {response.status}")
                if response.status == 200:
                    print("✅ REST API is accessible")
                else:
                    print(f"⚠️  REST API returned status {response.status}")
    
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Connection Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_connection())
