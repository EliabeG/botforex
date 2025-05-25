# data/redis_cache.py
"""Sistema de cache Redis para dados de mercado"""
import aioredis
import json
import pickle
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import numpy as np

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("redis_cache")

class RedisCache:
    """Gerenciador de cache Redis para dados de alta frequência"""
    
    def __init__(self):
        self.redis = None
        self.connected = False
        
        # Configurações de TTL (segundos)
        self.ttl_config = {
            'tick': CONFIG.REDIS_TTL_HOURS * 3600,
            'ohlc': 86400,  # 24 horas
            'indicator': 300,  # 5 minutos
            'market_state': 60,  # 1 minuto
            'position': 30,  # 30 segundos
            'performance': 600,  # 10 minutos
        }
        
    async def connect(self):
        """Conecta ao Redis"""
        try:
            self.redis = await aioredis.create_redis_pool(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                encoding='utf-8',
                minsize=5,
                maxsize=10
            )
            self.connected = True
            logger.info("Conectado ao Redis")
            
            # Verificar conexão
            await self.redis.ping()
            
        except Exception as e:
            logger.error(f"Erro ao conectar ao Redis: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Desconecta do Redis"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            self.connected = False
            logger.info("Desconectado do Redis")
    
    # === Métodos para Ticks ===
    
    async def store_tick(self, symbol: str, tick: Dict):
        """Armazena tick no cache"""
        if not self.connected:
            return
        
        try:
            # Chave para último tick
            key_latest = f"tick:{symbol}:latest"
            await self.redis.setex(
                key_latest,
                self.ttl_config['tick'],
                json.dumps(tick, default=str)
            )
            
            # Adicionar à lista de ticks recentes
            key_list = f"tick:{symbol}:list"
            await self.redis.lpush(key_list, json.dumps(tick, default=str))
            await self.redis.ltrim(key_list, 0, 10000)  # Manter últimos 10k
            await self.redis.expire(key_list, self.ttl_config['tick'])
            
            # Atualizar timestamp do símbolo
            await self.redis.zadd(
                "symbols:active",
                int(datetime.now().timestamp()),
                symbol
            )
            
        except Exception as e:
            logger.error(f"Erro ao armazenar tick: {e}")
    
    async def get_latest_tick(self, symbol: str) -> Optional[Dict]:
        """Obtém último tick de um símbolo"""
        if not self.connected:
            return None
        
        try:
            key = f"tick:{symbol}:latest"
            data = await self.redis.get(key)
            
            if data:
                return json.loads(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter tick: {e}")
        
        return None
    
    async def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """Obtém ticks recentes"""
        if not self.connected:
            return []
        
        try:
            key = f"tick:{symbol}:list"
            ticks_json = await self.redis.lrange(key, 0, count - 1)
            
            ticks = []
            for tick_str in ticks_json:
                try:
                    tick = json.loads(tick_str)
                    ticks.append(tick)
                except:
                    pass
            
            return list(reversed(ticks))  # Ordem cronológica
            
        except Exception as e:
            logger.error(f"Erro ao obter ticks recentes: {e}")
            return []
    
    # === Métodos para Indicadores ===
    
    async def store_indicator(self, symbol: str, indicator_name: str, 
                            value: Union[float, Dict], timeframe: str = "current"):
        """Armazena valor de indicador"""
        if not self.connected:
            return
        
        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            
            if isinstance(value, dict):
                data = json.dumps(value)
            else:
                data = str(value)
            
            await self.redis.setex(
                key,
                self.ttl_config['indicator'],
                data
            )
            
        except Exception as e:
            logger.error(f"Erro ao armazenar indicador: {e}")
    
    async def get_indicator(self, symbol: str, indicator_name: str,
                          timeframe: str = "current") -> Optional[Union[float, Dict]]:
        """Obtém valor de indicador"""
        if not self.connected:
            return None
        
        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            data = await self.redis.get(key)
            
            if data:
                try:
                    return json.loads(data)
                except:
                    return float(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter indicador: {e}")
        
        return None
    
    # === Métodos para Estado do Mercado ===
    
    async def store_market_state(self, state: Dict):
        """Armazena estado atual do mercado"""
        if not self.connected:
            return
        
        try:
            key = "market:state:current"
            await self.redis.setex(
                key,
                self.ttl_config['market_state'],
                json.dumps(state, default=str)
            )
            
            # Histórico de estados
            history_key = "market:state:history"
            await self.redis.lpush(history_key, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'state': state
            }, default=str))
            await self.redis.ltrim(history_key, 0, 1000)
            
        except Exception as e:
            logger.error(f"Erro ao armazenar estado do mercado: {e}")
    
    async def get_market_state(self) -> Optional[Dict]:
        """Obtém estado atual do mercado"""
        if not self.connected:
            return None
        
        try:
            key = "market:state:current"
            data = await self.redis.get(key)
            
            if data:
                return json.loads(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter estado do mercado: {e}")
        
        return None
    
    # === Métodos para Performance ===
    
    async def store_strategy_performance(self, strategy_name: str, metrics: Dict):
        """Armazena métricas de performance de estratégia"""
        if not self.connected:
            return
        
        try:
            key = f"performance:strategy:{strategy_name}"
            await self.redis.setex(
                key,
                self.ttl_config['performance'],
                json.dumps(metrics, default=str)
            )
            
            # Ranking de estratégias por score
            if 'score' in metrics:
                await self.redis.zadd(
                    "strategies:ranking",
                    metrics['score'],
                    strategy_name
                )
            
        except Exception as e:
            logger.error(f"Erro ao armazenar performance: {e}")
    
    async def get_strategy_performance(self, strategy_name: str) -> Optional[Dict]:
        """Obtém métricas de performance de estratégia"""
        if not self.connected:
            return None
        
        try:
            key = f"performance:strategy:{strategy_name}"
            data = await self.redis.get(key)
            
            if data:
                return json.loads(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter performance: {e}")
        
        return None
    
    async def get_top_strategies(self, count: int = 10) -> List[Tuple[str, float]]:
        """Obtém ranking das melhores estratégias"""
        if not self.connected:
            return []
        
        try:
            # Obter top N com scores
            result = await self.redis.zrevrange(
                "strategies:ranking",
                0,
                count - 1,
                withscores=True
            )
            
            return [(name, score) for name, score in result]
            
        except Exception as e:
            logger.error(f"Erro ao obter ranking: {e}")
            return []
    
    # === Métodos para Posições ===
    
    async def store_position(self, position_id: str, position_data: Dict):
        """Armazena dados de posição"""
        if not self.connected:
            return
        
        try:
            key = f"position:{position_id}"
            await self.redis.setex(
                key,
                self.ttl_config['position'],
                json.dumps(position_data, default=str)
            )
            
            # Adicionar ao conjunto de posições ativas
            await self.redis.sadd("positions:active", position_id)
            
        except Exception as e:
            logger.error(f"Erro ao armazenar posição: {e}")
    
    async def get_position(self, position_id: str) -> Optional[Dict]:
        """Obtém dados de posição"""
        if not self.connected:
            return None
        
        try:
            key = f"position:{position_id}"
            data = await self.redis.get(key)
            
            if data:
                return json.loads(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter posição: {e}")
        
        return None
    
    async def remove_position(self, position_id: str):
        """Remove posição do cache"""
        if not self.connected:
            return
        
        try:
            key = f"position:{position_id}"
            await self.redis.delete(key)
            await self.redis.srem("positions:active", position_id)
            
        except Exception as e:
            logger.error(f"Erro ao remover posição: {e}")
    
    # === Métodos para Arrays/Séries ===
    
    async def store_array(self, key: str, array: np.ndarray, ttl: Optional[int] = None):
        """Armazena array numpy"""
        if not self.connected:
            return
        
        try:
            # Serializar com pickle
            data = pickle.dumps(array)
            ttl = ttl or self.ttl_config['indicator']
            
            await self.redis.setex(f"array:{key}", ttl, data)
            
        except Exception as e:
            logger.error(f"Erro ao armazenar array: {e}")
    
    async def get_array(self, key: str) -> Optional[np.ndarray]:
        """Obtém array numpy"""
        if not self.connected:
            return None
        
        try:
            data = await self.redis.get(f"array:{key}")
            
            if data:
                return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Erro ao obter array: {e}")
        
        return None
    
    # === Métodos Utilitários ===
    
    async def exists(self, key: str) -> bool:
        """Verifica se chave existe"""
        if not self.connected:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except:
            return False
    
    async def expire_in(self, key: str, seconds: int):
        """Define expiração para chave"""
        if not self.connected:
            return
        
        try:
            await self.redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Erro ao definir expiração: {e}")
    
    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Busca chaves por padrão"""
        if not self.connected:
            return []
        
        try:
            return await self.redis.keys(pattern)
        except Exception as e:
            logger.error(f"Erro ao buscar chaves: {e}")
            return []
    
    async def flush_pattern(self, pattern: str):
        """Remove todas as chaves que correspondem ao padrão"""
        if not self.connected:
            return
        
        try:
            keys = await self.get_keys_pattern(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Removidas {len(keys)} chaves com padrão '{pattern}'")
                
        except Exception as e:
            logger.error(f"Erro ao limpar chaves: {e}")
    
    async def get_info(self) -> Dict:
        """Obtém informações do Redis"""
        if not self.connected:
            return {}
        
        try:
            info = await self.redis.info()
            
            return {
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1))
                    if info.get('keyspace_hits', 0) > 0 else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter info do Redis: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Verifica saúde do Redis"""
        if not self.connected:
            return False
        
        try:
            await self.redis.ping()
            return True
        except:
            return False