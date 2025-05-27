# data/redis_cache.py
"""Sistema de cache Redis para dados de mercado"""
import aioredis # type: ignore 
import json
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple 
from datetime import datetime, timedelta, timezone, date # <-- ESTA LINHA E CRUCIAL
import asyncio
import numpy as np

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("redis_cache")

class RedisCache:
    """Gerenciador de cache Redis para dados de alta frequencia"""

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None 
        self.connected: bool = False

        self.ttl_config: Dict[str, int] = {
            'tick': CONFIG.REDIS_TTL_HOURS * 3600, 
            'ohlc': 86400,  
            'indicator': 300,  
            'market_state': 60,  
            'position': 3600,  
            'performance': 600,  
            'array': 3600, 
        }
    # ... (o restante do arquivo como fornecido anteriormente) ...
    # O importante e garantir que a linha de import de 'date' esteja correta.
    # Cole o restante do arquivo que eu forneci na resposta anterior aqui.

    async def connect(self):
        if self.connected and self.redis:
            try: 
                await self.redis.ping()
                logger.info("Ja conectado ao Redis.")
                return
            except (aioredis.exceptions.ConnectionError, ConnectionRefusedError, asyncio.TimeoutError):
                logger.warning("Conexao Redis existente falhou. Tentando reconectar.")
                self.connected = False 
                if self.redis:
                    try:
                        await self.redis.close()
                    except Exception:
                        pass 
                    self.redis = None
        try:
            self.redis = await aioredis.from_url(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                encoding='utf-8', 
                health_check_interval=30 
            )
            await self.redis.ping() 
            self.connected = True
            logger.info(f"Conectado ao Redis em {CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}")
        except Exception as e:
            logger.exception("Erro ao conectar ao Redis:") 
            self.connected = False
            self.redis = None 

    async def disconnect(self):
        if self.redis:
            try:
                await self.redis.close()
                logger.info("Conexao com Redis fechada.")
            except Exception as e:
                logger.error(f"Erro ao fechar conexao Redis: {e}")
            finally:
                self.redis = None
                self.connected = False
        else:
            logger.info("Nenhuma conexao Redis para fechar.")
        self.connected = False

    async def store_tick(self, symbol: str, tick: Dict[str, Any]): 
        if not self.is_connected_guard(): return 
        if not self.redis: return
        try:
            tick_to_store = tick.copy() 
            if isinstance(tick_to_store.get('timestamp'), datetime):
                tick_to_store['timestamp'] = tick_to_store['timestamp'].isoformat()
            tick_json = json.dumps(tick_to_store)
            key_latest = f"tick:{symbol}:latest"
            await self.redis.setex(key_latest, self.ttl_config['tick'], tick_json)
            key_list = f"tick:{symbol}:list"
            async with self.redis.pipeline(transaction=False) as pipe: 
                await pipe.lpush(key_list, tick_json)
                await pipe.ltrim(key_list, 0, 9999)  
                await pipe.expire(key_list, self.ttl_config['tick'])
                await pipe.execute()
        except Exception as e:
            logger.exception(f"Erro ao armazenar tick para {symbol} no Redis:")

    async def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]: 
        if not self.is_connected_guard(): return None
        if not self.redis: return None
        try:
            key = f"tick:{symbol}:latest"
            data_bytes = await self.redis.get(key) 
            if data_bytes: 
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                tick_dict = json.loads(data_str)
                if 'timestamp' in tick_dict and isinstance(tick_dict['timestamp'], str):
                    try:
                        tick_dict['timestamp'] = datetime.fromisoformat(tick_dict['timestamp'].replace('Z', '+00:00'))
                    except ValueError: 
                        tick_dict['timestamp'] = datetime.now(timezone.utc) 
                return tick_dict
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter ultimo tick para {symbol} do Redis:")
            return None

    async def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]: 
        if not self.is_connected_guard(): return []
        if not self.redis: return []
        try:
            key = f"tick:{symbol}:list"
            ticks_json_list = await self.redis.lrange(key, 0, count - 1) 
            ticks: List[Dict[str, Any]] = []
            for tick_entry in ticks_json_list: 
                if tick_entry:
                    try:
                        tick_str = tick_entry.decode('utf-8') if isinstance(tick_entry, bytes) else tick_entry
                        tick_dict = json.loads(tick_str)
                        if 'timestamp' in tick_dict and isinstance(tick_dict['timestamp'], str):
                             tick_dict['timestamp'] = datetime.fromisoformat(tick_dict['timestamp'].replace('Z', '+00:00'))
                        ticks.append(tick_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Erro ao decodificar JSON de tick recente do Redis: {tick_entry}")
                    except Exception as e_parse:
                        logger.warning(f"Erro ao parsear tick recente do Redis: {e_parse}, dado: {tick_entry}")
            return list(reversed(ticks))  
        except Exception as e:
            logger.exception(f"Erro ao obter ticks recentes para {symbol} do Redis:")
            return []

    async def store_indicator(self, symbol: str, indicator_name: str,
                            value: Union[float, int, str, Dict[str, Any], List[Any]], 
                            timeframe: str = "current"):
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            data_to_store: str
            if isinstance(value, (dict, list)): data_to_store = json.dumps(value)
            elif isinstance(value, (float, int)): data_to_store = str(value)
            elif isinstance(value, str): data_to_store = value
            else:
                logger.warning(f"Tipo de valor nao suportado para indicador '{indicator_name}': {type(value)}")
                return
            await self.redis.setex(key, self.ttl_config['indicator'], data_to_store)
        except Exception as e:
            logger.exception(f"Erro ao armazenar indicador {indicator_name} para {symbol} no Redis:")

    async def get_indicator(self, symbol: str, indicator_name: str,
                          timeframe: str = "current") -> Optional[Any]: 
        if not self.is_connected_guard(): return None
        if not self.redis: return None
        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            data_bytes = await self.redis.get(key)
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                try: return json.loads(data_str)
                except json.JSONDecodeError:
                    try: return float(data_str)
                    except ValueError:
                        try: return int(data_str)
                        except ValueError: return data_str 
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter indicador {indicator_name} para {symbol} do Redis:")
            return None

    async def store_market_state(self, state: Dict[str, Any]): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            state_to_store = state.copy()
            state_to_store['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
            state_json = json.dumps(state_to_store)
            key_current = "market:state:current"
            await self.redis.setex(key_current, self.ttl_config['market_state'], state_json)
            history_key = "market:state:history"
            async with self.redis.pipeline(transaction=False) as pipe: 
                await pipe.lpush(history_key, state_json)
                await pipe.ltrim(history_key, 0, 999) 
                await pipe.expire(history_key, self.ttl_config['market_state'] * 24) 
                await pipe.execute()
        except Exception as e:
            logger.exception("Erro ao armazenar estado do mercado no Redis:")

    async def get_market_state(self) -> Optional[Dict[str, Any]]: 
        if not self.is_connected_guard(): return None
        if not self.redis: return None
        try:
            key = "market:state:current"
            data_bytes = await self.redis.get(key)
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                return json.loads(data_str)
            return None
        except Exception as e:
            logger.exception("Erro ao obter estado do mercado do Redis:")
            return None

    async def store_strategy_performance(self, strategy_name: str, metrics: Dict[str, Any]): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            key = f"performance:strategy:{strategy_name}"
            metrics_json = json.dumps(metrics) 
            await self.redis.setex(key, self.ttl_config['performance'], metrics_json)
            if 'score' in metrics and isinstance(metrics['score'], (int, float)):
                await self.redis.zadd("strategies:ranking:score", {strategy_name: float(metrics['score'])})
            if 'sharpe_ratio' in metrics and isinstance(metrics['sharpe_ratio'], (int, float)):
                 await self.redis.zadd("strategies:ranking:sharpe", {strategy_name: float(metrics['sharpe_ratio'])})
        except Exception as e:
            logger.exception(f"Erro ao armazenar performance de {strategy_name} no Redis:")

    async def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]: 
        if not self.is_connected_guard(): return None
        if not self.redis: return None
        try:
            key = f"performance:strategy:{strategy_name}"
            data_bytes = await self.redis.get(key)
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                return json.loads(data_str)
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter performance de {strategy_name} do Redis:")
            return None

    async def get_top_strategies(self, count: int = 10, by_metric: str = 'score') -> List[Tuple[str, float]]:
        if not self.is_connected_guard(): return []
        if not self.redis: return []
        ranking_key = f"strategies:ranking:{by_metric.lower()}"
        try:
            result_tuples = await self.redis.zrevrange(ranking_key, 0, count - 1, withscores=True)
            parsed_results: List[Tuple[str, float]] = []
            for item in result_tuples:
                name_bytes, score_val = item 
                name_str = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)
                parsed_results.append((name_str, float(score_val)))
            return parsed_results
        except Exception as e:
            logger.exception(f"Erro ao obter ranking de estrategias (metrica: {by_metric}) do Redis:")
            return []

    async def store_position_data(self, position_id: str, position_data: Dict[str, Any]): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            key = f"position_data:{position_id}" 
            def dt_serializer(obj):
                if isinstance(obj, datetime): return obj.isoformat()
                raise TypeError ("Type not serializable")
            pos_data_json = json.dumps(position_data, default=dt_serializer)
            await self.redis.setex(key, self.ttl_config['position'], pos_data_json)
            await self.redis.sadd("positions:active_ids", position_id)
        except Exception as e:
            logger.exception(f"Erro ao armazenar dados da posicao {position_id} no Redis:")

    async def get_position_data(self, position_id: str) -> Optional[Dict[str, Any]]: 
        if not self.is_connected_guard(): return None
        if not self.redis: return None
        try:
            key = f"position_data:{position_id}"
            data_bytes = await self.redis.get(key)
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                pos_data = json.loads(data_str)
                for k, v in pos_data.items():
                    if isinstance(v, str) and ('time' in k.lower() or 'timestamp' in k.lower()):
                        try: pos_data[k] = datetime.fromisoformat(v.replace('Z', '+00:00'))
                        except ValueError: pass 
                return pos_data
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter dados da posicao {position_id} do Redis:")
            return None

    async def remove_position_data(self, position_id: str): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            key = f"position_data:{position_id}"
            async with self.redis.pipeline(transaction=False) as pipe: 
                await pipe.delete(key)
                await pipe.srem("positions:active_ids", position_id)
                await pipe.execute()
            logger.debug(f"Dados da posicao {position_id} removidos do Redis.")
        except Exception as e:
            logger.exception(f"Erro ao remover dados da posicao {position_id} do Redis:")

    async def store_numpy_array(self, key: str, array: np.ndarray, ttl_seconds: Optional[int] = None): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            serialized_data = pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL) 
            effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_config.get('array', 3600)
            await self.redis.setex(f"numpy_array:{key}", effective_ttl, serialized_data)
        except Exception as e:
            logger.exception(f"Erro ao armazenar array numpy '{key}' no Redis:")

    async def get_numpy_array(self, key: str) -> Optional[np.ndarray]: 
        if not self.is_connected_guard(): return None
        if not self.redis: return None
        try:
            serialized_data = await self.redis.get(f"numpy_array:{key}")
            if serialized_data: 
                return pickle.loads(serialized_data) 
            return None
        except ModuleNotFoundError: 
            logger.error("Modulo 'pickle' nao encontrado para desserializar array numpy.")
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter array numpy '{key}' do Redis:")
            return None

    def is_connected_guard(self) -> bool:
        if not self.connected or not self.redis:
            logger.warning("Cliente Redis nao conectado. Operacao ignorada.")
            return False
        return True

    async def exists(self, key: str) -> bool:
        if not self.is_connected_guard(): return False
        if not self.redis: return False
        try:
            return await self.redis.exists(key) > 0 
        except Exception as e:
            logger.exception(f"Erro ao verificar existencia da chave '{key}' no Redis:")
            return False

    async def set_expiry(self, key: str, seconds: int): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            await self.redis.expire(key, seconds) 
        except Exception as e:
            logger.exception(f"Erro ao definir expiracao para chave '{key}' no Redis:")

    async def get_keys_by_pattern(self, pattern: str) -> List[str]: 
        if not self.is_connected_guard(): return []
        if not self.redis: return []
        try:
            keys_found: List[str] = []
            cursor = b'0' 
            while cursor: # Await a chamada scan
                cursor, current_keys_bytes = await self.redis.scan(cursor, match=pattern, count=100) 
                keys_found.extend([key.decode('utf-8') for key in current_keys_bytes])
            return keys_found
        except Exception as e:
            logger.exception(f"Erro ao buscar chaves por padrao '{pattern}' no Redis:")
            return []

    async def delete_keys_by_pattern(self, pattern: str): 
        if not self.is_connected_guard(): return
        if not self.redis: return
        try:
            keys_to_delete = await self.get_keys_by_pattern(pattern)
            if keys_to_delete:
                async with self.redis.pipeline(transaction=False) as pipe: 
                    for key_del in keys_to_delete: 
                        await pipe.unlink(key_del) 
                    await pipe.execute()
                logger.info(f"Removidas (ou agendadas para remocao) {len(keys_to_delete)} chaves com padrao '{pattern}' do Redis.")
            else:
                logger.info(f"Nenhuma chave encontrada com padrao '{pattern}' para remover do Redis.")
        except Exception as e:
            logger.exception(f"Erro ao limpar chaves por padrao '{pattern}' no Redis:")

    async def get_redis_info(self) -> Dict[str, Any]: 
        if not self.is_connected_guard(): return {}
        if not self.redis: return {}
        try:
            info: Dict[str, Any] = await self.redis.info() 
            parsed_info: Dict[str, Any] = {}
            for key_info, value_info in info.items(): 
                try:
                    if isinstance(value_info, str) and '.' in value_info:
                        parsed_info[key_info] = float(value_info)
                    elif isinstance(value_info, str) and value_info.isdigit():
                        parsed_info[key_info] = int(value_info)
                    else:
                        parsed_info[key_info] = value_info
                except ValueError:
                    parsed_info[key_info] = value_info 

            used_memory_mb = parsed_info.get('used_memory', 0.0) / (1024 * 1024)
            total_commands = parsed_info.get('total_commands_processed', 0)
            hits = parsed_info.get('keyspace_hits', 0)
            misses = parsed_info.get('keyspace_misses', 0)

            return {
                'redis_version': parsed_info.get('redis_version'),
                'uptime_in_seconds': parsed_info.get('uptime_in_seconds'),
                'used_memory_mb': round(used_memory_mb, 2),
                'connected_clients': parsed_info.get('connected_clients', 0),
                'total_commands_processed': total_commands,
                'keyspace_hits': hits,
                'keyspace_misses': misses,
                'hit_rate': round(hits / (hits + misses + 1e-9), 4) 
            }
        except Exception as e:
            logger.exception("Erro ao obter informacoes do Redis:")
            return {}

    async def health_check(self) -> bool:
        if not self.is_connected_guard(): return False
        if not self.redis: return False
        try:
            return await self.redis.ping() 
        except Exception:
            logger.warning("Falha no health check do Redis (ping).")
            self.connected = False 
            return False