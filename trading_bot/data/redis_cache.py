# data/redis_cache.py
"""Sistema de cache Redis para dados de mercado usando redis-py async"""
import redis.asyncio as redis_async
import json
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta, timezone, date
import asyncio
import numpy as np

from config.settings import CONFIG # Apenas CONFIG e necessario aqui, DM_CFG sera usado no DataManager
from utils.logger import setup_logger

logger = setup_logger("redis_cache")

class RedisCache:
    """Gerenciador de cache Redis para dados de alta frequencia com redis-py async"""

    def __init__(self):
        self.redis: Optional[redis_async.Redis] = None
        self.pool: Optional[redis_async.ConnectionPool] = None
        self.connected: bool = False

        # TTLs padrao para diferentes tipos de chaves em segundos
        self.ttl_config: Dict[str, int] = {
            'tick': CONFIG.REDIS_TTL_HOURS * 3600 if hasattr(CONFIG, 'REDIS_TTL_HOURS') else 24 * 3600,
            'ohlc': 86400,  # 1 dia para OHLC (pode ser mais curto se atualizado frequentemente)
            'indicator': 300,  # 5 minutos para indicadores
            'market_state': 60,  # 1 minuto para estado de mercado
            'position': 3600 * 24 * 7,  # 7 dias para dados de posicao (ou ate serem fechadas)
            'performance': 600,  # 10 minutos para metricas de performance
            'array': 3600, # 1 hora para arrays numpy
        }

    async def connect(self):
        """Conecta ao Redis usando redis-py async"""
        if self.connected and self.redis:
            try:
                await self.redis.ping()
                logger.info("Ja conectado ao Redis (redis-py).")
                return
            except (redis_async.exceptions.ConnectionError, ConnectionRefusedError, asyncio.TimeoutError) as e:
                logger.warning(f"Conexao Redis (redis-py) existente falhou: {e}. Tentando reconectar.")
                self.connected = False
                if self.redis:
                    try: await self.redis.close()
                    except Exception as ex_close: logger.debug(f"Excecao ao fechar redis existente: {ex_close}")
                if self.pool:
                    try: await self.pool.disconnect()
                    except Exception as ex_pool: logger.debug(f"Excecao ao desconectar pool existente: {ex_pool}")
                self.redis = None
                self.pool = None

        try:
            self.pool = redis_async.ConnectionPool.from_url(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                decode_responses=True # Retorna strings em vez de bytes para a maioria dos comandos
            )
            self.redis = redis_async.Redis(connection_pool=self.pool)
            await self.redis.ping()
            self.connected = True
            logger.info(f"Conectado ao Redis (redis-py) em {CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}, DB: {CONFIG.REDIS_DB}")

        except Exception as e:
            logger.exception("Erro ao conectar ao Redis (redis-py):")
            self.connected = False
            self.redis = None
            self.pool = None


    async def disconnect(self):
        """Desconecta do Redis (redis-py)"""
        try:
            if self.redis:
                await self.redis.close()
            if self.pool:
                await self.pool.disconnect()
            logger.info("Conexao com Redis (redis-py) fechada.")
        except Exception as e:
            logger.error(f"Erro ao desconectar do Redis: {e}")
        finally:
            self.redis = None
            self.pool = None
            self.connected = False


    def is_connected_guard(self) -> bool:
        """Guarda para verificar se o Redis esta conectado antes de uma operacao."""
        if not self.connected or not self.redis:
            # Nao logar warning aqui para cada chamada, pode poluir os logs.
            # O chamador deve decidir se loga.
            return False
        return True

    # === Metodos para Ticks ===

    async def store_tick(self, symbol: str, tick: Dict[str, Any]):
        if not self.is_connected_guard() or not self.redis: return

        try:
            tick_to_store = tick.copy()
            if isinstance(tick_to_store.get('timestamp'), datetime):
                tick_to_store['timestamp'] = tick_to_store['timestamp'].isoformat()

            tick_json = json.dumps(tick_to_store)
            ttl = self.ttl_config.get('tick', 3600 * 24) # Default 24h

            key_latest = f"tick:{symbol}:latest"
            await self.redis.setex(key_latest, ttl, tick_json)

            key_list = f"tick:{symbol}:list"
            pipe = self.redis.pipeline()
            pipe.lpush(key_list, tick_json)
            pipe.ltrim(key_list, 0, CONFIG.MAX_RECENT_TICKS -1 if hasattr(CONFIG, 'MAX_RECENT_TICKS') else 9999) # Usar MAX_RECENT_TICKS de CONFIG
            pipe.expire(key_list, ttl)
            await pipe.execute()
        except Exception as e:
            logger.exception(f"Erro ao armazenar tick para {symbol} no Redis (redis-py):")


    async def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected_guard() or not self.redis: return None
        try:
            key = f"tick:{symbol}:latest"
            data_str = await self.redis.get(key)

            if data_str:
                tick_dict = json.loads(data_str)
                if 'timestamp' in tick_dict and isinstance(tick_dict['timestamp'], str):
                    try:
                        # Tentar parse com 'Z' e sem, e com microsegundos variaveis
                        ts_str = tick_dict['timestamp'].replace('Z', '+00:00')
                        if '.' in ts_str and '+' in ts_str: # Formato com microsegundos e timezone
                             dt_part, tz_part = ts_str.split('+')
                             if len(dt_part.split('.')[-1]) > 6: # Truncar microsegundos
                                 dt_part = dt_part.split('.')[0] + '.' + dt_part.split('.')[-1][:6]
                             ts_str = dt_part + '+' + tz_part
                        tick_dict['timestamp'] = datetime.fromisoformat(ts_str)
                    except ValueError as ve:
                        logger.warning(f"Erro ao parsear timestamp ISO do Redis para tick: {tick_dict['timestamp']}, erro: {ve}. Retornando como string.")
                return tick_dict
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter ultimo tick para {symbol} do Redis (redis-py):")
            return None


    async def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        if not self.is_connected_guard() or not self.redis: return []
        try:
            key = f"tick:{symbol}:list"
            safe_count = min(count, CONFIG.MAX_RECENT_TICKS if hasattr(CONFIG, 'MAX_RECENT_TICKS') else 10000) # Limitar ao maximo armazenado
            ticks_str_list = await self.redis.lrange(key, 0, safe_count - 1)

            ticks: List[Dict[str, Any]] = []
            for tick_str_entry in ticks_str_list:
                if tick_str_entry:
                    try:
                        tick_dict = json.loads(tick_str_entry)
                        if 'timestamp' in tick_dict and isinstance(tick_dict['timestamp'], str):
                            try:
                                ts_str = tick_dict['timestamp'].replace('Z', '+00:00')
                                if '.' in ts_str and '+' in ts_str:
                                     dt_part, tz_part = ts_str.split('+')
                                     if len(dt_part.split('.')[-1]) > 6:
                                         dt_part = dt_part.split('.')[0] + '.' + dt_part.split('.')[-1][:6]
                                     ts_str = dt_part + '+' + tz_part
                                tick_dict['timestamp'] = datetime.fromisoformat(ts_str)
                            except ValueError as ve:
                                logger.warning(f"Erro ao parsear timestamp ISO do Redis para lista de ticks: {tick_dict['timestamp']}, erro: {ve}. Retornando como string.")
                        ticks.append(tick_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Erro ao decodificar JSON de tick recente do Redis (redis-py): {tick_str_entry}")
                    except Exception as e_parse:
                        logger.warning(f"Erro ao parsear tick recente do Redis (redis-py): {e_parse}, dado: {tick_str_entry}")
            return list(reversed(ticks)) # Historicamente, o mais recente e o primeiro na lista do lrange
        except Exception as e:
            logger.exception(f"Erro ao obter ticks recentes para {symbol} do Redis (redis-py):")
            return []

    # === Metodos para Indicadores ===

    async def store_indicator(self, symbol: str, indicator_name: str,
                            value: Union[float, int, str, bool, Dict[str, Any], List[Any]],
                            timeframe: str = "current"):
        if not self.is_connected_guard() or not self.redis: return
        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            data_to_store: str
            if isinstance(value, (dict, list, bool)): # bool e subclasse de int, mas json.dumps lida bem
                data_to_store = json.dumps(value)
            elif isinstance(value, (float, int)):
                data_to_store = str(value) # Armazenar numeros como string para consistencia no get
            elif isinstance(value, str):
                data_to_store = value
            else:
                logger.warning(f"Tipo de valor nao suportado para indicador '{indicator_name}': {type(value)}")
                return
            await self.redis.setex(key, self.ttl_config.get('indicator', 300), data_to_store)
        except Exception as e:
            logger.exception(f"Erro ao armazenar indicador {indicator_name} para {symbol} no Redis (redis-py):")


    async def get_indicator(self, symbol: str, indicator_name: str,
                          timeframe: str = "current") -> Optional[Any]:
        if not self.is_connected_guard() or not self.redis: return None
        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            data_str = await self.redis.get(key)

            if data_str is not None: # Checar por None explicitamente, pois string vazia pode ser valor valido
                try: # Tentar parsear como JSON primeiro (para dicts, lists, bools)
                    return json.loads(data_str)
                except json.JSONDecodeError:
                    try: # Tentar como float
                        return float(data_str)
                    except ValueError:
                        try: # Tentar como int
                            return int(data_str)
                        except ValueError: # Se nada disso, retornar como string (valor original)
                            return data_str
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter indicador {indicator_name} para {symbol} do Redis (redis-py):")
            return None

    # === Metodos para Estado de Mercado ===

    async def store_market_state(self, state: Dict[str, Any]):
        if not self.is_connected_guard() or not self.redis: return
        try:
            state_to_store = state.copy()
            state_to_store['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
            state_json = json.dumps(state_to_store)
            key_current = "market:state:current"
            ttl = self.ttl_config.get('market_state', 60)
            await self.redis.setex(key_current, ttl, state_json)

            # Opcional: armazenar historico de estados
            # history_key = "market:state:history"
            # pipe = self.redis.pipeline()
            # pipe.lpush(history_key, state_json)
            # pipe.ltrim(history_key, 0, 999)
            # pipe.expire(history_key, ttl * 24 * 7) # Ex: manter historico por 7 dias
            # await pipe.execute()
        except Exception as e:
            logger.exception("Erro ao armazenar estado do mercado no Redis (redis-py):")

    async def get_market_state(self) -> Optional[Dict[str, Any]]:
        if not self.is_connected_guard() or not self.redis: return None
        try:
            key = "market:state:current"
            data_str = await self.redis.get(key)
            if data_str:
                return json.loads(data_str)
            return None
        except Exception as e:
            logger.exception("Erro ao obter estado do mercado do Redis (redis-py):")
            return None

    # === Metodos para Performance de Estrategia ===

    async def store_strategy_performance(self, strategy_name: str, metrics: Dict[str, Any]):
        if not self.is_connected_guard() or not self.redis: return
        try:
            key = f"performance:strategy:{strategy_name}"
            metrics_json = json.dumps(metrics)
            await self.redis.setex(key, self.ttl_config.get('performance', 600), metrics_json)

            pipe = self.redis.pipeline()
            needs_execute = False
            if 'score' in metrics and isinstance(metrics['score'], (int, float)):
                pipe.zadd("strategies:ranking:score", {strategy_name: float(metrics['score'])})
                needs_execute = True
            # Adicionar outras metricas para ranking se necessario (ex: sharpe, pnl)
            # if 'sharpe_ratio' in metrics and isinstance(metrics['sharpe_ratio'], (int, float)):
            #      pipe.zadd("strategies:ranking:sharpe", {strategy_name: float(metrics['sharpe_ratio'])})
            #      needs_execute = True
            if needs_execute:
                await pipe.execute()
        except Exception as e:
            logger.exception(f"Erro ao armazenar performance de {strategy_name} no Redis (redis-py):")

    async def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected_guard() or not self.redis: return None
        try:
            key = f"performance:strategy:{strategy_name}"
            data_str = await self.redis.get(key)
            if data_str:
                return json.loads(data_str)
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter performance de {strategy_name} do Redis (redis-py):")
            return None

    async def get_top_strategies(self, count: int = 10, by_metric: str = 'score') -> List[Tuple[str, float]]:
        if not self.is_connected_guard() or not self.redis: return []
        ranking_key = f"strategies:ranking:{by_metric.lower()}"
        try:
            result_tuples = await self.redis.zrevrange(ranking_key, 0, count - 1, withscores=True)
            return [(str(name), float(score)) for name, score in result_tuples]
        except Exception as e:
            logger.exception(f"Erro ao obter ranking de estrategias (metrica: {by_metric}) do Redis (redis-py):")
            return []

    # === Metodos para Dados de Posicao ===

    async def store_position_data(self, position_id: str, position_data: Dict[str, Any]):
        if not self.is_connected_guard() or not self.redis: return
        try:
            key = f"position_data:{position_id}"
            def dt_serializer(obj):
                if isinstance(obj, datetime): return obj.isoformat()
                raise TypeError ("Tipo nao serializavel para JSON: %s" % type(obj))
            pos_data_json = json.dumps(position_data, default=dt_serializer)

            pipe = self.redis.pipeline()
            pipe.setex(key, self.ttl_config.get('position', 3600*24*7), pos_data_json)
            pipe.sadd("positions:active_ids", position_id)
            await pipe.execute()
        except Exception as e:
            logger.exception(f"Erro ao armazenar dados da posicao {position_id} no Redis (redis-py):")

    async def get_position_data(self, position_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected_guard() or not self.redis: return None
        try:
            key = f"position_data:{position_id}"
            data_str = await self.redis.get(key)
            if data_str:
                pos_data = json.loads(data_str)
                # Reconverter timestamps ISO para datetime
                for k, v in pos_data.items():
                    if isinstance(v, str) and ('time' in k.lower() or 'timestamp' in k.lower() or k.endswith('_at')):
                        try:
                            ts_str = v.replace('Z', '+00:00')
                            if '.' in ts_str and '+' in ts_str:
                                 dt_part, tz_part = ts_str.split('+')
                                 if len(dt_part.split('.')[-1]) > 6:
                                     dt_part = dt_part.split('.')[0] + '.' + dt_part.split('.')[-1][:6]
                                 ts_str = dt_part + '+' + tz_part
                            pos_data[k] = datetime.fromisoformat(ts_str)
                        except ValueError: pass # Manter como string se nao for ISO
                return pos_data
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter dados da posicao {position_id} do Redis (redis-py):")
            return None

    async def remove_position_data(self, position_id: str):
        if not self.is_connected_guard() or not self.redis: return
        try:
            key = f"position_data:{position_id}"
            pipe = self.redis.pipeline()
            pipe.delete(key)
            pipe.srem("positions:active_ids", position_id)
            await pipe.execute()
            logger.debug(f"Dados da posicao {position_id} removidos do Redis (redis-py).")
        except Exception as e:
            logger.exception(f"Erro ao remover dados da posicao {position_id} do Redis (redis-py):")

    # === Metodos para Arrays NumPy (usando pickle) ===

    async def store_numpy_array(self, key: str, array: np.ndarray, ttl_seconds: Optional[int] = None):
        if not self.is_connected_guard() or not self.redis: return
        try:
            # Usar um cliente Redis sem decode_responses para bytes
            pool_bytes = redis_async.ConnectionPool.from_url(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                decode_responses=False # Importante para bytes
            )
            redis_bytes_client = redis_async.Redis(connection_pool=pool_bytes)

            serialized_data_bytes = pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL)
            effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_config.get('array', 3600)

            await redis_bytes_client.setex(f"numpy_array:{key}", effective_ttl, serialized_data_bytes)
            await redis_bytes_client.close()
            await pool_bytes.disconnect()
        except Exception as e:
            logger.exception(f"Erro ao armazenar array numpy '{key}' no Redis (redis-py):")

    async def get_numpy_array(self, key: str) -> Optional[np.ndarray]:
        if not self.is_connected_guard() or not self.redis: return None
        try:
            pool_bytes = redis_async.ConnectionPool.from_url(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                decode_responses=False
            )
            redis_bytes_client = redis_async.Redis(connection_pool=pool_bytes)

            serialized_data_bytes = await redis_bytes_client.get(f"numpy_array:{key}")

            await redis_bytes_client.close()
            await pool_bytes.disconnect()

            if serialized_data_bytes:
                return pickle.loads(serialized_data_bytes)
            return None
        except ModuleNotFoundError:
            logger.error("Modulo 'pickle' nao encontrado para desserializar array numpy.")
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter array numpy '{key}' do Redis (redis-py):")
            return None

    # === Metodos Utilitarios ===

    async def exists(self, key: str) -> bool:
        if not self.is_connected_guard() or not self.redis: return False
        try:
            result = await self.redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.exception(f"Erro ao verificar existencia da chave '{key}' no Redis (redis-py):")
            return False

    async def set_expiry(self, key: str, seconds: int):
        if not self.is_connected_guard() or not self.redis: return
        try:
            await self.redis.expire(key, seconds)
        except Exception as e:
            logger.exception(f"Erro ao definir expiracao para chave '{key}' no Redis (redis-py):")

    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        if not self.is_connected_guard() or not self.redis: return []
        try:
            keys_found: List[str] = []
            cursor = '0' # scan usa cursor string/bytes dependendo de decode_responses
            while cursor != 0: # Redis retorna 0 quando a iteracao esta completa
                # Se decode_responses=True no pool principal, cursor sera string
                # Se o pool for de bytes, cursor sera bytes
                # Para consistencia, o pool principal do self.redis e decode_responses=True
                # entao o cursor inicial e '0'
                cursor, current_keys = await self.redis.scan(cursor, match=pattern, count=1000)
                keys_found.extend(current_keys)
                if cursor == '0' or cursor == 0: # Checar ambos para seguranca
                    break
            return keys_found
        except Exception as e:
            logger.exception(f"Erro ao buscar chaves por padrao '{pattern}' no Redis (redis-py):")
            return []

    async def delete_keys_by_pattern(self, pattern: str):
        if not self.is_connected_guard() or not self.redis: return
        try:
            keys_to_delete = await self.get_keys_by_pattern(pattern)
            if keys_to_delete:
                # Garantir que a lista nao esteja vazia antes de chamar unlink
                # unlink aceita multiplos argumentos de chave
                await self.redis.unlink(*keys_to_delete)
                logger.info(f"Removidas (ou agendadas para remocao) {len(keys_to_delete)} chaves com padrao '{pattern}' do Redis (redis-py).")
            else:
                logger.info(f"Nenhuma chave encontrada com padrao '{pattern}' para remover do Redis (redis-py).")
        except Exception as e:
            logger.exception(f"Erro ao limpar chaves por padrao '{pattern}' no Redis (redis-py):")

    async def get_redis_info(self) -> Dict[str, Any]:
        if not self.is_connected_guard() or not self.redis: return {}
        try:
            info: Dict[str, Any] = await self.redis.info() # info retorna um dict

            used_memory_mb = info.get('used_memory', 0) / (1024 * 1024) # used_memory e int
            total_commands = info.get('total_commands_processed', 0)
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)

            return {
                'redis_version': info.get('redis_version'),
                'uptime_in_seconds': info.get('uptime_in_seconds'),
                'used_memory_mb': round(used_memory_mb, 2),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': total_commands,
                'keyspace_hits': hits,
                'keyspace_misses': misses,
                'hit_rate': round(hits / (hits + misses + 1e-9), 4) if (hits + misses) > 0 else 0.0
            }
        except Exception as e:
            logger.exception("Erro ao obter informacoes do Redis (redis-py):")
            return {}

    async def health_check(self) -> bool:
        if not self.is_connected_guard() or not self.redis:
            logger.warning("Health check: Redis nao conectado.")
            return False
        try:
            ping_result = await self.redis.ping()
            if not ping_result: # ping() retorna True em sucesso
                logger.warning("Falha no health check do Redis (redis-py) (ping retornou False).")
                self.connected = False # Marcar como desconectado se o ping falhar
                return False
            return True
        except Exception as e:
            logger.warning(f"Falha no health check do Redis (redis-py) (ping com excecao: {e}).")
            self.connected = False
            return False