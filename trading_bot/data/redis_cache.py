# data/redis_cache.py
"""Sistema de cache Redis para dados de mercado"""
import aioredis # type: ignore # Se aioredis causar problemas de tipo com o linter
import json
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple # Adicionado Tuple
from datetime import datetime, timedelta, timezone # Adicionado timezone
import asyncio
import numpy as np

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("redis_cache")

class RedisCache:
    """Gerenciador de cache Redis para dados de alta frequência"""

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None # Tipagem mais específica
        self.connected: bool = False

        # Configurações de TTL (segundos)
        self.ttl_config: Dict[str, int] = {
            'tick': CONFIG.REDIS_TTL_HOURS * 3600, # Ticks individuais e lista recente
            'ohlc': 86400,  # 24 horas para barras OHLC cacheadas
            'indicator': 300,  # 5 minutos para valores de indicadores
            'market_state': 60,  # 1 minuto para o estado agregado do mercado
            'position': 3600,  # 1 hora para dados de posição (atualizado com frequência menor)
            'performance': 600,  # 10 minutos para métricas de performance
            'array': 3600, # 1 hora para arrays genéricos
        }

    async def connect(self):
        """Conecta ao Redis"""
        if self.connected and self.redis:
            try: # Testar a conexão existente
                await self.redis.ping()
                logger.info("Já conectado ao Redis.")
                return
            except (aioredis.exceptions.ConnectionError, ConnectionRefusedError, asyncio.TimeoutError):
                logger.warning("Conexão Redis existente falhou. Tentando reconectar.")
                self.connected = False # Forçar reconexão
                if self.redis:
                    await self.redis.close() # Fechar pool antigo explicitamente
                    self.redis = None


        try:
            # Usar from_url para criar a conexão/pool
            self.redis = await aioredis.from_url(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                encoding='utf-8', # Encoding para chaves e valores string
                # decode_responses=True, # Se quiser que o aioredis decodifique automaticamente para str
                health_check_interval=30 # Verificar saúde da conexão a cada 30s
            )
            await self.redis.ping() # Testar a nova conexão
            self.connected = True
            logger.info(f"Conectado ao Redis em {CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}")

        except Exception as e:
            logger.exception("Erro ao conectar ao Redis:") # Usar logger.exception
            self.connected = False
            self.redis = None # Garantir que redis seja None se a conexão falhar
            raise # Relançar para que o chamador saiba da falha


    async def disconnect(self):
        """Desconecta do Redis"""
        if self.redis:
            try:
                await self.redis.close()
                # await self.redis.wait_closed() # wait_closed não é mais necessário em aioredis 2.x
                logger.info("Conexão com Redis fechada.")
            except Exception as e:
                logger.error(f"Erro ao fechar conexão Redis: {e}")
            finally:
                self.redis = None
                self.connected = False
        else:
            logger.info("Nenhuma conexão Redis para fechar.")
        self.connected = False


    # === Métodos para Ticks ===

    async def store_tick(self, symbol: str, tick: Dict[str, Any]): # Usar Any
        """Armazena tick no cache (último tick e lista recente)."""
        if not self.is_connected_guard(): return # Usar guarda

        try:
            # Garantir que o timestamp no tick é string ISO para consistência no JSON
            if isinstance(tick.get('timestamp'), datetime):
                tick_to_store = tick.copy()
                tick_to_store['timestamp'] = tick['timestamp'].isoformat()
            else:
                tick_to_store = tick

            tick_json = json.dumps(tick_to_store)

            key_latest = f"tick:{symbol}:latest"
            await self.redis.setex( # type: ignore
                key_latest,
                self.ttl_config['tick'],
                tick_json
            )

            key_list = f"tick:{symbol}:list"
            # Usar pipeline para operações atômicas se possível e para performance
            async with self.redis.pipeline(transaction=False) as pipe: # type: ignore
                await pipe.lpush(key_list, tick_json)
                await pipe.ltrim(key_list, 0, 9999)  # Manter últimos 10k
                await pipe.expire(key_list, self.ttl_config['tick'])
                await pipe.execute()


            # Atualizar timestamp do símbolo ativo (opcional, se usado para rastrear atividade)
            # await self.redis.zadd( # type: ignore
            #     "symbols:active",
            #     {symbol: int(datetime.now(timezone.utc).timestamp())} # Usar dict para zadd
            # )

        except Exception as e:
            logger.exception(f"Erro ao armazenar tick para {symbol} no Redis:")


    async def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]: # Usar Any
        """Obtém último tick de um símbolo do Redis."""
        if not self.is_connected_guard(): return None

        try:
            key = f"tick:{symbol}:latest"
            data_bytes = await self.redis.get(key) # type: ignore

            if data_bytes: # aioredis com encoding='utf-8' retorna str, senão bytes
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                tick_dict = json.loads(data_str)
                # Converter timestamp de volta para datetime se necessário
                if 'timestamp' in tick_dict and isinstance(tick_dict['timestamp'], str):
                    try:
                        tick_dict['timestamp'] = datetime.fromisoformat(tick_dict['timestamp'].replace('Z', '+00:00'))
                    except ValueError: # Fallback se não for ISO
                        tick_dict['timestamp'] = datetime.now(timezone.utc) # Ou None
                return tick_dict
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter último tick para {symbol} do Redis:")
            return None


    async def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]: # Usar Any
        """Obtém ticks recentes do Redis."""
        if not self.is_connected_guard(): return []

        try:
            key = f"tick:{symbol}:list"
            # lrange retorna List[Optional[bytes]] se decode_responses=False
            # ou List[Optional[str]] se decode_responses=True (ou encoding set)
            ticks_json_list = await self.redis.lrange(key, 0, count - 1) # type: ignore

            ticks: List[Dict[str, Any]] = []
            for tick_entry in ticks_json_list: # tick_entry pode ser str ou bytes
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

            return list(reversed(ticks))  # Ordem cronológica (lpush insere na cabeça)
        except Exception as e:
            logger.exception(f"Erro ao obter ticks recentes para {symbol} do Redis:")
            return []

    # === Métodos para Indicadores ===

    async def store_indicator(self, symbol: str, indicator_name: str,
                            value: Union[float, int, str, Dict[str, Any], List[Any]], # Tipagem mais ampla
                            timeframe: str = "current"):
        """Armazena valor de indicador no Redis."""
        if not self.is_connected_guard(): return

        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"

            if isinstance(value, (dict, list)):
                data_to_store = json.dumps(value)
            elif isinstance(value, (float, int)):
                data_to_store = str(value)
            elif isinstance(value, str):
                data_to_store = value
            else:
                logger.warning(f"Tipo de valor não suportado para indicador '{indicator_name}': {type(value)}")
                return


            await self.redis.setex(key, self.ttl_config['indicator'], data_to_store) # type: ignore
        except Exception as e:
            logger.exception(f"Erro ao armazenar indicador {indicator_name} para {symbol} no Redis:")


    async def get_indicator(self, symbol: str, indicator_name: str,
                          timeframe: str = "current") -> Optional[Any]: # Retorno pode ser de vários tipos
        """Obtém valor de indicador do Redis."""
        if not self.is_connected_guard(): return None

        try:
            key = f"indicator:{symbol}:{timeframe}:{indicator_name}"
            data_bytes = await self.redis.get(key) # type: ignore

            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                try:
                    # Tentar decodificar como JSON primeiro (para dicts/lists)
                    return json.loads(data_str)
                except json.JSONDecodeError:
                    # Se falhar, tentar converter para float, depois int, senão retornar como string
                    try:
                        return float(data_str)
                    except ValueError:
                        try:
                            return int(data_str)
                        except ValueError:
                            return data_str # Retornar como string se não for JSON nem numérico
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter indicador {indicator_name} para {symbol} do Redis:")
            return None

    # === Métodos para Estado do Mercado ===

    async def store_market_state(self, state: Dict[str, Any]): # Usar Any
        """Armazena estado atual do mercado no Redis."""
        if not self.is_connected_guard(): return

        try:
            # Adicionar timestamp ao estado antes de salvar
            state_to_store = state.copy()
            state_to_store['timestamp_utc'] = datetime.now(timezone.utc).isoformat()

            state_json = json.dumps(state_to_store)

            key_current = "market:state:current"
            await self.redis.setex(key_current, self.ttl_config['market_state'], state_json) # type: ignore

            history_key = "market:state:history"
            async with self.redis.pipeline(transaction=False) as pipe: # type: ignore
                await pipe.lpush(history_key, state_json)
                await pipe.ltrim(history_key, 0, 999) # Manter últimos 1000 estados
                await pipe.expire(history_key, self.ttl_config['market_state'] * 24) # Manter histórico por mais tempo
                await pipe.execute()

        except Exception as e:
            logger.exception("Erro ao armazenar estado do mercado no Redis:")


    async def get_market_state(self) -> Optional[Dict[str, Any]]: # Usar Any
        """Obtém estado atual do mercado do Redis."""
        if not self.is_connected_guard(): return None

        try:
            key = "market:state:current"
            data_bytes = await self.redis.get(key) # type: ignore
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                return json.loads(data_str)
            return None
        except Exception as e:
            logger.exception("Erro ao obter estado do mercado do Redis:")
            return None

    # === Métodos para Performance ===

    async def store_strategy_performance(self, strategy_name: str, metrics: Dict[str, Any]): # Usar Any
        """Armazena métricas de performance de estratégia no Redis."""
        if not self.is_connected_guard(): return

        try:
            key = f"performance:strategy:{strategy_name}"
            metrics_json = json.dumps(metrics) # default=str não necessário se métricas já são serializáveis
            await self.redis.setex(key, self.ttl_config['performance'], metrics_json) # type: ignore

            if 'score' in metrics and isinstance(metrics['score'], (int, float)):
                await self.redis.zadd( # type: ignore
                    "strategies:ranking:score", # Chave mais específica
                    {strategy_name: float(metrics['score'])}
                )
            if 'sharpe_ratio' in metrics and isinstance(metrics['sharpe_ratio'], (int, float)):
                 await self.redis.zadd( # type: ignore
                    "strategies:ranking:sharpe",
                    {strategy_name: float(metrics['sharpe_ratio'])}
                )


        except Exception as e:
            logger.exception(f"Erro ao armazenar performance de {strategy_name} no Redis:")


    async def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]: # Usar Any
        """Obtém métricas de performance de estratégia do Redis."""
        if not self.is_connected_guard(): return None

        try:
            key = f"performance:strategy:{strategy_name}"
            data_bytes = await self.redis.get(key) # type: ignore
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                return json.loads(data_str)
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter performance de {strategy_name} do Redis:")
            return None

    async def get_top_strategies(self, count: int = 10, by_metric: str = 'score') -> List[Tuple[str, float]]:
        """Obtém ranking das melhores estratégias por uma métrica (score, sharpe)."""
        if not self.is_connected_guard(): return []

        ranking_key = f"strategies:ranking:{by_metric.lower()}"
        try:
            # ZREVRANGE para pegar do maior para o menor score
            # Retorna List[bytes] ou List[str] dependendo da configuração do cliente
            result_tuples = await self.redis.zrevrange( # type: ignore
                ranking_key,
                0,
                count - 1,
                withscores=True
            )
            # Converter para o formato esperado (str, float)
            # aioredis retorna (value_bytes, score_float)
            parsed_results: List[Tuple[str, float]] = []
            for item in result_tuples:
                name_bytes, score_val = item # Desempacotar tupla
                name_str = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)
                parsed_results.append((name_str, float(score_val)))
            return parsed_results

        except Exception as e:
            logger.exception(f"Erro ao obter ranking de estratégias (métrica: {by_metric}) do Redis:")
            return []


    # === Métodos para Posições ===
    # Estes métodos podem ser úteis se o ExecutionEngine ou Orchestrator
    # quiserem cachear dados de posição no Redis para acesso rápido ou compartilhamento.

    async def store_position_data(self, position_id: str, position_data: Dict[str, Any]): # Renomeado, Usar Any
        """Armazena dados de posição no Redis."""
        if not self.is_connected_guard(): return

        try:
            key = f"position_data:{position_id}" # Chave mais específica
            # Garantir que datetimes sejam serializados (ex: para ISO string)
            def dt_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError ("Type not serializable")
            pos_data_json = json.dumps(position_data, default=dt_serializer)

            await self.redis.setex(key, self.ttl_config['position'], pos_data_json) # type: ignore
            await self.redis.sadd("positions:active_ids", position_id) # type: ignore
        except Exception as e:
            logger.exception(f"Erro ao armazenar dados da posição {position_id} no Redis:")


    async def get_position_data(self, position_id: str) -> Optional[Dict[str, Any]]: # Renomeado, Usar Any
        """Obtém dados de posição do Redis."""
        if not self.is_connected_guard(): return None

        try:
            key = f"position_data:{position_id}"
            data_bytes = await self.redis.get(key) # type: ignore
            if data_bytes:
                data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes
                pos_data = json.loads(data_str)
                # Converter timestamps de volta se necessário
                for k, v in pos_data.items():
                    if isinstance(v, str) and ('time' in k.lower() or 'timestamp' in k.lower()):
                        try:
                            pos_data[k] = datetime.fromisoformat(v.replace('Z', '+00:00'))
                        except ValueError:
                            pass # Manter como string se não for ISO
                return pos_data
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter dados da posição {position_id} do Redis:")
            return None

    async def remove_position_data(self, position_id: str): # Renomeado
        """Remove dados de posição do cache Redis."""
        if not self.is_connected_guard(): return

        try:
            key = f"position_data:{position_id}"
            async with self.redis.pipeline(transaction=False) as pipe: # type: ignore
                await pipe.delete(key)
                await pipe.srem("positions:active_ids", position_id)
                await pipe.execute()
            logger.debug(f"Dados da posição {position_id} removidos do Redis.")
        except Exception as e:
            logger.exception(f"Erro ao remover dados da posição {position_id} do Redis:")


    # === Métodos para Arrays/Séries ===
    async def store_numpy_array(self, key: str, array: np.ndarray, ttl_seconds: Optional[int] = None): # Renomeado, Usar ttl_seconds
        """Armazena array numpy usando pickle (cuidado com segurança e compatibilidade)."""
        if not self.is_connected_guard(): return

        try:
            # Serializar com pickle e compressão pode ser uma opção para arrays grandes
            # import gzip
            # serialized_data = gzip.compress(pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL))
            serialized_data = pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL) # Usar protocolo alto
            effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_config.get('array', 3600)

            await self.redis.setex(f"numpy_array:{key}", effective_ttl, serialized_data) # type: ignore
        except Exception as e:
            logger.exception(f"Erro ao armazenar array numpy '{key}' no Redis:")


    async def get_numpy_array(self, key: str) -> Optional[np.ndarray]: # Renomeado
        """Obtém array numpy do Redis."""
        if not self.is_connected_guard(): return None

        try:
            serialized_data = await self.redis.get(f"numpy_array:{key}") # type: ignore
            if serialized_data: # serialized_data será bytes
                # import gzip
                # return pickle.loads(gzip.decompress(serialized_data))
                return pickle.loads(serialized_data) # type: ignore
            return None
        except ModuleNotFoundError: # Se pickle não estiver disponível (improvável)
            logger.error("Módulo 'pickle' não encontrado para desserializar array numpy.")
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter array numpy '{key}' do Redis:")
            return None

    # === Métodos Utilitários ===

    def is_connected_guard(self) -> bool:
        """Guarda para verificar se o Redis está conectado antes de uma operação."""
        if not self.connected or not self.redis:
            logger.warning("Cliente Redis não conectado. Operação ignorada.")
            return False
        return True


    async def exists(self, key: str) -> bool:
        """Verifica se chave existe no Redis."""
        if not self.is_connected_guard(): return False
        try:
            return await self.redis.exists(key) > 0 # type: ignore
        except Exception as e:
            logger.exception(f"Erro ao verificar existência da chave '{key}' no Redis:")
            return False

    async def set_expiry(self, key: str, seconds: int): # Renomeado de expire_in
        """Define expiração para chave no Redis."""
        if not self.is_connected_guard(): return
        try:
            await self.redis.expire(key, seconds) # type: ignore
        except Exception as e:
            logger.exception(f"Erro ao definir expiração para chave '{key}' no Redis:")


    async def get_keys_by_pattern(self, pattern: str) -> List[str]: # Renomeado
        """Busca chaves por padrão no Redis. Use com CUIDADO em produção (pode ser lento)."""
        if not self.is_connected_guard(): return []
        try:
            # KEYS é bloqueante. SCAN é preferível para produção.
            # result_keys_bytes = await self.redis.keys(pattern) # type: ignore
            # return [key.decode('utf-8') for key in result_keys_bytes]
            
            # Usando SCAN para melhor performance em produção
            keys_found: List[str] = []
            cursor = b'0' # Cursor inicial para SCAN precisa ser bytes
            while cursor:
                cursor, current_keys_bytes = await self.redis.scan(cursor, match=pattern, count=100) # type: ignore
                keys_found.extend([key.decode('utf-8') for key in current_keys_bytes])
            return keys_found

        except Exception as e:
            logger.exception(f"Erro ao buscar chaves por padrão '{pattern}' no Redis:")
            return []

    async def delete_keys_by_pattern(self, pattern: str): # Renomeado de flush_pattern
        """Remove todas as chaves que correspondem ao padrão no Redis. Use com CUIDADO."""
        if not self.is_connected_guard(): return
        try:
            keys_to_delete = await self.get_keys_by_pattern(pattern)
            if keys_to_delete:
                # Deletar em batch se possível, ou um por um.
                # Para muitos chaves, usar pipeline ou UNLINK (não bloqueante)
                async with self.redis.pipeline(transaction=False) as pipe: # type: ignore
                    for key_del in keys_to_delete: # Renomeado
                        await pipe.unlink(key_del) # UNLINK é não bloqueante no servidor
                    await pipe.execute()
                logger.info(f"Removidas (ou agendadas para remoção) {len(keys_to_delete)} chaves com padrão '{pattern}' do Redis.")
            else:
                logger.info(f"Nenhuma chave encontrada com padrão '{pattern}' para remover do Redis.")
        except Exception as e:
            logger.exception(f"Erro ao limpar chaves por padrão '{pattern}' no Redis:")


    async def get_redis_info(self) -> Dict[str, Any]: # Renomeado, Usar Any
        """Obtém informações do servidor Redis."""
        if not self.is_connected_guard(): return {}
        try:
            info: Dict[str, Any] = await self.redis.info() # type: ignore
            # Converter valores numéricos para tipos corretos
            parsed_info: Dict[str, Any] = {}
            for key_info, value_info in info.items(): # Renomeado
                try:
                    if isinstance(value_info, str) and '.' in value_info:
                        parsed_info[key_info] = float(value_info)
                    elif isinstance(value_info, str) and value_info.isdigit():
                        parsed_info[key_info] = int(value_info)
                    else:
                        parsed_info[key_info] = value_info
                except ValueError:
                    parsed_info[key_info] = value_info # Manter como string se não puder converter


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
                'hit_rate': round(hits / (hits + misses + 1e-9), 4) # Evitar divisão por zero
            }
        except Exception as e:
            logger.exception("Erro ao obter informações do Redis:")
            return {}


    async def health_check(self) -> bool:
        """Verifica saúde da conexão Redis."""
        if not self.is_connected_guard(): return False
        try:
            return await self.redis.ping() # type: ignore
        except Exception:
            logger.warning("Falha no health check do Redis (ping).")
            self.connected = False # Marcar como não conectado se o ping falhar
            return False