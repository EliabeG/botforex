# data/market_data.py
"""Processamento e agregação de dados de mercado"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any # Adicionado Any
from datetime import datetime, timedelta, timezone # Adicionado timezone
from collections import deque
import asyncio # Não usado diretamente, mas pode ser relevante se houver I/O futuro

from utils.logger import setup_logger
# Importar TickData se for usar explicitamente, senão o dicionário é suficiente
# from api.ticktrader_ws import TickData

logger = setup_logger("market_data")

class MarketDataProcessor:
    """Processador de dados de mercado em tempo real"""

    def __init__(self):
        # Buffers para diferentes timeframes
        self.tick_buffers: Dict[str, deque[Dict[str, Any]]] = {} # Tipagem mais específica
        self.ohlc_data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {} # Symbol -> Timeframe -> PeriodKey -> BarData
        self.volume_profiles: Dict[str, Dict[str, Any]] = {} # Symbol -> ProfileData

        # Configurações
        self.timeframes: List[str] = ['1T', '5T', '15T', '1H', '4H', '1D']  # pandas freq strings
        self.max_buffer_size: int = 100000 # Aumentado, mas depende da memória disponível

    def process_tick(self, symbol: str, tick: Dict[str, Any]): # tick é um dicionário
        """Processa um tick e atualiza estruturas de dados"""
        if not isinstance(tick, dict) or 'timestamp' not in tick or 'mid' not in tick:
            logger.warning(f"Tick inválido recebido para {symbol}: {tick}")
            return

        # Inicializar buffer se necessário
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.ohlc_data[symbol] = {} # Inicializar dict para o símbolo
            self.volume_profiles[symbol] = {'current': {}} # Inicializar perfil de volume atual
            for tf in self.timeframes: # Inicializar todos os timeframes para o símbolo
                self.ohlc_data[symbol][tf] = {}


        # Adicionar ao buffer
        self.tick_buffers[symbol].append(tick)

        # Atualizar OHLC
        self._update_ohlc(symbol, tick)

        # Atualizar volume profile
        self._update_volume_profile(symbol, tick)

    def _update_ohlc(self, symbol: str, tick: Dict[str, Any]):
        """Atualiza dados OHLC para diferentes timeframes"""
        try:
            # Garantir que o timestamp do tick seja um objeto datetime ciente do fuso horário (UTC)
            if isinstance(tick['timestamp'], (int, float)):
                timestamp_dt = datetime.fromtimestamp(tick['timestamp'] / 1000, tz=timezone.utc)
            elif isinstance(tick['timestamp'], str):
                timestamp_dt = pd.Timestamp(tick['timestamp'], tz='UTC').to_pydatetime(warn=False)
            elif isinstance(tick['timestamp'], datetime):
                timestamp_dt = tick['timestamp']
                if timestamp_dt.tzinfo is None: # Se for naive, assumir UTC
                    timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)
            else:
                logger.warning(f"Formato de timestamp desconhecido no tick para OHLC: {tick['timestamp']}")
                return

            price = float(tick['mid'])
            # Estimar volume do tick
            volume = (float(tick.get('bid_volume', 0.0)) + float(tick.get('ask_volume', 0.0))) / 2.0
            if volume == 0.0: # Se não houver volume de bid/ask, usar um valor mínimo simbólico ou 1 tick
                volume = tick.get('volume', 1.0) # Se houver um campo 'volume' direto

        except (TypeError, ValueError) as e:
            logger.error(f"Erro ao processar dados do tick para OHLC: {e}. Tick: {tick}")
            return


        for tf_pandas_str in self.timeframes: # Renomeado tf para tf_pandas_str
            # Garantir que o timeframe existe para o símbolo
            if tf_pandas_str not in self.ohlc_data[symbol]:
                self.ohlc_data[symbol][tf_pandas_str] = {}

            # Determinar período atual
            # pd.Timestamp.floor() é eficiente para isso
            period_start_dt = pd.Timestamp(timestamp_dt).floor(tf_pandas_str).to_pydatetime(warn=False)
            period_key = period_start_dt.isoformat() # Usar ISO format como chave garante unicidade e ordenação

            current_bar = self.ohlc_data[symbol][tf_pandas_str].get(period_key)

            if not current_bar:
                # Novo período
                self.ohlc_data[symbol][tf_pandas_str][period_key] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'tick_count': 1,
                    'timestamp': period_start_dt # Armazenar como datetime object
                }
            else:
                # Atualizar período existente
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += volume
                current_bar['tick_count'] += 1

    def _update_volume_profile(self, symbol: str, tick: Dict[str, Any]):
        """Atualiza perfil de volume intraday (ou da sessão)."""
        try:
            price = round(float(tick['mid']), 5)  # Arredondar para nível de preço (5 casas para EURUSD)
            bid_vol = float(tick.get('bid_volume', 0.0))
            ask_vol = float(tick.get('ask_volume', 0.0))
            # Volume do tick (pode ser diferente da soma de bid/ask volume no top of book)
            # Se o tick tiver um campo 'LastSize' ou 'VolumeTraded', usar esse.
            # Senão, estimar.
            tick_actual_volume = float(tick.get('volume', (bid_vol + ask_vol) / 2.0 ))
            if tick_actual_volume == 0 and (bid_vol > 0 or ask_vol > 0): # Fallback se 'volume' for 0 mas bid/ask vol não
                tick_actual_volume = 1.0 # Contar como 1 unidade de trade se não houver volume melhor

        except (TypeError, ValueError) as e:
            logger.error(f"Erro ao processar dados do tick para Volume Profile: {e}. Tick: {tick}")
            return

        # Assegurar que o perfil 'current' exista para o símbolo
        if symbol not in self.volume_profiles:
            self.volume_profiles[symbol] = {'current': {}}
        elif 'current' not in self.volume_profiles[symbol]:
             self.volume_profiles[symbol]['current'] = {}

        profile = self.volume_profiles[symbol]['current']

        if price not in profile:
            profile[price] = {
                'volume': 0.0,
                'buy_volume': 0.0, # Volume agressor de compra
                'sell_volume': 0.0, # Volume agressor de venda
                'count': 0 # Número de trades (ticks) nesse nível
            }

        profile[price]['volume'] += tick_actual_volume
        profile[price]['count'] += 1

        # Estimar direção do fluxo agressor (simplificado)
        # Idealmente, isso viria do 'AggressorSide' ou comparando com o tick anterior.
        last_price_field = tick.get('last_price', tick['mid']) # Se o tick tiver 'last_price' (preço do último trade)
        if float(last_price_field) >= float(tick['ask']): # Trade no ask ou acima -> comprador agrediu
            profile[price]['buy_volume'] += tick_actual_volume
        elif float(last_price_field) <= float(tick['bid']): # Trade no bid ou abaixo -> vendedor agrediu
            profile[price]['sell_volume'] += tick_actual_volume
        # else: trade entre spread, difícil determinar agressor sem mais info

    def get_ohlc(self, symbol: str, timeframe: str,
                 periods: int = 100) -> pd.DataFrame:
        """Retorna dados OHLC para timeframe específico, ordenados e com timestamp como índice."""
        if symbol not in self.ohlc_data or timeframe not in self.ohlc_data[symbol]:
            return pd.DataFrame()

        # Converter para DataFrame e ordenar
        # As chaves do dicionário (period_key) são strings ISO, então a ordenação lexicográfica funciona.
        # No entanto, é mais robusto converter para datetime e ordenar.
        bars_dict = self.ohlc_data[symbol][timeframe]
        if not bars_dict:
            return pd.DataFrame()

        # Criar lista de dicionários, adicionando a chave como 'period_start_iso'
        bars_list = []
        for period_iso, bar_data in bars_dict.items():
            # O timestamp já deve ser um objeto datetime
            # Se não for, converter: bar_data['timestamp'] = datetime.fromisoformat(period_iso)
            bars_list.append(bar_data)


        if not bars_list:
            return pd.DataFrame()

        df = pd.DataFrame(bars_list)
        df.sort_values('timestamp', inplace=True) # Ordenar por timestamp
        df.set_index('timestamp', inplace=True)   # Definir timestamp como índice

        # Selecionar colunas e últimos N períodos
        df = df[['open', 'high', 'low', 'close', 'volume', 'tick_count']]
        if len(df) > periods:
            return df.iloc[-periods:]
        return df


    def get_volume_profile(self, symbol: str, reset_period: Optional[str] = None) -> Dict[str, Any]:
        """
        Retorna perfil de volume atual.
        'reset_period' pode ser 'daily', 'hourly' para resetar o perfil (não implementado aqui, apenas retorna 'current').
        """
        if symbol not in self.volume_profiles or 'current' not in self.volume_profiles[symbol]:
            return {'profile': {}, 'poc': 0.0, 'vah': 0.0, 'val': 0.0, 'total_volume': 0.0}

        profile = self.volume_profiles[symbol].get('current', {})
        if not profile:
            return {'profile': {}, 'poc': 0.0, 'vah': 0.0, 'val': 0.0, 'total_volume': 0.0}

        # Calcular estatísticas
        prices = list(profile.keys())
        volumes = np.array([p_data['volume'] for p_data in profile.values()]) # Renomeado p para p_data

        total_volume = np.sum(volumes)
        if total_volume == 0:
            return {'profile': profile, 'poc': 0.0, 'vah': 0.0, 'val': 0.0, 'total_volume': 0.0}


        # POC (Point of Control) - preço com maior volume
        poc_idx = np.argmax(volumes)
        poc_price = prices[poc_idx]

        # Value Area (70% do volume)
        # Ordenar por preço para cálculo correto da Value Area
        sorted_profile_by_price = sorted(profile.items(), key=lambda item: item[0]) # item[0] é o preço

        # Encontrar o POC dentro do perfil ordenado por preço para iniciar a VA
        poc_item = next(item for item in sorted_profile_by_price if item[0] == poc_price)
        poc_index_in_sorted = sorted_profile_by_price.index(poc_item)


        # Expandir a partir do POC para formar a Value Area
        value_area_volume_target = total_volume * 0.7
        current_va_volume = poc_item[1]['volume']
        value_area_prices = {poc_price}


        # Índices para expansão
        low_idx = poc_index_in_sorted - 1
        high_idx = poc_index_in_sorted + 1

        while current_va_volume < value_area_volume_target:
            add_low = False
            add_high = False

            if low_idx >= 0 and high_idx < len(sorted_profile_by_price):
                if sorted_profile_by_price[low_idx][1]['volume'] >= sorted_profile_by_price[high_idx][1]['volume']:
                    add_low = True
                else:
                    add_high = True
            elif low_idx >= 0:
                add_low = True
            elif high_idx < len(sorted_profile_by_price):
                add_high = True
            else:
                break # Não há mais níveis para adicionar

            if add_low:
                current_va_volume += sorted_profile_by_price[low_idx][1]['volume']
                value_area_prices.add(sorted_profile_by_price[low_idx][0])
                low_idx -= 1
            elif add_high:
                current_va_volume += sorted_profile_by_price[high_idx][1]['volume']
                value_area_prices.add(sorted_profile_by_price[high_idx][0])
                high_idx += 1
        
        vah = max(value_area_prices) if value_area_prices else poc_price
        val = min(value_area_prices) if value_area_prices else poc_price


        return {
            'profile': dict(sorted(profile.items())), # Retornar perfil ordenado por preço
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'total_volume': total_volume
        }

    def calculate_vwap(self, symbol: str, periods: Optional[int] = None) -> float:
        """Calcula VWAP (Volume Weighted Average Price) dos ticks no buffer."""
        if symbol not in self.tick_buffers or not self.tick_buffers[symbol]:
            return 0.0

        # Usar uma cópia para evitar problemas de concorrência se o buffer for modificado
        ticks_to_process = list(self.tick_buffers[symbol])

        if periods and periods < len(ticks_to_process):
            ticks_to_process = ticks_to_process[-periods:]

        if not ticks_to_process:
            return 0.0

        total_pv = 0.0  # Soma de (preço * volume)
        total_volume = 0.0

        for tick in ticks_to_process:
            try:
                price = float(tick['mid'])
                # Estimar volume do tick como no _update_ohlc
                volume = (float(tick.get('bid_volume', 0.0)) + float(tick.get('ask_volume', 0.0))) / 2.0
                if volume == 0.0: volume = float(tick.get('volume', 1.0))


                if volume > 0: # Considerar apenas ticks com volume
                    total_pv += price * volume
                    total_volume += volume
            except (TypeError, ValueError) as e:
                logger.warning(f"Dado de tick inválido para cálculo de VWAP: {tick}, erro: {e}")
                continue


        if total_volume > 0:
            return total_pv / total_volume

        # Fallback se não houver volume, retornar o mid do último tick válido
        return float(ticks_to_process[-1]['mid']) if ticks_to_process and 'mid' in ticks_to_process[-1] else 0.0


    def calculate_market_microstructure(self, symbol: str, window_size: int = 1000) -> Dict[str, Any]: # Adicionado window_size
        """Calcula métricas de microestrutura de mercado dos ticks no buffer."""
        if symbol not in self.tick_buffers or len(self.tick_buffers[symbol]) < 20: # Mínimo de 20 ticks
            return {'realized_volatility': 0.0, 'average_spread': 0.0, 'tick_count': 0}

        ticks = list(self.tick_buffers[symbol])[-window_size:] # Usar janela configurável

        if len(ticks) < 2: # Precisa de pelo menos 2 para np.diff
             return {'realized_volatility': 0.0, 'average_spread': 0.0, 'tick_count': len(ticks)}

        # Extrair arrays numpy de forma segura
        spreads = np.array([float(t.get('spread', 0.0)) for t in ticks if t.get('spread') is not None])
        prices = np.array([float(t.get('mid', 0.0)) for t in ticks if t.get('mid') is not None and float(t.get('mid',0.0)) > 0]) # Filtrar preços > 0 para log


        # Volatilidade realizada (anualizada, assumindo ticks por segundo como aproximação)
        realized_vol = 0.0
        if len(prices) >= 2:
            log_returns = np.diff(np.log(prices))
            # Fator de anualização depende da frequência média dos ticks.
            # Se ticks são esporádicos, a anualização pode ser enganosa.
            # Usar uma métrica de volatilidade de curto prazo sem anualização pode ser mais robusto.
            # Ex: std dev dos últimos N retornos.
            realized_vol_short_term = np.std(log_returns) if len(log_returns) > 0 else 0.0
            # Para uma estimativa anualizada, se ticks fossem ~1 por segundo:
            # realized_vol = realized_vol_short_term * np.sqrt(252 * 24 * 60 * 60)
            realized_vol = realized_vol_short_term # Mantendo como volatilidade do período da janela


        # Métricas de Spread
        avg_spread = np.mean(spreads) if len(spreads) > 0 else 0.0
        spread_vol = np.std(spreads) if len(spreads) > 0 else 0.0

        # Price impact (simplificado - variação média do preço por tick)
        avg_price_impact = 0.0
        if len(prices) >=2:
            price_changes = np.abs(np.diff(prices))
            avg_price_impact = np.mean(price_changes) if len(price_changes) > 0 else 0.0


        # Eficiência de preço (autocorrelação de retornos de 1 tick)
        autocorr = 0.0
        if 'log_returns' in locals() and len(log_returns) > 1: # Checar se log_returns foi definido
            # np.corrcoef retorna uma matriz, pegar o elemento [0,1]
            # Adicionar verificação para std dev zero para evitar warning de runtime
            if np.std(log_returns[:-1]) > 1e-9 and np.std(log_returns[1:]) > 1e-9:
                 corr_matrix = np.corrcoef(log_returns[:-1], log_returns[1:])
                 if corr_matrix.shape == (2,2): # Garantir que a matriz é 2x2
                    autocorr = corr_matrix[0, 1]
            else:
                autocorr = 0.0 # Ou np.nan se preferir


        return {
            'realized_volatility': realized_vol,
            'average_spread': avg_spread,
            'spread_volatility': spread_vol,
            'average_price_impact': avg_price_impact,
            'price_efficiency_autocorr': autocorr, # Autocorrelação de 1 tick
            'tick_count_in_window': len(ticks)
        }


    def detect_liquidity_events(self, symbol: str, sweep_window: int = 10, sweep_velocity_pips_sec: float = 10.0,
                                absorption_vol_pct_threshold: float = 0.2, absorption_min_ticks: int = 20,
                                absorption_price_range_pips: float = 2.0) -> List[Dict[str, Any]]: # Adicionado tipos
        """Detecta eventos de liquidez (sweeps, absorção) dos ticks no buffer."""
        events = []
        if symbol not in self.tick_buffers or len(self.tick_buffers[symbol]) < 100: # Precisa de histórico
            return events

        ticks = list(self.tick_buffers[symbol])[-100:] # Analisar os últimos 100 ticks

        # Detectar sweep de liquidez (movimento rápido de preço)
        for i in range(sweep_window, len(ticks)):
            window_ticks = ticks[i - sweep_window : i]
            try:
                price_change_abs = abs(float(window_ticks[-1]['mid']) - float(window_ticks[0]['mid']))
                # Converter timestamps para datetime para cálculo de diferença
                ts_end = pd.Timestamp(window_ticks[-1]['timestamp'], tz='UTC')
                ts_start = pd.Timestamp(window_ticks[0]['timestamp'], tz='UTC')
                time_diff_seconds = (ts_end - ts_start).total_seconds()

                if time_diff_seconds > 1e-3: # Evitar divisão por zero e movimentos instantâneos
                    velocity_price_per_sec = price_change_abs / time_diff_seconds
                    velocity_pips_per_sec = velocity_price_per_sec * 10000 # Assumindo EURUSD (4 casas)

                    if velocity_pips_per_sec > sweep_velocity_pips_sec:
                        events.append({
                            'type': 'liquidity_sweep',
                            'timestamp': window_ticks[-1]['timestamp'], # Timestamp do fim do sweep
                            'velocity_pips_sec': round(velocity_pips_per_sec, 2),
                            'price_change_pips': round(price_change_abs * 10000, 2),
                            'duration_sec': round(time_diff_seconds, 3),
                            'direction': 'up' if window_ticks[-1]['mid'] > window_ticks[0]['mid'] else 'down'
                        })
            except (TypeError, ValueError, KeyError) as e:
                logger.warning(f"Dado de tick inválido para detecção de sweep: {e}")
                continue


        # Detectar absorção (grande volume sem movimento de preço)
        volume_profile_data = self.get_volume_profile(symbol) # Renomeado
        if volume_profile_data and 'profile' in volume_profile_data and volume_profile_data.get('total_volume', 0) > 0:
            profile = volume_profile_data['profile']
            total_vol = volume_profile_data['total_volume']

            for price_level_str, data in profile.items(): # price_level é string aqui
                price_level = float(price_level_str)
                if data['volume'] > total_vol * absorption_vol_pct_threshold:
                    # Verificar se preço não se moveu muito em torno deste nível de preço
                    # (ticks recentes cujo 'mid' está próximo de 'price_level')
                    price_range_abs = absorption_price_range_pips / 10000.0
                    ticks_at_level = [t for t in ticks if abs(float(t['mid']) - price_level) < price_range_abs]

                    if len(ticks_at_level) >= absorption_min_ticks:
                        events.append({
                            'type': 'absorption',
                            'timestamp': ticks_at_level[-1]['timestamp'], # Timestamp do último tick no nível
                            'price_level': price_level,
                            'volume_at_level': data['volume'],
                            'tick_count_at_level': len(ticks_at_level)
                        })
        return events


    def calculate_order_flow_imbalance(self, symbol: str, window: int = 50) -> float:
        """Calcula desequilíbrio de fluxo de ordens (OFI) dos ticks no buffer."""
        if symbol not in self.tick_buffers or len(self.tick_buffers[symbol]) < 2: # Precisa de pelo menos 2 ticks
            return 0.0

        ticks_to_analyze = list(self.tick_buffers[symbol])[-window:] # Renomeado
        if len(ticks_to_analyze) < 2:
            return 0.0

        buy_pressure_volume = 0.0 # Renomeado
        sell_pressure_volume = 0.0 # Renomeado

        for i in range(1, len(ticks_to_analyze)):
            current_tick = ticks_to_analyze[i] # Renomeado
            prev_tick = ticks_to_analyze[i-1]

            try:
                current_mid = float(current_tick['mid'])
                prev_mid = float(prev_tick['mid'])
                current_ask = float(current_tick['ask'])
                current_bid = float(current_tick['bid'])
                # Estimar volume do tick
                volume = (float(current_tick.get('bid_volume', 0.0)) + float(current_tick.get('ask_volume', 0.0))) / 2.0
                if volume == 0.0: volume = float(current_tick.get('volume', 1.0))

            except (TypeError, ValueError, KeyError) as e:
                logger.warning(f"Dado de tick inválido para cálculo de OFI: {e}")
                continue


            if volume > 0:
                if current_mid > prev_mid: # Uptick
                    buy_pressure_volume += volume
                elif current_mid < prev_mid: # Downtick
                    sell_pressure_volume += volume
                else: # Preço não mudou, usar regra do agressor no spread
                    if current_mid >= current_ask: # Agressor comprou no ask (ou acima)
                        buy_pressure_volume += volume
                    elif current_mid <= current_bid: # Agressor vendeu no bid (ou abaixo)
                        sell_pressure_volume += volume
                    # Se o trade foi dentro do spread, pode ser difícil classificar sem dados de tape


        total_flow_volume = buy_pressure_volume + sell_pressure_volume # Renomeado

        if total_flow_volume > 0:
            # Retorna valor entre -1 (pressão de venda) e 1 (pressão de compra)
            return (buy_pressure_volume - sell_pressure_volume) / total_flow_volume

        return 0.0


    def get_market_summary(self, symbol: str) -> Dict[str, Any]: # Adicionada tipagem
        """Retorna resumo completo do mercado para um símbolo."""
        summary: Dict[str, Any] = { # Adicionada tipagem
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(), # Usar UTC e ISO
            'tick_count_buffer': len(self.tick_buffers.get(symbol, [])),
            'last_price': None, 'bid': None, 'ask': None, 'spread': None,
            'vwap': 0.0,
            'poc': None, 'value_area_high': None, 'value_area_low': None,
            'realized_volatility': 0.0, 'average_spread': 0.0,
            'order_flow_imbalance': 0.0
        }

        if symbol in self.tick_buffers and self.tick_buffers[symbol]:
            last_tick = self.tick_buffers[symbol][-1] # É um dict
            try:
                summary['last_price'] = float(last_tick['mid'])
                summary['bid'] = float(last_tick['bid'])
                summary['ask'] = float(last_tick['ask'])
                summary['spread'] = float(last_tick['spread'])
            except (TypeError, ValueError, KeyError) as e:
                 logger.warning(f"Erro ao extrair dados do último tick para sumário de {symbol}: {e}")


        summary['vwap'] = self.calculate_vwap(symbol, periods=1000) # VWAP dos últimos 1000 ticks no buffer

        micro_structure = self.calculate_market_microstructure(symbol, window_size=200) # Janela menor para microestrutura
        summary.update(micro_structure) # Adiciona chaves de micro_structure ao sumário

        summary['order_flow_imbalance'] = self.calculate_order_flow_imbalance(symbol, window=100) # OFI da última janela

        volume_profile_data = self.get_volume_profile(symbol) # Renomeado
        if volume_profile_data:
            summary['poc'] = volume_profile_data.get('poc')
            summary['value_area_high'] = volume_profile_data.get('vah')
            summary['value_area_low'] = volume_profile_data.get('val')
            summary['profile_total_volume'] = volume_profile_data.get('total_volume')


        return summary


    def cleanup_old_data(self, hours_to_keep: int = 72): # Aumentado para 3 dias
        """Remove dados OHLC antigos da memória para liberar memória."""
        # Este método limpa apenas o OHLC em memória. Ticks são gerenciados por deque(maxlen).
        # Perfis de volume 'current' são sobrescritos; se houvesse perfis históricos, precisariam de limpeza.
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_to_keep) # Usar UTC

        cleaned_symbols_count = 0
        for symbol in list(self.ohlc_data.keys()): # Iterar sobre cópia das chaves
            for timeframe in list(self.ohlc_data[symbol].keys()):
                ohlc_bars = self.ohlc_data[symbol][timeframe]
                keys_to_delete = [
                    period_key for period_key, bar_data in ohlc_bars.items()
                    if bar_data.get('timestamp') and bar_data['timestamp'] < cutoff_time
                ]
                if keys_to_delete:
                    cleaned_symbols_count +=1
                    for key in keys_to_delete:
                        del ohlc_bars[key]

        if cleaned_symbols_count > 0:
            logger.info(f"Limpeza de dados OHLC em memória ({hours_to_keep}h) concluída para {cleaned_symbols_count} entradas de símbolo/timeframe.")
        else:
            logger.debug(f"Limpeza de dados OHLC em memória: Nenhum dado antigo encontrado para remover (limite: {hours_to_keep}h).")

    def reset_daily_structures(self, symbol: str):
        """Reseta estruturas que são por dia/sessão, como o perfil de volume 'current'."""
        if symbol in self.volume_profiles:
            self.volume_profiles[symbol]['current'] = {}
            logger.info(f"Perfil de volume 'current' resetado para {symbol}.")
        # Adicionar reset para outras estruturas se necessário (ex: estatísticas diárias de OFI)