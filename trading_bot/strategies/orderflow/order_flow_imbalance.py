# strategies/orderflow/order_flow_imbalance.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados se necessário
# import talib # Não usado diretamente, mas poderia ser para indicadores contextuais
from typing import Dict, Optional, Any, List, Deque, Tuple # Adicionado Tuple
from datetime import datetime, timedelta, timezone # Adicionado timezone
from collections import deque, defaultdict

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
# Importar TickData e DOMSnapshot se forem usados explicitamente nos tipos de market_context
from api.ticktrader_ws import TickData, DOMSnapshot
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default, etc.

logger = setup_logger("order_flow_imbalance_strategy_logger") # Nome do logger específico

class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Estratégia baseada em desequilíbrio de fluxo de ordens (Order Flow Imbalance - OFI).
    Analisa o fluxo de ordens agressivas (market orders) para detectar desequilíbrios
    significativos entre compra e venda, entrando na direção do fluxo dominante.
    Requer dados de alta qualidade (idealmente Time & Sales e DOM L2/L3).
    """

    def __init__(self):
        super().__init__("OrderFlowImbalancePressure") # Nome da estratégia
        self.suitable_regimes = [MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY, MarketRegime.RANGE] # OFI pode dar sinais em vários regimes
        self.min_time_between_signals_sec = 120  # 2 minutos

        # Buffers para análise de fluxo (precisam ser persistentes entre chamadas de calculate_indicators)
        # O tamanho máximo deve ser suficiente para o lookback mais longo.
        # Maxlen para trade_flow_buffer pode ser maior, ex: para cobrir várias janelas de lookback_seconds.
        self.trade_flow_buffer: deque[Dict[str, Any]] = deque(maxlen=5000) # Aumentado, guarda trades estimados
        self.dom_snapshots_buffer: deque[Dict[str, Any]] = deque(maxlen=100) # Guarda snapshots do DOM # Renomeado
        # Perfil de volume da sessão/dia. Precisa de lógica de reset.
        self.current_session_volume_profile: Dict[float, Dict[str, float]] = defaultdict(lambda: {'buy_vol': 0.0, 'sell_vol': 0.0, 'total_vol': 0.0}) # Renomeado
        self.cumulative_delta_session: float = 0.0 # Delta cumulativo da sessão/dia # Renomeado
        self.last_profile_reset_time: Optional[datetime] = None


    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Parâmetros de Fluxo de Ordens e Imbalance
            'ofi_imbalance_threshold_ratio': 0.65,  # Ex: 65% de desequilíbrio (ratio de buy_vol/total_vol) # Renomeado
            'min_total_volume_for_signal_usd': 500000, # Volume total mínimo (em USD ou lotes) na janela para considerar OFI # Renomeado
            'ofi_lookback_seconds': 60,           # Janela de análise para OFI
            'cumulative_delta_abs_threshold_usd': 1000000, # Delta cumulativo mínimo (absoluto) para confirmação # Renomeado

            # Análise do DOM (Depth of Market)
            'dom_analysis_levels': 10,                 # Níveis do DOM para analisar
            'dom_absorption_ratio_threshold': 2.5,     # Ratio para detectar absorção (ex: vol no nível / delta recente) # Renomeado
            'dom_sweep_min_ticks_cleared': 3,      # Mínimo de níveis varridos para ser um sweep
            'dom_wall_min_size_usd': 200000,       # Tamanho mínimo para considerar um "wall" no DOM

            # Confirmações e Filtros Adicionais
            'use_vwap_filter_ofi': True, # Renomeado
            'vwap_max_deviation_pips_ofi': 5.0, # Máximo desvio do VWAP em pips # Renomeado
            'momentum_flow_confirmation_required': True, # Renomeado
            'min_ticks_in_lookback_window': 30, # Mínimo de ticks (trades estimados) na janela de lookback # Renomeado

            # Gestão de Risco para OFI
            'atr_period_ofi_stops': 14,
            'atr_multiplier_sl_ofi': 1.5,
            'fixed_tp_pips_ofi': 10.0,
            'use_dom_levels_for_stops': True, # Usar níveis de suporte/resistência do DOM para stops

            # Filtros Gerais de Execução
            'min_spread_pips_ofi_entry': 0.2, # Renomeado
            'max_spread_pips_ofi_entry': 1.5, # Renomeado
            # 'avoid_news_ofi': True, # Pode ser herdado de BaseStrategy
            # 'min_liquidity_score_ofi': 0.7 # Se tivermos um score de liquidez
            'reset_profile_delta_period': "session", # "session", "hour", ou None
        }

    def _reset_session_dependent_state(self, current_event_time: datetime):
        """Reseta estado que depende da sessão (perfil de volume, delta cumulativo)."""
        self.current_session_volume_profile.clear()
        self.cumulative_delta_session = 0.0
        self.last_profile_reset_time = current_event_time
        self.logger.info(f"Estado de sessão da OrderFlowImbalanceStrategy resetado às {current_event_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")


    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula métricas de fluxo de ordens, imbalance, DOM, VWAP, etc."""
        self.current_indicators = {} # Resetar

        current_tick_obj: Optional[TickData] = market_context.get('tick')
        current_dom_snapshot: Optional[DOMSnapshot] = market_context.get('dom')

        if not current_tick_obj or not hasattr(current_tick_obj, 'timestamp') or not hasattr(current_tick_obj, 'mid'):
            self.logger.debug("Tick atual inválido ou ausente para OrderFlowImbalance.")
            return
        
        current_event_time_utc = current_tick_obj.timestamp

        # Resetar perfil de volume e delta no início de nova sessão/hora
        reset_period_cfg = self.parameters['reset_profile_delta_period']
        if reset_period_cfg:
            if self.last_profile_reset_time is None: # Primeira vez
                self._reset_session_dependent_state(current_event_time_utc)
            else:
                if reset_period_cfg == "hour" and current_event_time_utc.hour != self.last_profile_reset_time.hour:
                    self._reset_session_dependent_state(current_event_time_utc)
                elif reset_period_cfg == "session":
                    # Lógica para detectar mudança de sessão (ex: Asia->London)
                    current_session_name = self._get_trading_session_name(current_event_time_utc.hour) # Helper
                    last_reset_session_name = self._get_trading_session_name(self.last_profile_reset_time.hour)
                    if current_session_name != last_reset_session_name:
                        self._reset_session_dependent_state(current_event_time_utc)


        # 1. Atualizar buffers de fluxo (trade_flow_buffer, DOM, perfil de volume, delta)
        self._update_internal_flow_buffers(current_tick_obj, current_dom_snapshot)

        # 2. Filtrar trades recentes para a janela de lookback do OFI
        cutoff_timestamp = current_event_time_utc - timedelta(seconds=self.parameters['ofi_lookback_seconds'])
        recent_trades_in_window = [t for t in self.trade_flow_buffer if t['timestamp_dt'] >= cutoff_timestamp] # Renomeado

        if len(recent_trades_in_window) < self.parameters['min_ticks_in_lookback_window']:
            self.logger.debug(f"Trades insuficientes na janela ({len(recent_trades_in_window)}) para análise OFI.")
            return # Não há dados suficientes na janela

        # 3. Calcular Volume Direcional e OFI na janela
        buy_volume_window = sum(t['volume_usd'] for t in recent_trades_in_window if t['aggressor_side'] == 'buy') # Renomeado
        sell_volume_window = sum(t['volume_usd'] for t in recent_trades_in_window if t['aggressor_side'] == 'sell') # Renomeado
        total_volume_window_usd = buy_volume_window + sell_volume_window # Renomeado

        if total_volume_window_usd < self.parameters['min_total_volume_for_signal_usd']:
            self.logger.debug(f"Volume total na janela ({total_volume_window_usd:.0f} USD) abaixo do limiar.")
            return # Volume muito baixo

        # Order Flow Imbalance (OFI) - ratio do fluxo dominante
        # Ex: buy_ofi_ratio = buy_volume_window / total_volume_window_usd
        #     sell_ofi_ratio = sell_volume_window / total_volume_window_usd
        # Ou como delta normalizado: ofi_delta_norm = (buy_volume_window - sell_volume_window) / total_volume_window_usd
        buy_ofi_ratio = buy_volume_window / total_volume_window_usd if total_volume_window_usd > 0 else 0.5
        sell_ofi_ratio = sell_volume_window / total_volume_window_usd if total_volume_window_usd > 0 else 0.5
        ofi_delta_normalized = (buy_volume_window - sell_volume_window) / (total_volume_window_usd + 1e-9) # Evitar div por zero


        # 4. Análise do DOM (se disponível)
        dom_metrics_dict = self._analyze_current_dom(current_dom_snapshot) if current_dom_snapshot else {} # Renomeado

        # 5. VWAP dos trades recentes na janela
        vwap_window = self._calculate_vwap_for_trades(recent_trades_in_window) # Renomeado

        # 6. Detectar eventos especiais (absorção, sweep) - podem usar dados do DOM e trades
        absorption_event = self._detect_market_absorption(recent_trades_in_window, dom_metrics_dict) if self.parameters.get('enable_dom_absorption_check',True) else None # Renomeado
        sweep_event = self._detect_liquidity_sweep(recent_trades_in_window, dom_metrics_dict) # Renomeado

        # 7. Ponto de Controle (POC) do perfil de volume da sessão
        poc_price_session = self._get_current_poc_from_profile() # Renomeado

        # 8. Momentum do Fluxo de Ordens (mudança no OFI)
        flow_momentum_val = self._calculate_order_flow_momentum(ofi_delta_normalized) # Renomeado

        pip_size_ofi = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01 # Renomeado

        self.current_indicators = {
            'buy_ofi_ratio_window': buy_ofi_ratio,
            'sell_ofi_ratio_window': sell_ofi_ratio,
            'ofi_delta_normalized_window': ofi_delta_normalized,
            'total_volume_window_usd': total_volume_window_usd,
            'cumulative_delta_session_usd': self.cumulative_delta_session,
            'dom_analysis': dom_metrics_dict,
            'vwap_ofi_window': vwap_window,
            'current_price_mid': current_tick_obj.mid,
            'current_spread_pips': current_tick_obj.spread / pip_size_ofi,
            'session_poc_price': poc_price_session,
            'order_flow_momentum': flow_momentum_val,
            'absorption_event_detected': absorption_event,
            'sweep_event_detected': sweep_event,
            'trade_count_in_window': len(recent_trades_in_window)
        }
        # Guardar o OFI atual para cálculo de momentum no próximo tick
        self.internal_state['last_ofi_delta_normalized'] = ofi_delta_normalized


    def _update_internal_flow_buffers(self, tick_obj: TickData, dom_snapshot: Optional[DOMSnapshot]): # Renomeado
        """Atualiza buffers internos: trade_flow_buffer, DOM snapshots, perfil de volume da sessão, delta cumulativo."""
        # Estimar lado agressor e volume do trade (MUITO SIMPLIFICADO)
        # Idealmente, viria de dados de Time & Sales.
        aggressor_side_est: Optional[str] = None # Renomeado
        trade_volume_est_units = (tick_obj.bid_volume + tick_obj.ask_volume) / 2.0 # Em unidades (ex: EUR)
        if trade_volume_est_units == 0: trade_volume_est_units = 1.0 # Mínimo para contagem

        # Lógica de Tick Rule para estimar agressor
        last_trade_price_in_buffer = self.trade_flow_buffer[-1]['price_mid'] if self.trade_flow_buffer else tick_obj.mid
        
        if tick_obj.mid > last_trade_price_in_buffer: aggressor_side_est = 'buy'
        elif tick_obj.mid < last_trade_price_in_buffer: aggressor_side_est = 'sell'
        else: # Preço não mudou, usar proximidade ao bid/ask
            if abs(tick_obj.mid - tick_obj.ask) < abs(tick_obj.mid - tick_obj.bid): aggressor_side_est = 'buy' # Mais perto do Ask
            else: aggressor_side_est = 'sell' # Mais perto do Bid ou igual

        if aggressor_side_est: # Apenas adicionar se pudermos estimar o lado
            trade_event = {
                'timestamp_dt': tick_obj.timestamp, # Já é datetime UTC
                'price_mid': tick_obj.mid,
                'aggressor_side': aggressor_side_est,
                'volume_units': trade_volume_est_units, # Volume em unidades da moeda base
                'volume_usd': trade_volume_est_units * tick_obj.mid, # Volume em USD (aproximado para EURUSD)
                'bid_at_trade': tick_obj.bid,
                'ask_at_trade': tick_obj.ask
            }
            self.trade_flow_buffer.append(trade_event)

            # Atualizar Delta Cumulativo da Sessão
            delta_change = trade_event['volume_usd'] if aggressor_side_est == 'buy' else -trade_event['volume_usd']
            self.cumulative_delta_session += delta_change

            # Atualizar Perfil de Volume da Sessão
            price_level_vp = round(tick_obj.mid, 5) # Arredondar para nível de preço (5 casas para EURUSD) # Renomeado
            self.current_session_volume_profile[price_level_vp]['total_vol'] += trade_event['volume_usd']
            if aggressor_side_est == 'buy':
                self.current_session_volume_profile[price_level_vp]['buy_vol'] += trade_event['volume_usd']
            else:
                self.current_session_volume_profile[price_level_vp]['sell_vol'] += trade_event['volume_usd']


        # Armazenar Snapshot do DOM (se disponível)
        if dom_snapshot and dom_snapshot.symbol: # Checar se dom_snapshot e symbol existem
            self.dom_snapshots_buffer.append({
                'timestamp_dt': dom_snapshot.timestamp,
                'dom_data': dom_snapshot # Armazenar o objeto DOMSnapshot
            })


    def _analyze_current_dom(self, dom_snapshot_obj: DOMSnapshot) -> Dict[str, Any]: # Renomeado
        """Analisa o snapshot atual do DOM para métricas relevantes."""
        if not dom_snapshot_obj: return {}

        dom_depth_data = dom_snapshot_obj.get_depth(self.parameters['dom_analysis_levels']) # Renomeado
        bid_vol_dom = dom_depth_data['bid_volume'] # Renomeado
        ask_vol_dom = dom_depth_data['ask_volume'] # Renomeado
        total_depth_vol = bid_vol_dom + ask_vol_dom # Renomeado

        dom_imbalance_ratio = 0.0
        if total_depth_vol > 0:
            # Ratio Bid/Ask (ex: >1 significa mais liquidez no Bid)
            # Ou (Bid - Ask) / Total
            dom_imbalance_ratio = (bid_vol_dom - ask_vol_dom) / total_depth_vol


        # Detectar "walls" (grandes ordens limitadas)
        # bid_wall_info = max(dom_depth_data['bids'], key=lambda x: x[1], default=(0,0)) # (price, size)
        # ask_wall_info = max(dom_depth_data['asks'], key=lambda x: x[1], default=(0,0))
        bid_wall_info = (0.0, 0.0) # Default
        if dom_depth_data['bids']: bid_wall_info = max(dom_depth_data['bids'], key=lambda x_item: x_item[1])

        ask_wall_info = (0.0, 0.0) # Default
        if dom_depth_data['asks']: ask_wall_info = max(dom_depth_data['asks'], key=lambda x_item: x_item[1])


        # Pressão de compra/venda nos primeiros N níveis (ex: 3 níveis)
        near_bid_vol = sum(vol for _, vol in dom_depth_data['bids'][:3]) # Renomeado
        near_ask_vol = sum(vol for _, vol in dom_depth_data['asks'][:3]) # Renomeado
        total_near_vol = near_bid_vol + near_ask_vol # Renomeado
        near_dom_pressure_ratio = (near_bid_vol - near_ask_vol) / (total_near_vol + 1e-9) if total_near_vol > 0 else 0.0 # Renomeado

        return {
            'dom_imbalance_ratio_levels_N': dom_imbalance_ratio, # Renomeado
            'total_dom_depth_volume_usd': total_depth_vol, # Assumindo que volume no DOM é em USD (ou precisa converter)
            'top_bid_volume_usd': bid_vol_dom,
            'top_ask_volume_usd': ask_vol_dom,
            'bid_wall_price': bid_wall_info[0],
            'bid_wall_size_usd': bid_wall_info[1],
            'ask_wall_price': ask_wall_info[0],
            'ask_wall_size_usd': ask_wall_info[1],
            'near_dom_pressure_ratio_3levels': near_dom_pressure_ratio # Renomeado
        }

    def _detect_market_absorption(self, recent_trades: List[Dict[str, Any]], # Renomeado
                                 dom_analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detecta absorção de ordens (grande volume sem movimento de preço, contra um nível de DOM)."""
        if not recent_trades or len(recent_trades) < 10 or not dom_analysis_results: # Precisa de alguns trades e DOM
            return None

        # Analisar os últimos N trades (ex: últimos 20)
        last_n_trades_for_abs = recent_trades[-20:] # Renomeado
        if not last_n_trades_for_abs: return None

        price_range_abs = max(t['price_mid'] for t in last_n_trades_for_abs) - min(t['price_mid'] for t in last_n_trades_for_abs)
        volume_in_range_usd = sum(t['volume_usd'] for t in last_n_trades_for_abs) # Renomeado
        pip_sz_abs = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01 # Renomeado

        # Condição de absorção: Range de preço pequeno (ex: < 2 pips) com volume alto
        if price_range_abs < (2.0 * pip_sz_abs) and volume_in_range_usd > self.parameters['min_total_volume_for_signal_usd'] * 0.5: # Metade do volume de sinal
            # Verificar se há um "wall" no DOM sendo consumido
            # Ex: Se houve muito volume de compra mas o preço não subiu e há um ask_wall
            current_price = last_n_trades_for_abs[-1]['price_mid']
            delta_in_range = sum(t['volume_usd'] if t['aggressor_side'] == 'buy' else -t['volume_usd'] for t in last_n_trades_for_abs)

            if delta_in_range > 0 and dom_analysis_results.get('ask_wall_size_usd',0) > volume_in_range_usd * 0.2 and \
               abs(current_price - dom_analysis_results.get('ask_wall_price',0)) < (3 * pip_sz_abs) : # Perto do Ask Wall
                return {'type': 'buy_absorption_at_ask_wall', 'price': dom_analysis_results.get('ask_wall_price'), 'volume_absorbed_usd': delta_in_range}
            
            elif delta_in_range < 0 and dom_analysis_results.get('bid_wall_size_usd',0) > volume_in_range_usd * 0.2 and \
                 abs(current_price - dom_analysis_results.get('bid_wall_price',0)) < (3 * pip_sz_abs) : # Perto do Bid Wall
                return {'type': 'sell_absorption_at_bid_wall', 'price': dom_analysis_results.get('bid_wall_price'), 'volume_absorbed_usd': abs(delta_in_range)}
        return None


    def _detect_liquidity_sweep(self, recent_trades: List[Dict[str, Any]], # Renomeado
                               dom_analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detecta sweep de liquidez (movimento rápido através de níveis do DOM)."""
        if not recent_trades or len(recent_trades) < 5 or not dom_analysis_results:
            return None
        
        # Analisar os últimos N trades (ex: últimos 5-10)
        last_n_trades_for_sweep = recent_trades[-10:] # Renomeado
        if not last_n_trades_for_sweep or len(last_n_trades_for_sweep) < 2: return None


        time_span_seconds = (last_n_trades_for_sweep[-1]['timestamp_dt'] - last_n_trades_for_sweep[0]['timestamp_dt']).total_seconds()
        if time_span_seconds < 1e-3 : return None # Muito rápido ou timestamps iguais

        price_change_sweep_abs = abs(last_n_trades_for_sweep[-1]['price_mid'] - last_n_trades_for_sweep[0]['price_mid']) # Renomeado
        pip_sz_sweep_det = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01 # Renomeado
        velocity_pips_s_sweep = (price_change_sweep_abs / pip_sz_sweep_det) / time_span_seconds if time_span_seconds > 0 else 0 # Renomeado

        # Condição de Sweep: movimento rápido com volume
        volume_burst_usd_sweep = sum(t['volume_usd'] for t in last_n_trades_for_sweep) # Renomeado
        # Usar um threshold de velocidade diferente, ex: vindo de parâmetros da estratégia StopHunt
        sweep_min_vel_param = self.parameters.get('sweep_min_velocity_pips_per_sec_ofi', 10.0) # Default diferente

        if velocity_pips_s_sweep > sweep_min_vel_param and \
           volume_burst_usd_sweep > self.parameters['min_total_volume_for_signal_usd'] * 0.3: # Volume considerável
            
            # Checar se níveis do DOM foram varridos (simplificado)
            # Contar quantos níveis de preço distintos foram negociados na direção do sweep
            direction_of_sweep = 'up' if last_n_trades_for_sweep[-1]['price_mid'] > last_n_trades_for_sweep[0]['price_mid'] else 'down'
            prices_in_sweep = sorted(list(set(t['price_mid'] for t in last_n_trades_for_sweep)))
            levels_cleared = 0
            if direction_of_sweep == 'up':
                # Contar quantos preços distintos acima do preço inicial foram atingidos
                levels_cleared = sum(1 for p in prices_in_sweep if p > last_n_trades_for_sweep[0]['price_mid'])
            else:
                levels_cleared = sum(1 for p in prices_in_sweep if p < last_n_trades_for_sweep[0]['price_mid'])

            if levels_cleared >= self.parameters['dom_sweep_min_ticks_cleared']:
                return {'type': f'{direction_of_sweep}_sweep', 'velocity_pips_s': velocity_pips_s_sweep, 'levels_cleared': levels_cleared, 'volume_usd': volume_burst_usd_sweep}
        return None

    def _calculate_vwap_for_trades(self, trades_list_vwap: List[Dict[str, Any]]) -> float: # Renomeado
        """Calcula VWAP para uma lista de trades (dicts com 'price_mid' e 'volume_usd')."""
        if not trades_list_vwap: return 0.0
        
        total_pv_vwap = sum(t['price_mid'] * t['volume_usd'] for t in trades_list_vwap) # Renomeado
        total_vol_vwap = sum(t['volume_usd'] for t in trades_list_vwap) # Renomeado
        if total_vol_vwap == 0:
            return trades_list_vwap[-1]['price_mid'] if trades_list_vwap else 0.0
        return total_pv_vwap / total_vol_vwap


    def _get_current_poc_from_profile(self) -> float: # Renomeado
        """Encontra o Ponto de Controle (preço com maior volume) do perfil de volume da sessão atual."""
        if not self.current_session_volume_profile: return 0.0
        
        poc_price_val = 0.0 # Renomeado
        max_volume_at_poc = 0.0 # Renomeado

        for price_vp, volumes_data in self.current_session_volume_profile.items(): # Renomeado
            if volumes_data['total_vol'] > max_volume_at_poc:
                max_volume_at_poc = volumes_data['total_vol']
                poc_price_val = price_vp # price_vp já é float
        return poc_price_val


    def _calculate_order_flow_momentum(self, current_ofi_delta_norm: float) -> float: # Renomeado
        """Calcula o momentum do OFI (mudança no OFI delta normalizado)."""
        # Requer guardar o OFI do período anterior no internal_state
        last_ofi = self.internal_state.get('last_ofi_delta_normalized', 0.0)
        momentum = current_ofi_delta_norm - last_ofi
        # Opcional: normalizar o momentum ou usar uma média móvel dele.
        return momentum


    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de entrada se um desequilíbrio de fluxo de ordens significativo for detectado."""
        indic = self.current_indicators
        if not indic or 'buy_ofi_ratio_window' not in indic: # Checar se indicadores OFI existem
            return None

        # Filtro de Spread
        max_spread = self.parameters['max_spread_pips_ofi_entry']
        if indic.get('current_spread_pips', float('inf')) > max_spread:
            self.logger.debug(f"Spread ({indic.get('current_spread_pips'):.1f} pips) muito alto para OFI (Max: {max_spread}).")
            return None

        # Checar se o volume total na janela de OFI é suficiente (já feito em calculate_indicators, mas bom checar de novo)
        if indic.get('total_volume_window_usd', 0.0) < self.parameters['min_total_volume_for_signal_usd']:
            return None


        signal_side_to_gen_ofi: Optional[str] = None # Renomeado
        signal_strength_ofi = 0.0 # Renomeado (0.0 a 1.0+)

        # Sinal de COMPRA (Buy)
        if indic['buy_ofi_ratio_window'] >= self.parameters['ofi_imbalance_threshold_ratio']:
            signal_side_to_gen_ofi = 'buy'
            signal_strength_ofi = indic['buy_ofi_ratio_window'] # Força baseada no próprio ratio
            # Confirmações adicionais podem aumentar a força
            if self.parameters['momentum_flow_confirmation_required'] and indic.get('order_flow_momentum', 0.0) > 0.05: # Momentum positivo
                signal_strength_ofi += 0.15
            if indic.get('dom_analysis', {}).get('dom_imbalance_ratio_levels_N', 0.0) > 0.2: # DOM inclinado para compra
                signal_strength_ofi += 0.10
            if indic.get('cumulative_delta_session_usd', 0.0) > self.parameters['cumulative_delta_abs_threshold_usd']:
                signal_strength_ofi += 0.10
            if indic.get('sweep_event_detected') and indic['sweep_event_detected']['type'] == 'up_sweep':
                signal_strength_ofi += 0.20 # Sweep de alta confirma
            if indic.get('absorption_event_detected') and 'buy_absorption' in indic['absorption_event_detected']['type']:
                 signal_strength_ofi += 0.15 # Absorção de venda no bid wall confirma compra

        # Sinal de VENDA (Sell)
        elif indic['sell_ofi_ratio_window'] >= self.parameters['ofi_imbalance_threshold_ratio']:
            signal_side_to_gen_ofi = 'sell'
            signal_strength_ofi = indic['sell_ofi_ratio_window']
            if self.parameters['momentum_flow_confirmation_required'] and indic.get('order_flow_momentum', 0.0) < -0.05: # Momentum negativo
                signal_strength_ofi += 0.15
            if indic.get('dom_analysis', {}).get('dom_imbalance_ratio_levels_N', 0.0) < -0.2: # DOM inclinado para venda
                signal_strength_ofi += 0.10
            if indic.get('cumulative_delta_session_usd', 0.0) < -self.parameters['cumulative_delta_abs_threshold_usd']:
                signal_strength_ofi += 0.10
            if indic.get('sweep_event_detected') and indic['sweep_event_detected']['type'] == 'down_sweep':
                signal_strength_ofi += 0.20
            if indic.get('absorption_event_detected') and 'sell_absorption' in indic['absorption_event_detected']['type']:
                 signal_strength_ofi += 0.15


        if signal_side_to_gen_ofi:
            # Filtro VWAP
            if self.parameters['use_vwap_filter_ofi'] and indic.get('vwap_ofi_window', 0.0) > 0:
                pip_size_vwap = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01
                vwap_dist_pips = abs(indic['current_price_mid'] - indic['vwap_ofi_window']) / pip_size_vwap
                if vwap_dist_pips > self.parameters['vwap_max_deviation_pips_ofi']:
                    self.logger.debug(f"Sinal OFI {signal_side_to_gen_ofi} filtrado por VWAP: Dist {vwap_dist_pips:.1f} pips > Limite {self.parameters['vwap_max_deviation_pips_ofi']:.1f} pips.")
                    signal_strength_ofi *= 0.5 # Reduzir força se longe do VWAP, ou descartar

            # Limiar mínimo de força para gerar o sinal
            if signal_strength_ofi >= self.parameters.get('min_ofi_signal_strength_threshold', 0.70): # Ex: 0.70
                return self._create_order_flow_trade_signal(signal_side_to_gen_ofi, indic, signal_strength_ofi, market_context) # Renomeado
        return None


    def _create_order_flow_trade_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                               signal_strength_value: float, market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a estratégia OFI."""
        current_price_create_ofi = indicators_dict['current_price_mid'] # Renomeado
        # ATR deve vir do market_context ou ser calculado por BaseStrategy e estar em indicators_dict
        # Assumindo que BaseStrategy._get_prices_from_context e calculate_atr foram chamados.
        # Se não, precisamos recalcular ATR aqui ou usar um default.
        # Para este exemplo, vamos pegar de indicators_dict se existir.
        atr_pips_create_ofi = self.current_indicators.get('atr_pips') # Se BaseStrategy o calcula
        if not atr_pips_create_ofi : # Fallback se não calculado pela BaseStrategy
            high_prices = self._get_prices_from_context(market_context, 'high', lookback=self.parameters['atr_period_ofi_stops']+5)
            low_prices = self._get_prices_from_context(market_context, 'low', lookback=self.parameters['atr_period_ofi_stops']+5)
            close_prices = self._get_prices_from_context(market_context, 'mid', lookback=self.parameters['atr_period_ofi_stops']+5)
            if len(high_prices) >= self.parameters['atr_period_ofi_stops']:
                 atr_val_price = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.parameters['atr_period_ofi_stops'])[-1]
                 pip_size_temp = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01
                 atr_pips_create_ofi = atr_val_price / pip_size_temp if atr_val_price > 0 else 10.0
            else:
                 atr_pips_create_ofi = 10.0 # Default ATR
        if atr_pips_create_ofi == 0.0: atr_pips_create_ofi = 10.0

        pip_size_create_ofi = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado


        sl_atr_mult_ofi = self.parameters['atr_multiplier_sl_ofi'] # Renomeado
        fixed_tp_pips_val = self.parameters['fixed_tp_pips_ofi'] # Renomeado

        stop_loss_val_ofi: float # Adicionada tipagem
        take_profit_val_ofi: float # Adicionada tipagem

        # Calcular Stops e TPs
        if self.parameters['use_dom_levels_for_stops'] and indicators_dict.get('dom_analysis'):
            dom_info = indicators_dict['dom_analysis'] # Renomeado
            if signal_side == 'buy':
                # SL abaixo de um bid wall ou nível de suporte do DOM
                sl_candidate_dom = dom_info.get('bid_wall_price', current_price_create_ofi) - (2 * pip_size_create_ofi) # 2 pips abaixo
                sl_atr_based_ofi = current_price_create_ofi - (atr_pips_create_ofi * sl_atr_mult_ofi * pip_size_create_ofi)
                stop_loss_val_ofi = min(sl_candidate_dom, sl_atr_based_ofi) if dom_info.get('bid_wall_price',0)>0 else sl_atr_based_ofi
                take_profit_val_ofi = current_price_create_ofi + (fixed_tp_pips_val * pip_size_create_ofi)
            else: # sell
                sl_candidate_dom = dom_info.get('ask_wall_price', current_price_create_ofi) + (2 * pip_size_create_ofi)
                sl_atr_based_ofi = current_price_create_ofi + (atr_pips_create_ofi * sl_atr_mult_ofi * pip_size_create_ofi)
                stop_loss_val_ofi = max(sl_candidate_dom, sl_atr_based_ofi) if dom_info.get('ask_wall_price',0)>0 else sl_atr_based_ofi
                take_profit_val_ofi = current_price_create_ofi - (fixed_tp_pips_val * pip_size_create_ofi)
        else: # Stops padrão baseados em ATR
            sl_distance_pips = atr_pips_create_ofi * sl_atr_mult_ofi
            if signal_side == 'buy':
                stop_loss_val_ofi = current_price_create_ofi - (sl_distance_pips * pip_size_create_ofi)
                take_profit_val_ofi = current_price_create_ofi + (fixed_tp_pips_val * pip_size_create_ofi)
            else: # sell
                stop_loss_val_ofi = current_price_create_ofi + (sl_distance_pips * pip_size_create_ofi)
                take_profit_val_ofi = current_price_create_ofi - (fixed_tp_pips_val * pip_size_create_ofi)

        # Confiança (0.5 a 1.0), signal_strength_value já está numa escala de ~0.65 a 1.0+
        confidence_final_ofi = round(np.clip(0.5 + (signal_strength_value - 0.65) * 0.8, 0.5, 0.95) , 4) # Escalar para range 0.5-0.95

        reason_str = f"OrderFlow Imbalance {signal_side.upper()}. OFI Ratio: {indicators_dict.get('buy_ofi_ratio_window' if signal_side=='buy' else 'sell_ofi_ratio_window', 0.0):.2f}." # Renomeado
        if indicators_dict.get('sweep_event_detected'): reason_str += " SWEEP detectado."
        if indicators_dict.get('absorption_event_detected'): reason_str += " ABSORÇÃO detectada."

        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=confidence_final_ofi,
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val_ofi, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val_ofi, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=reason_str,
            metadata={
                'ofi_delta_norm_window': indicators_dict.get('ofi_delta_normalized_window'),
                'total_volume_window_usd': indicators_dict.get('total_volume_window_usd'),
                'cumulative_delta_session_usd': indicators_dict.get('cumulative_delta_session_usd'),
                'dom_imbalance_at_signal': indicators_dict.get('dom_analysis',{}).get('dom_imbalance_ratio_levels_N'),
                'signal_strength_raw': signal_strength_value,
                'absorption_info': indicators_dict.get('absorption_event_detected'),
                'sweep_info': indicators_dict.get('sweep_event_detected')
            }
        )


    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Avalia condições de saída para a estratégia OFI."""
        indic = self.current_indicators
        if not indic or 'buy_ofi_ratio_window' not in indic: # Checar se OFI foi calculado
            return None

        # 1. Sair se o fluxo de ordens reverter fortemente contra a posição
        ofi_reversal_threshold = self.parameters['ofi_imbalance_threshold_ratio'] * 0.9 # Ex: 90% do limiar de entrada
        
        if open_position.side.lower() == 'buy':
            # Sair se fluxo virar vendedor forte
            if indic.get('sell_ofi_ratio_window', 0.0) >= ofi_reversal_threshold:
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Fluxo de ordens reverteu para venda.")
            # Sair se delta cumulativo da sessão virar fortemente negativo
            if indic.get('cumulative_delta_session_usd', 0.0) < -self.parameters['cumulative_delta_abs_threshold_usd'] * 0.75: # 75% do limiar
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Delta cumulativo da sessão tornou-se fortemente negativo.")
        
        elif open_position.side.lower() == 'sell':
            # Sair se fluxo virar comprador forte
            if indic.get('buy_ofi_ratio_window', 0.0) >= ofi_reversal_threshold:
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Fluxo de ordens reverteu para compra.")
            # Sair se delta cumulativo da sessão virar fortemente positivo
            if indic.get('cumulative_delta_session_usd', 0.0) > self.parameters['cumulative_delta_abs_threshold_usd'] * 0.75:
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Delta cumulativo da sessão tornou-se fortemente positivo.")

        # 2. Sair se detectar absorção significativa contra a posição
        absorption_info_exit = indic.get('absorption_event_detected') # Renomeado
        if absorption_info_exit:
            if open_position.side.lower() == 'buy' and 'sell_absorption' in absorption_info_exit.get('type',''): # Absorção de compra no ask wall
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Absorção de compra detectada contra posição (no Ask Wall).")
            elif open_position.side.lower() == 'sell' and 'buy_absorption' in absorption_info_exit.get('type',''): # Absorção de venda no bid wall
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Absorção de venda detectada contra posição (no Bid Wall).")

        # 3. (Opcional) Saída por tempo ou se o PnL atingir um múltiplo de ATR rapidamente
        # max_hold_ofi_seconds = self.parameters.get('max_ofi_trade_duration_seconds', 2 * 3600) # Ex: 2 horas
        # time_held_ofi = (datetime.now(timezone.utc) - open_position.open_time).total_seconds()
        # if time_held_ofi > max_hold_ofi_seconds:
        #     return ExitSignal(position_id_to_close=open_position.id, reason="Saída OFI: Tempo máximo de holding atingido.")


        return None