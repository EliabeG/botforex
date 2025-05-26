# strategies/liquidity_hunt/stop_hunt_strategy.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados se necessário
import talib # Para ATR
from typing import Dict, Optional, Any, List, Set, Tuple # Adicionado Tuple e Set
from datetime import datetime, timedelta, timezone # Adicionado timezone
from collections import defaultdict, deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime # MarketRegime é importado
# Importar TickData e DOMSnapshot se forem usados explicitamente nos tipos de market_context
from api.ticktrader_ws import TickData, DOMSnapshot
from utils.logger import setup_logger
from config.settings import CONFIG # Para configurações globais como SYMBOL

logger = setup_logger("stop_hunt_strategy_logger") # Nome do logger mais específico

class StopHuntStrategy(BaseStrategy):
    """
    Estratégia que tenta detectar e operar "stop hunts" ou varreduras de liquidez (liquidity sweeps).
    Identifica níveis prováveis de concentração de ordens stop, detecta movimentos rápidos
    através desses níveis (sweeps) e tenta entrar na reversão subsequente.
    """

    def __init__(self):
        super().__init__("LiquiditySweepReversal") # Nome mais descritivo
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.TREND] # Pode ocorrer em ambos
        self.min_time_between_signals_sec = 300  # 5 minutos (herdado e pode ser ajustado)

        # Rastreamento de níveis e eventos
        self.identified_liquidity_levels: Dict[float, Dict[str, Any]] = defaultdict(lambda: { # Renomeado e tipado
            'type': 'unknown', 'touches': 0, 'strength': 0.0,
            'created_at': datetime.now(timezone.utc), 'last_touch_at': datetime.now(timezone.utc)
        })
        # Usar deques para highs/lows recentes para cálculo de swing points
        self.recent_high_prices_buffer: deque[float] = deque(maxlen=200) # Renomeado e aumentado
        self.recent_low_prices_buffer: deque[float] = deque(maxlen=200)  # Renomeado e aumentado
        self.detected_sweep_events: deque[Dict[str, Any]] = deque(maxlen=20) # Renomeado e tipado

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Detecção de Nível de Liquidez
            'level_detection_lookback_periods': 100, # Renomeado e aumentado
            'level_cluster_threshold_pips': 3.0, # Renomeado e em pips
            'min_touches_for_strong_level': 3, # Renomeado
            'level_strength_decay_hours': 24 * 5, # Força do nível decai após X horas sem toque

            # Detecção de Sweep (Varredura)
            'sweep_detection_window_ticks': 20, # Janela de ticks para detectar sweep
            'sweep_min_velocity_pips_per_sec': 10.0, # Renomeado e unidade clara
            'sweep_min_penetration_pips': 2.0, # Renomeado e em pips
            'sweep_max_duration_seconds': 15.0, # Renomeado
            'reversal_confirmation_ticks': 3, # Ticks de confirmação da reversão pós-sweep
            'min_volume_increase_on_sweep': 1.2, # Aumento de volume durante o sweep (ratio)

            # Análise do DOM (se disponível e usado)
            'dom_imbalance_ratio_for_hunt': 2.5, # Renomeado
            # 'min_liquidity_size_at_level': 1000000, # $1M (difícil de estimar sem dados de volume reais por nível)
            'enable_dom_absorption_check': True, # Renomeado

            # Gestão de Risco para esta estratégia
            'atr_period_stops': 14,
            'atr_multiplier_sl_hunt': 1.2, # SL mais apertado para stop hunts
            'target_risk_reward_ratio_hunt': 3.0, # R:R maior devido à natureza da reversão
            'use_swept_level_for_stop': True, # Colocar stop além do nível varrido

            # Filtros
            'round_number_check_buffer_pips': 5.0,
            'apply_session_filter_hunt': True, # Filtro de sessão específico
            # 'avoid_news_window_minutes': 15, # Pode ser herdado de BaseStrategy
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula e atualiza níveis de liquidez, detecta sweeps e outros indicadores."""
        recent_ticks_list = market_context.get('recent_ticks', []) # Lista de TickData
        if not recent_ticks_list or len(recent_ticks_list) < self.parameters['level_detection_lookback_periods']:
            self.logger.debug("Dados insuficientes para indicadores da StopHuntStrategy.")
            self.current_indicators = {}
            return

        # Converter ticks para arrays para TA-Lib e cálculos numpy
        # Usar 'ask' para highs e 'bid' para lows para Donchian/swing points
        high_prices_arr = np.array([tick.ask for tick in recent_ticks_list if hasattr(tick, 'ask')]) # Renomeado
        low_prices_arr = np.array([tick.bid for tick in recent_ticks_list if hasattr(tick, 'bid')])   # Renomeado
        mid_prices_arr = np.array([tick.mid for tick in recent_ticks_list if hasattr(tick, 'mid')])   # Renomeado
        # Timestamps para cálculo de duração de sweep
        timestamps_arr = np.array([pd.Timestamp(tick.timestamp, tz='UTC') for tick in recent_ticks_list if hasattr(tick, 'timestamp')])


        # Atualizar buffers de highs/lows recentes para detecção de swing points
        if hasattr(recent_ticks_list[-1], 'ask'): self.recent_high_prices_buffer.append(recent_ticks_list[-1].ask)
        if hasattr(recent_ticks_list[-1], 'bid'): self.recent_low_prices_buffer.append(recent_ticks_list[-1].bid)


        # Identificar níveis de liquidez (esta função atualiza self.identified_liquidity_levels)
        self._update_identified_liquidity_levels(high_prices_arr, low_prices_arr, mid_prices_arr[-1])

        # Detectar sweep atual
        sweep_detected_flag, sweep_info_dict = self._check_for_active_sweep( # Renomeado
            mid_prices_arr, timestamps_arr, market_context.get('dom') # Passar DOM se usado
        )
        if sweep_detected_flag and sweep_info_dict:
            self.detected_sweep_events.append(sweep_info_dict) # Adicionar ao histórico de sweeps


        # ATR para stops e volatilidade
        atr_val = 0.0
        pip_size_val = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01
        if len(high_prices_arr) >= self.parameters['atr_period_stops'] and \
           len(low_prices_arr) >= self.parameters['atr_period_stops'] and \
           len(mid_prices_arr) >= self.parameters['atr_period_stops']:
            atr_val = talib.ATR(high_prices_arr, low_prices_arr, mid_prices_arr,
                                timeperiod=self.parameters['atr_period_stops'])[-1]


        current_mid_price_val = mid_prices_arr[-1] # Renomeado
        nearest_res_level, nearest_sup_level = self._find_nearest_liquidity_levels(current_mid_price_val) # Renomeado

        self.current_indicators = {
            'identified_liquidity_levels': self.identified_liquidity_levels.copy(), # Enviar uma cópia
            'is_sweep_detected_now': sweep_detected_flag, # Renomeado
            'current_sweep_info': sweep_info_dict, # Renomeado
            # 'hidden_liquidity_analysis': hidden_liquidity_data, # Se _analyze_hidden_liquidity for implementado
            'nearest_resistance_level': nearest_res_level, # Renomeado
            'nearest_support_level': nearest_sup_level,   # Renomeado
            'current_price_mid': current_mid_price_val,
            'atr_value_price': atr_val, # ATR em valor de preço
            'atr_pips': atr_val / pip_size_val if atr_val > 0 and pip_size_val > 0 else 0.0,
            'spread_pips': market_context.get('spread', 0.0) / pip_size_val,
            'recent_sweep_count': len(self.detected_sweep_events), # Renomeado
            # 'current_session': self._get_trading_session_from_context(market_context) # Se BaseStrategy não tiver
        }

    def _update_identified_liquidity_levels(self, high_prices: np.ndarray, low_prices: np.ndarray, current_price: float): # Renomeado
        """Identifica e atualiza níveis com provável concentração de stops (swing points, números redondos)."""
        # Níveis de Swing (exemplo simplificado usando máximas/mínimas do buffer)
        # Uma detecção de swing points mais robusta (ex: 3 barras de cada lado) seria melhor.
        lookback = self.parameters['level_detection_lookback_periods']
        if len(self.recent_high_prices_buffer) >= lookback:
            potential_resistance = np.max(list(self.recent_high_prices_buffer)[-lookback:])
            self._add_or_update_liquidity_level(potential_resistance, 'swing_high')
        if len(self.recent_low_prices_buffer) >= lookback:
            potential_support = np.min(list(self.recent_low_prices_buffer)[-lookback:])
            self._add_or_update_liquidity_level(potential_support, 'swing_low')

        # Níveis Redondos próximos ao preço atual
        pip_unit = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01
        buffer_pips = self.parameters['round_number_check_buffer_pips']
        # Exemplo: para 1.0834, checar 1.0800, 1.0850, 1.0900
        # Níveis .00
        base_00 = round(current_price / (100 * pip_unit)) * (100 * pip_unit)
        for i in range(-2, 3): # Alguns níveis .00 acima e abaixo
            level_00 = base_00 + i * (100 * pip_unit)
            if abs(level_00 - current_price) < buffer_pips * 5 * pip_unit: # Se próximo o suficiente
                self._add_or_update_liquidity_level(level_00, 'round_00')
        # Níveis .50
        base_50 = round(current_price / (50 * pip_unit)) * (50 * pip_unit)
        for i in range(-2, 3): # Alguns níveis .50
            level_50 = base_50 + i * (50 * pip_unit)
            # Evitar duplicar com níveis .00 se forem o mesmo
            if abs(level_50 - current_price) < buffer_pips * 5 * pip_unit and not any(abs(level_50 - l) < pip_unit for l in self.identified_liquidity_levels if self.identified_liquidity_levels[l]['type'] == 'round_00'):
                self._add_or_update_liquidity_level(level_50, 'round_50')

        # Decaimento da força dos níveis não tocados recentemente
        now = datetime.now(timezone.utc)
        decay_hours = self.parameters['level_strength_decay_hours']
        levels_to_delete = []
        for price_level, info in self.identified_liquidity_levels.items():
            if (now - info['last_touch_at']).total_seconds() > decay_hours * 3600:
                info['strength'] *= 0.9 # Reduzir força
                if info['strength'] < 0.1: # Remover níveis muito fracos/antigos
                    levels_to_delete.append(price_level)
        for p_level in levels_to_delete: # Renomeado
            del self.identified_liquidity_levels[p_level]


    def _add_or_update_liquidity_level(self, price_val: float, level_type_str: str): # Renomeado
        """Adiciona ou atualiza um nível de liquidez, agrupando níveis próximos."""
        if np.isnan(price_val): return

        cluster_thresh_price = self.parameters['level_cluster_threshold_pips'] * (0.0001 if "JPY" not in CONFIG.SYMBOL else 0.01) # Renomeado
        
        # Procurar nível existente próximo para agrupar
        merged = False
        for existing_level_price, info in list(self.identified_liquidity_levels.items()): # Iterar sobre cópia
            if abs(existing_level_price - price_val) <= cluster_thresh_price:
                # Mesclar com o nível existente: atualizar preço para média, incrementar toques, aumentar força
                new_avg_price = (existing_level_price * info['touches'] + price_val) / (info['touches'] + 1)
                info['touches'] += 1
                info['strength'] = min(1.0, info['strength'] + 0.2) # Aumentar força com cada toque
                info['last_touch_at'] = datetime.now(timezone.utc)
                # Se o preço médio mudou, precisa remapear no dicionário
                if abs(new_avg_price - existing_level_price) > 1e-6: # Se houve mudança significativa
                    del self.identified_liquidity_levels[existing_level_price]
                    self.identified_liquidity_levels[new_avg_price] = info
                merged = True
                break
        
        if not merged: # Criar novo nível
            self.identified_liquidity_levels[price_val] = {
                'type': level_type_str,
                'touches': 1,
                'strength': 0.3, # Força inicial
                'created_at': datetime.now(timezone.utc),
                'last_touch_at': datetime.now(timezone.utc)
            }

    # _find_round_numbers foi integrado em _update_identified_liquidity_levels

    def _check_for_active_sweep(self, mid_prices: np.ndarray, timestamps: np.ndarray, # Renomeado
                        current_dom: Optional[DOMSnapshot]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verifica se um sweep de liquidez está ocorrendo ou acabou de ocorrer."""
        if len(mid_prices) < self.parameters['sweep_detection_window_ticks']:
            return False, None

        pip_sz_sweep = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01 # Renomeado

        # Analisar a janela mais recente de ticks
        window_prices = mid_prices[-self.parameters['sweep_detection_window_ticks']:]
        window_timestamps_pd = pd.to_datetime(timestamps[-self.parameters['sweep_detection_window_ticks':]], utc=True) # Converter para pd.Timestamp

        # Verificar cada nível de liquidez identificado
        for level_price_val, level_info_dict in self.identified_liquidity_levels.items(): # Renomeado
            # Determinar direção esperada do sweep em relação ao nível
            # Se nível é resistência, esperamos sweep para CIMA. Se suporte, para BAIXO.
            is_resistance = 'high' in level_info_dict['type'] or 'round' in level_info_dict['type'] # round pode ser ambos
            is_support = 'low' in level_info_dict['type'] or 'round' in level_info_dict['type']

            # Verificar se o preço cruzou o nível na janela
            price_at_start_of_window = window_prices[0]
            price_at_end_of_window = window_prices[-1]
            max_price_in_window = np.max(window_prices)
            min_price_in_window = np.min(window_prices)

            sweep_direction_str: Optional[str] = None # Renomeado
            penetration_pips_val = 0.0 # Renomeado

            if is_resistance and price_at_start_of_window < level_price_val and max_price_in_window > level_price_val:
                # Potencial sweep de alta através de uma resistência
                penetration_pips_val = (max_price_in_window - level_price_val) / pip_sz_sweep
                if penetration_pips_val >= self.parameters['sweep_min_penetration_pips']:
                    sweep_direction_str = 'up'
            elif is_support and price_at_start_of_window > level_price_val and min_price_in_window < level_price_val:
                # Potencial sweep de baixa através de um suporte
                penetration_pips_val = (level_price_val - min_price_in_window) / pip_sz_sweep
                if penetration_pips_val >= self.parameters['sweep_min_penetration_pips']:
                    sweep_direction_str = 'down'
            
            if sweep_direction_str:
                # Calcular velocidade e duração do movimento que causou o sweep
                # Encontrar o ponto onde o nível foi cruzado
                cross_idx = -1
                for k_idx in range(len(window_prices)): # Renomeado k para k_idx
                    if (sweep_direction_str == 'up' and window_prices[k_idx] > level_price_val) or \
                       (sweep_direction_str == 'down' and window_prices[k_idx] < level_price_val):
                        cross_idx = k_idx
                        break
                
                if cross_idx > 0: # Nível foi cruzado dentro da janela (não no primeiro tick)
                    # Movimento do início da janela até o cruzamento, ou do cruzamento até o pico/vale
                    # Vamos focar no movimento que *rompeu* o nível
                    # Achar o tick *antes* de cruzar e o tick do *pico/vale* da penetração
                    entry_tick_idx = np.where(window_prices[:cross_idx+1] <= level_price_val)[0][-1] if sweep_direction_str == 'up' else \
                                     np.where(window_prices[:cross_idx+1] >= level_price_val)[0][-1]
                    
                    peak_or_valley_price = max_price_in_window if sweep_direction_str == 'up' else min_price_in_window
                    # Encontrar o índice do pico/vale *após* o cruzamento ou no cruzamento
                    idx_of_extreme_penetration = cross_idx + np.argmax(window_prices[cross_idx:]) if sweep_direction_str == 'up' else \
                                                 cross_idx + np.argmin(window_prices[cross_idx:])

                    price_change_sweep = abs(peak_or_valley_price - window_prices[entry_tick_idx])
                    time_diff_sweep_seconds = (window_timestamps_pd[idx_of_extreme_penetration] - window_timestamps_pd[entry_tick_idx]).total_seconds()

                    if time_diff_sweep_seconds > 1e-3 and time_diff_sweep_seconds <= self.parameters['sweep_max_duration_seconds']:
                        velocity_pips_s = (price_change_sweep / pip_sz_sweep) / time_diff_sweep_seconds
                        
                        if velocity_pips_s >= self.parameters['sweep_min_velocity_pips_per_sec']:
                            # Checar se houve reversão APÓS o pico/vale do sweep
                            # (ex: últimos N ticks mostraram movimento contrário)
                            if self._confirm_post_sweep_reversal(mid_prices, sweep_direction_str, peak_or_valley_price):
                                sweep_event_info = { # Renomeado
                                    'swept_level_price': level_price_val,
                                    'level_type': level_info_dict['type'],
                                    'level_strength': level_info_dict['strength'],
                                    'sweep_velocity_pips_s': velocity_pips_s,
                                    'penetration_pips': penetration_pips_val,
                                    'sweep_timestamp_utc': timestamps[-1].isoformat(), # Timestamp do último tick da janela de detecção
                                    'sweep_direction': sweep_direction_str, # Direção do sweep (up/down)
                                    'peak_or_valley_of_sweep': peak_or_valley_price
                                }
                                return True, sweep_event_info
        return False, None


    def _confirm_post_sweep_reversal(self, all_mid_prices: np.ndarray, sweep_direction: str, # Renomeado
                                    extreme_price_of_sweep: float) -> bool:
        """Verifica se houve uma reversão nos últimos N ticks após o pico/vale do sweep."""
        num_reversal_ticks = self.parameters['reversal_confirmation_ticks']
        if len(all_mid_prices) < num_reversal_ticks:
            return False

        last_n_prices = all_mid_prices[-num_reversal_ticks:]
        current_price = all_mid_prices[-1]

        if sweep_direction == 'up': # Sweep foi para cima, esperamos reversão para baixo
            # Preço atual deve ser menor que o pico do sweep
            # E os últimos N ticks devem mostrar movimento de queda ou consolidação abaixo do pico
            return current_price < extreme_price_of_sweep and (last_n_prices[-1] < last_n_prices[0] or np.std(last_n_prices) < (0.5 / 10000)) # Ex: std dev < 0.5 pips
        elif sweep_direction == 'down': # Sweep foi para baixo, esperamos reversão para cima
            return current_price > extreme_price_of_sweep and (last_n_prices[-1] > last_n_prices[0] or np.std(last_n_prices) < (0.5 / 10000))
        return False

    # _analyze_hidden_liquidity foi removido pois requer dados de DOM muito específicos e
    # a implementação original estava vazia. Pode ser adicionado se houver uma lógica clara.


    def _find_nearest_liquidity_levels(self, current_mid_price: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: # Renomeado
        """Encontra os níveis de liquidez (resistência e suporte) mais próximos do preço atual."""
        nearest_res: Optional[Dict[str, Any]] = None # Renomeado
        min_dist_to_res = float('inf')
        nearest_sup: Optional[Dict[str, Any]] = None # Renomeado
        min_dist_to_sup = float('inf')

        if not self.identified_liquidity_levels:
            return None, None

        for level_px, level_inf in self.identified_liquidity_levels.items(): # Renomeado
            distance = abs(level_px - current_mid_price)
            level_data_to_store = {'price': level_px, **level_inf} # Incluir preço no dict retornado

            if level_px > current_mid_price: # Potencial resistência
                if distance < min_dist_to_res:
                    min_dist_to_res = distance
                    nearest_res = level_data_to_store
            elif level_px < current_mid_price: # Potencial suporte
                if distance < min_dist_to_sup:
                    min_dist_to_sup = distance
                    nearest_sup = level_data_to_store
        
        return nearest_res, nearest_sup

    # _get_current_session removido, pois BaseStrategy ou Orchestrator podem fornecer isso.

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera um sinal de trading se um stop hunt seguido de reversão for detectado."""
        indic = self.current_indicators
        if not indic or not indic.get('is_sweep_detected_now', False) or not indic.get('current_sweep_info'):
            return None

        current_sweep = indic['current_sweep_info'] # Renomeado

        # Aplicar filtros (ex: sessão, evitar notícias - BaseStrategy pode lidar com isso)
        # if self.parameters['apply_session_filter_hunt']:
        #     if indic.get('current_session') not in ['london', 'newyork', 'overlap']: # Exemplo
        #         return None

        # O sinal é na DIREÇÃO OPOSTA ao sweep (pois é uma estratégia de reversão)
        signal_trade_side: Optional[str] = None # Renomeado
        if current_sweep['sweep_direction'] == 'up': # Sweep de alta (varreu stops de venda) -> Entrar vendido
            signal_trade_side = 'sell'
        elif current_sweep['sweep_direction'] == 'down': # Sweep de baixa (varreu stops de compra) -> Entrar comprado
            signal_trade_side = 'buy'
        
        if not signal_trade_side:
            return None

        # (Opcional) Confirmação adicional com DOM ou volume no momento da reversão
        # if self.parameters['enable_dom_absorption_check']:
        #    dom_data = market_context.get('dom') # Precisa ser o DOM ATUAL da reversão
        #    if dom_data and not self._confirm_absorption_against_sweep(dom_data, signal_trade_side):
        #        return None


        return self._create_stop_hunt_reversal_signal( # Renomeado
            signal_trade_side, indic, current_sweep, market_context
        )


    def _create_stop_hunt_reversal_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                           sweep_details: Dict[str, Any], market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a entrada de reversão pós-stop hunt."""
        current_price_ref = indicators_dict['current_price_mid'] # Preço de referência para entrada
        atr_pips_val = indicators_dict.get('atr_pips', 10.0) # Default ATR de 10 pips se não calculado
        pip_size_val = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01

        sl_atr_mult_val = self.parameters['atr_multiplier_sl_hunt'] # Renomeado
        swept_level = sweep_details['swept_level_price']

        stop_loss_val: float # Adicionada tipagem
        take_profit_val: float # Adicionada tipagem

        if signal_side == 'buy': # Entrando comprado após sweep de baixa
            # SL um pouco abaixo do mínimo do sweep ou do nível varrido + ATR buffer
            sl_reference = sweep_details.get('peak_or_valley_of_sweep', swept_level) # Mínima do sweep
            stop_loss_val = sl_reference - (atr_pips_val * sl_atr_mult_val * 0.5 * pip_size_val) # Mais apertado
            if self.parameters['use_swept_level_for_stop']:
                 stop_loss_val = min(stop_loss_val, swept_level - (1 * pip_size_val)) # Garantir que está além do nível

            # TP pode ser o próximo nível de liquidez (resistência) ou R:R
            if indicators_dict.get('nearest_resistance_level') and \
               (indicators_dict['nearest_resistance_level']['price'] - current_price_ref) / pip_size_val > atr_pips_val * 1.5 : # Se TP > 1.5 * ATR
                take_profit_val = indicators_dict['nearest_resistance_level']['price'] - (1 * pip_size_val) # Um pouco antes
            else:
                risk_pips_val = abs(current_price_ref - stop_loss_val) / pip_size_val # Renomeado
                take_profit_val = current_price_ref + (risk_pips_val * self.parameters['target_risk_reward_ratio_hunt'] * pip_size_val)
        
        else: # signal_side == 'sell' - Entrando vendido após sweep de alta
            sl_reference = sweep_details.get('peak_or_valley_of_sweep', swept_level) # Máxima do sweep
            stop_loss_val = sl_reference + (atr_pips_val * sl_atr_mult_val * 0.5 * pip_size_val)
            if self.parameters['use_swept_level_for_stop']:
                stop_loss_val = max(stop_loss_val, swept_level + (1 * pip_size_val))

            if indicators_dict.get('nearest_support_level') and \
               (current_price_ref - indicators_dict['nearest_support_level']['price']) / pip_size_val > atr_pips_val * 1.5:
                take_profit_val = indicators_dict['nearest_support_level']['price'] + (1 * pip_size_val)
            else:
                risk_pips_val = abs(stop_loss_val - current_price_ref) / pip_size_val
                take_profit_val = current_price_ref - (risk_pips_val * self.parameters['target_risk_reward_ratio_hunt'] * pip_size_val)

        # Calcular confiança (0.5 a 1.0)
        confidence = 0.65 # Confiança base para este tipo de setup
        if sweep_details['sweep_velocity_pips_s'] > self.parameters['sweep_min_velocity_pips_per_sec'] * 1.5:
            confidence += 0.10
        if sweep_details['level_strength'] > 0.7: # Nível forte varrido
            confidence += 0.10
        # Adicionar mais fatores de confiança (ex: volume no sweep, força da reversão)

        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=round(np.clip(confidence, 0.5, 0.95), 4),
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"StopHunt Reversal: {signal_side.upper()} após sweep {sweep_details['sweep_direction']} do nível {sweep_details['swept_level_price']:.5f}",
            metadata={
                'sweep_details': sweep_details, # Guardar detalhes do sweep
                'atr_pips_at_signal': atr_pips_val,
                'nearest_resistance_at_signal': indicators_dict.get('nearest_resistance_level'),
                'nearest_support_at_signal': indicators_dict.get('nearest_support_level')
            }
        )


    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Condições de saída para a estratégia de Stop Hunt Reversal."""
        indic = self.current_indicators
        if not indic: return None

        current_price_mid_val = market_context['tick'].mid # Renomeado

        # 1. Saída se novo sweep na MESMA direção da posição (ou seja, falha da reversão e continuação)
        if indic.get('is_sweep_detected_now', False) and indic.get('current_sweep_info'):
            new_sweep_info = indic['current_sweep_info'] # Renomeado
            # Se a posição é BUY (esperando alta) e ocorre novo SWEEP PARA BAIXO do nível de entrada da posição ou do SL.
            if open_position.side.lower() == 'buy' and new_sweep_info['sweep_direction'] == 'down' and \
               new_sweep_info.get('peak_or_valley_of_sweep', current_price_mid_val) < open_position.entry_price:
                return ExitSignal(position_id_to_close=open_position.id, reason="Falha da reversão: Novo sweep de baixa contra posição comprada.")
            # Se a posição é SELL (esperando baixa) e ocorre novo SWEEP PARA CIMA.
            elif open_position.side.lower() == 'sell' and new_sweep_info['sweep_direction'] == 'up' and \
                 new_sweep_info.get('peak_or_valley_of_sweep', current_price_mid_val) > open_position.entry_price:
                return ExitSignal(position_id_to_close=open_position.id, reason="Falha da reversão: Novo sweep de alta contra posição vendida.")


        # 2. Trailing Stop Agressivo (a lógica de atualização do SL da posição é em BaseStrategy/ExecutionEngine)
        # Esta função pode decidir sair se o lucro atingir um certo ponto e a volatilidade aumentar,
        # ou se o preço voltar X pips do pico de lucro.
        # Exemplo: Sair se o preço retrair X% do MFE (Maximum Favorable Excursion)
        # Isso requer rastrear MFE na 'open_position.metadata'
        
        # if 'mfe_pips' in open_position.metadata:
        #     pnl_pips_current = 0.0
        #     pip_size_exit = 0.0001 if "JPY" not in open_position.symbol.upper() else 0.01
        #     if open_position.side.lower() == 'buy':
        #         pnl_pips_current = (current_price_mid_val - open_position.entry_price) / pip_size_exit
        #     else:
        #         pnl_pips_current = (open_position.entry_price - current_price_mid_val) / pip_size_exit
            
        #     mfe_pips_val = open_position.metadata['mfe_pips']
        #     if pnl_pips_current > atr_pips_val * 0.5 and pnl_pips_current < mfe_pips_val * 0.6: # Retraiu 40% do MFE
        #         return ExitSignal(position_id_to_close=open_position.id, reason="Retração significativa do MFE.")


        # 3. Saída por tempo se a reversão não se materializar rapidamente
        # (ex: se após N barras/tempo, o preço não se moveu favoravelmente)
        max_hold_seconds_no_profit = self.parameters.get('max_hold_seconds_if_no_profit', 30 * 60) # Ex: 30 min
        time_held_val = (datetime.now(timezone.utc) - open_position.open_time).total_seconds() # Renomeado

        current_pnl_val = 0.0 # Renomeado
        if open_position.side.lower() == 'buy': current_pnl_val = current_price_mid_val - open_position.entry_price
        else: current_pnl_val = open_position.entry_price - current_price_mid_val
        
        if time_held_val > max_hold_seconds_no_profit and current_pnl_val <= 0 : # Se não lucrativa após X tempo
             return ExitSignal(position_id_to_close=open_position.id, reason=f"Sem lucro após {max_hold_seconds_no_profit/60:.0f} min.")


        return None