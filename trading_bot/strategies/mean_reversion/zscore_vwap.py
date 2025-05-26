# strategies/mean_reversion/zscore_vwap.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados
import talib # Para RSI, ATR
from typing import Dict, Optional, Any, List, Tuple # Adicionado Tuple
from datetime import datetime, timedelta, timezone # Adicionado timezone
from collections import deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default e outras configs

logger = setup_logger("zscore_vwap_strategy_logger") # Nome do logger específico

class ZScoreVWAPStrategy(BaseStrategy):
    """
    Estratégia de reversão à média usando Z-Score do preço em relação ao VWAP da sessão/período.
    Entra quando o preço se desvia significativamente do VWAP, esperando um retorno à média.
    """

    def __init__(self):
        super().__init__("ZScoreVWAPReversion") # Nome da estratégia
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.LOW_VOLUME] # Mais adequado para estes
        self.min_time_between_signals_sec = 180  # 3 minutos

        # Buffers para cálculo de VWAP da sessão/período
        # Estes serão resetados conforme a lógica de 'reset_vwap_on_new_period'
        self.session_prices_for_vwap: List[float] = [] # Renomeado
        self.session_volumes_for_vwap: List[float] = [] # Renomeado
        self.session_vwap_history: deque[float] = deque(maxlen=200) # Para std dev do VWAP ou Z-score do VWAP em si
        self.current_session_start_time: Optional[datetime] = None # Renomeado

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Parâmetros de Z-Score e VWAP
            'zscore_entry_threshold': 2.0,     # Z-Score para entrar
            'zscore_exit_target_threshold': 0.25, # Z-Score para sair (próximo de zero) # Renomeado
            'zscore_stop_loss_threshold': 3.0,  # Z-Score para stop loss (movimento continuou contra) # Renomeado
            'vwap_calculation_lookback_ticks': 500, # Número de ticks recentes para calcular VWAP e StdDev do Preço vs VWAP
                                                 # Ou usar 'vwap_session_based': True para VWAP da sessão
            'reset_vwap_on_new_period': "hour", # "hour", "session" (London, NY), ou None para contínuo

            # Gestão de Risco
            'atr_period_for_sl': 14,
            'base_atr_multiplier_sl': 1.5, # SL base em ATR, pode ser ajustado pelo Z-Score
            # 'target_risk_reward_ratio': 1.5, # TP é geralmente o VWAP ou Z-Score de saída

            # Filtros
            'min_ticks_for_vwap_calc': 100, # Mínimo de ticks no período para VWAP válido
            'use_rsi_filter_zscore': True, # Renomeado
            'rsi_period_zscore': 14,
            'rsi_oversold_entry_zscore': 30,
            'rsi_overbought_entry_zscore': 70,
            'max_spread_pips_zscore_entry': 1.2,

            # Gestão da Posição
            'max_position_duration_hours': 4,  # Máximo 4 horas para uma trade de reversão à média
            # 'scale_in_enabled_zscore': False, # Não implementado no exemplo original
        }

    def _reset_vwap_session_data(self, current_event_time: datetime): # Renomeado
        """Reseta os buffers para cálculo de um novo VWAP de sessão/período."""
        self.session_prices_for_vwap = []
        self.session_volumes_for_vwap = []
        # self.session_vwap_history.clear() # Pode não ser necessário limpar o histórico de VWAPs anteriores
        self.current_session_start_time = current_event_time
        self.logger.info(f"Dados da sessão VWAP resetados às {current_event_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    def _should_reset_vwap(self, current_event_time: datetime) -> bool:
        """Verifica se o VWAP e os dados da sessão devem ser resetados."""
        if self.current_session_start_time is None:
            return True # Primeira vez, precisa inicializar

        reset_period = self.parameters['reset_vwap_on_new_period']
        last_start = self.current_session_start_time

        if reset_period == "hour":
            return current_event_time.hour != last_start.hour
        elif reset_period == "session":
            # Lógica para detectar mudança de sessão principal (ex: Ásia->Londres, Londres->NY)
            # Esta lógica pode ser complexa e depender dos horários exatos das sessões.
            # Exemplo simplificado:
            current_session_name = self._get_trading_session_name(current_event_time.hour) # Helper
            last_session_name = self._get_trading_session_name(last_start.hour)
            return current_session_name != last_session_name
        
        return False # Se None ou outro valor, não reseta periodicamente

    def _get_trading_session_name(self, hour_utc: int) -> str: # Helper para _should_reset_vwap
        # Simplificado, alinhar com Orchestrator._get_trading_session
        if 7 <= hour_utc < 13: return "London" # Antes do overlap
        if 13 <= hour_utc < 16: return "Overlap"
        if 16 <= hour_utc < 22: return "NewYork" # Após overlap
        if 22 <= hour_utc or hour_utc < 7: return "Asia_Sydney"
        return "Transition"


    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula VWAP, Z-Score do preço em relação ao VWAP, e outros indicadores."""
        self.current_indicators = {} # Resetar
        
        # Pegar tick atual e ticks recentes
        current_tick_obj: Optional[TickData] = market_context.get('tick') # Renomeado
        recent_ticks_list = market_context.get('recent_ticks', []) # Lista de TickData
        
        if not current_tick_obj or not hasattr(current_tick_obj, 'timestamp') or not hasattr(current_tick_obj, 'mid'):
            self.logger.debug("Tick atual inválido ou ausente para ZScoreVWAP.")
            return
        
        current_event_time_utc = current_tick_obj.timestamp # Já deve ser datetime UTC
        
        # Verificar se deve resetar dados da sessão VWAP
        if self._should_reset_vwap(current_event_time_utc):
            self._reset_vwap_session_data(current_event_time_utc)

        # Adicionar preço e volume atuais aos buffers da sessão
        current_price = current_tick_obj.mid
        current_volume = (current_tick_obj.bid_volume + current_tick_obj.ask_volume) / 2.0
        if current_volume == 0: current_volume = 1.0 # Volume mínimo para evitar problemas

        self.session_prices_for_vwap.append(current_price)
        self.session_volumes_for_vwap.append(current_volume)

        # Manter o buffer de sessão dentro de um limite razoável (ex: últimas N horas ou X ticks)
        # A lógica de reset já lida com isso para "hour" ou "session".
        # Se reset_vwap_on_new_period for None, precisa de um maxlen aqui.
        max_session_ticks = self.parameters.get('vwap_calculation_lookback_ticks', 500)
        if len(self.session_prices_for_vwap) > max_session_ticks:
            self.session_prices_for_vwap = self.session_prices_for_vwap[-max_session_ticks:]
            self.session_volumes_for_vwap = self.session_volumes_for_vwap[-max_session_ticks:]


        # Calcular VWAP e desvio padrão dos preços em relação ao VWAP para a sessão atual
        session_vwap_val = 0.0 # Renomeado
        price_vs_vwap_std_dev = 0.0 # Renomeado
        zscore_val_ci = 0.0 # Renomeado

        if len(self.session_prices_for_vwap) >= self.parameters['min_ticks_for_vwap_calc']:
            prices_arr = np.array(self.session_prices_for_vwap, dtype=float)
            volumes_arr = np.array(self.session_volumes_for_vwap, dtype=float)

            if np.sum(volumes_arr) > 1e-9: # Evitar divisão por zero
                session_vwap_val = np.sum(prices_arr * volumes_arr) / np.sum(volumes_arr)
                self.session_vwap_history.append(session_vwap_val) # Guardar histórico de VWAPs da sessão

                # Desvio padrão dos preços da sessão em relação ao VWAP da sessão
                deviations_from_vwap = prices_arr - session_vwap_val
                price_vs_vwap_std_dev = np.std(deviations_from_vwap)

                if price_vs_vwap_std_dev > 1e-9: # Evitar divisão por zero
                    zscore_val_ci = (current_price - session_vwap_val) / price_vs_vwap_std_dev
            else: # Se não houver volume, não se pode calcular VWAP de forma significativa
                session_vwap_val = current_price # Usar preço atual como fallback
        else: # Não há ticks suficientes para cálculo
            self.logger.debug(f"Ticks insuficientes na sessão ({len(self.session_prices_for_vwap)}) para calcular VWAP/ZScore.")
            # Manter valores default (0 ou None)


        # Indicadores auxiliares (RSI, ATR)
        # Usar todos os recent_ticks_list para estes, não apenas os da sessão VWAP
        close_prices_full_arr = self._get_prices_from_context(market_context, 'mid') # Renomeado
        rsi_val_ci = 50.0 # Default
        if self.parameters['use_rsi_filter_zscore'] and len(close_prices_full_arr) >= self.parameters['rsi_period_zscore'] + 1:
            rsi_val_ci = talib.RSI(close_prices_full_arr, timeperiod=self.parameters['rsi_period_zscore'])[-1]
            if np.isnan(rsi_val_ci): rsi_val_ci = 50.0

        atr_pips_val_ci = 0.0 # Renomeado
        pip_size_ci = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01
        if len(recent_ticks_list) >= self.parameters['atr_period_for_sl'] + 1:
            high_prices_full_arr = self._get_prices_from_context(market_context, 'high') # Renomeado
            low_prices_full_arr = self._get_prices_from_context(market_context, 'low')    # Renomeado
            if len(high_prices_full_arr) == len(close_prices_full_arr) and len(low_prices_full_arr) == len(close_prices_full_arr):
                atr_price_val = talib.ATR(high_prices_full_arr, low_prices_full_arr, close_prices_full_arr,
                                       timeperiod=self.parameters['atr_period_for_sl'])[-1]
                if not np.isnan(atr_price_val) and pip_size_ci > 0:
                    atr_pips_val_ci = atr_price_val / pip_size_ci


        self.current_indicators = {
            'session_vwap': session_vwap_val,
            'price_deviation_from_vwap_std': price_vs_vwap_std_dev, # Std dev dos (preço - vwap)
            'current_zscore_price_vs_vwap': zscore_val_ci, # Z-Score do preço atual vs VWAP da sessão
            'current_price_mid': current_price,
            'rsi_value': rsi_val_ci,
            'atr_pips': atr_pips_val_ci,
            'ticks_in_session_vwap_calc': len(self.session_prices_for_vwap),
            'spread_pips': market_context.get('spread', 0.0) / pip_size_ci,
            'session_start_time_utc': self.current_session_start_time
        }

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de trading (entrada) baseado na reversão do Z-Score ao VWAP."""
        indic = self.current_indicators
        if not indic or indic.get('ticks_in_session_vwap_calc', 0) < self.parameters['min_ticks_for_vwap_calc']:
            return None # Sem indicadores válidos ou ticks insuficientes

        # Aplicar filtros de entrada
        max_spread = self.parameters['max_spread_pips_zscore_entry']
        if indic.get('spread_pips', float('inf')) > max_spread:
            self.logger.debug(f"Spread ({indic.get('spread_pips'):.1f} pips) muito alto para ZScoreVWAP (Max: {max_spread}).")
            return None

        current_zscore_gs = indic.get('current_zscore_price_vs_vwap', 0.0) # Renomeado
        entry_z_thresh_gs = self.parameters['zscore_entry_threshold'] # Renomeado
        signal_side_gs: Optional[str] = None # Renomeado

        if abs(current_zscore_gs) >= entry_z_thresh_gs:
            # Z-Score positivo alto = preço muito acima do VWAP -> VENDER esperando reversão para baixo
            if current_zscore_gs >= entry_z_thresh_gs:
                if self.parameters['use_rsi_filter_zscore']:
                    if indic.get('rsi_value', 50.0) < self.parameters['rsi_overbought_entry_zscore']: # RSI não deve estar sobrecomprado para venda
                        self.logger.debug(f"Sinal de VENDA ZScoreVWAP ignorado: RSI ({indic.get('rsi_value'):.1f}) não confirma sobrecompra.")
                        return None
                signal_side_gs = 'sell'
            # Z-Score negativo baixo = preço muito abaixo do VWAP -> COMPRAR esperando reversão para cima
            elif current_zscore_gs <= -entry_z_thresh_gs:
                if self.parameters['use_rsi_filter_zscore']:
                    if indic.get('rsi_value', 50.0) > self.parameters['rsi_oversold_entry_zscore']: # RSI não deve estar sobrevendido para compra
                        self.logger.debug(f"Sinal de COMPRA ZScoreVWAP ignorado: RSI ({indic.get('rsi_value'):.1f}) não confirma sobrevenda.")
                        return None
                signal_side_gs = 'buy'

        if signal_side_gs:
            return self._create_zscore_reversion_signal(signal_side_gs, indic, market_context) # Renomeado
        return None

    def _create_zscore_reversion_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                               market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a estratégia ZScoreVWAP."""
        current_price_create = indicators_dict['current_price_mid'] # Renomeado
        session_vwap_create = indicators_dict['session_vwap'] # Renomeado
        atr_pips_create = indicators_dict.get('atr_pips', 10.0) # Renomeado
        if atr_pips_create == 0: atr_pips_create = 10.0 # Default se ATR for zero

        pip_size_create_sig = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        # Stop Loss: pode ser baseado em um Z-Score extremo ou um múltiplo de ATR
        z_stop_thresh_create = self.parameters['zscore_stop_loss_threshold'] # Renomeado
        atr_sl_mult_create = self.parameters['base_atr_multiplier_sl'] # Renomeado
        
        sl_based_on_z = 0.0
        if indicators_dict.get('price_deviation_from_vwap_std', 0.0) > 1e-9:
            if signal_side == 'buy': # Z-score era negativo, SL se ficar MAIS negativo
                sl_based_on_z = session_vwap_create - (z_stop_thresh_create * indicators_dict['price_deviation_from_vwap_std'])
            else: # Z-score era positivo, SL se ficar MAIS positivo
                sl_based_on_z = session_vwap_create + (z_stop_thresh_create * indicators_dict['price_deviation_from_vwap_std'])
        
        sl_based_on_atr = 0.0
        if signal_side == 'buy':
            sl_based_on_atr = current_price_create - (atr_pips_create * atr_sl_mult_create * pip_size_create_sig)
        else:
            sl_based_on_atr = current_price_create + (atr_pips_create * atr_sl_mult_create * pip_size_create_sig)

        # Usar o stop mais conservador (mais próximo da entrada para Z-Score, ou mais distante para ATR se Z-Score SL for muito apertado)
        # Para Z-Score, o SL protege contra a falha da hipótese de reversão.
        # Para compra (Z era negativo): SL_Z < SL_ATR (queremos o maior dos dois valores, pois são < entry)
        # Para venda (Z era positivo): SL_Z > SL_ATR (queremos o menor dos dois valores, pois são > entry)
        final_stop_loss: float # Adicionada tipagem
        if signal_side == 'buy':
            final_stop_loss = max(sl_based_on_z, sl_based_on_atr) if sl_based_on_z != 0.0 else sl_based_on_atr
        else:
            final_stop_loss = min(sl_based_on_z, sl_based_on_atr) if sl_based_on_z != 0.0 else sl_based_on_atr


        # Take Profit: geralmente o VWAP da sessão ou Z-Score próximo de zero.
        target_exit_z = self.parameters['zscore_exit_target_threshold']
        # Preço onde Z-score seria target_exit_z
        if signal_side == 'buy': # Comprado com Z negativo, TP quando Z se aproxima de -target_exit_z (ou seja, preço subiu)
            final_take_profit = session_vwap_create - (target_exit_z * indicators_dict.get('price_deviation_from_vwap_std', atr_pips_create * pip_size_create_sig * 0.5))
        else: # Vendido com Z positivo, TP quando Z se aproxima de +target_exit_z (preço caiu)
            final_take_profit = session_vwap_create + (target_exit_z * indicators_dict.get('price_deviation_from_vwap_std', atr_pips_create * pip_size_create_sig * 0.5))

        # Confiança do sinal
        z_abs = abs(indicators_dict.get('current_zscore_price_vs_vwap', 0.0))
        confidence = 0.5 + (min(z_abs, 3.0) / 3.0) * 0.45 # Escalar Z-score para confiança (max 0.95)
        if self.parameters['use_rsi_filter_zscore']:
            rsi_conf = indicators_dict.get('rsi_value', 50.0)
            if (signal_side == 'buy' and rsi_conf < self.parameters['rsi_oversold_entry_zscore'] + 5) or \
               (signal_side == 'sell' and rsi_conf > self.parameters['rsi_overbought_entry_zscore'] - 5):
                confidence = min(0.95, confidence + 0.05) # Pequeno bônus se RSI confirmar bem


        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=round(np.clip(confidence, 0.5, 0.95), 4),
            entry_price=None, # Entrada a mercado
            stop_loss=round(final_stop_loss, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(final_take_profit, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"ZScoreVWAP Reversion: {signal_side.upper()}. Z: {indicators_dict.get('current_zscore_price_vs_vwap', 0.0):.2f}",
            metadata={
                'zscore_at_entry': indicators_dict.get('current_zscore_price_vs_vwap'),
                'session_vwap_at_entry': session_vwap_create,
                'price_std_dev_vs_vwap': indicators_dict.get('price_deviation_from_vwap_std'),
                'rsi_at_entry': indicators_dict.get('rsi_value'),
                'atr_pips_at_entry': atr_pips_create
            }
        )

    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Avalia condições de saída para a posição ZScoreVWAP."""
        indic = self.current_indicators
        if not indic or indic.get('ticks_in_session_vwap_calc', 0) < self.parameters['min_ticks_for_vwap_calc'] // 2: # Menos ticks para saída
            self.logger.debug(f"Indicadores ZScoreVWAP não válidos para avaliar saída da posição {open_position.id}.")
            return None # Não sair se os indicadores não estiverem prontos

        current_z_exit = indic.get('current_zscore_price_vs_vwap', 0.0) # Renomeado
        target_exit_z_val = self.parameters['zscore_exit_target_threshold'] # Renomeado

        # 1. Saída se Z-Score retornou ao alvo (próximo de zero/VWAP)
        if open_position.side.lower() == 'buy': # Posição comprada (Z era negativo)
            if current_z_exit >= -target_exit_z_val: # Z-score subiu para perto de zero
                return ExitSignal(position_id_to_close=open_position.id,
                                  reason=f"TP Z-Score Atingido: {current_z_exit:.2f} (Alvo >= {-target_exit_z_val:.2f})")
        elif open_position.side.lower() == 'sell': # Posição vendida (Z era positivo)
            if current_z_exit <= target_exit_z_val: # Z-score caiu para perto de zero
                return ExitSignal(position_id_to_close=open_position.id,
                                  reason=f"TP Z-Score Atingido: {current_z_exit:.2f} (Alvo <= {target_exit_z_val:.2f})")

        # 2. Duração máxima da posição
        max_duration_s = self.parameters['max_position_duration_hours'] * 3600
        time_held_s_exit = (datetime.now(timezone.utc) - open_position.open_time).total_seconds() # Renomeado
        if time_held_s_exit > max_duration_s:
            return ExitSignal(position_id_to_close=open_position.id,
                              reason=f"Tempo máximo de holding ({self.parameters['max_position_duration_hours']}h) atingido.")

        # 3. (Opcional) Saída se o VWAP da sessão mudar significativamente (indicando nova dinâmica)
        # if self.session_vwap_history and len(self.session_vwap_history) > 10:
        #     vwap_change_pct = abs(self.session_vwap_history[-1] - np.mean(list(self.session_vwap_history)[-10:-1])) / (self.session_vwap_history[-1] + 1e-9)
        #     if vwap_change_pct > 0.002: # Ex: VWAP mudou mais de 0.2%
        #         return ExitSignal(position_id_to_close=open_position.id, reason="Mudança significativa no VWAP da sessão.")

        return None # Nenhuma condição de saída específica da estratégia atingida