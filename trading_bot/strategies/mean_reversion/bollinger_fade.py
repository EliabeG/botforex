# strategies/mean_reversion/bollinger_fade.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados se necessário
import talib # Para Bollinger Bands, RSI, ATR
from typing import Dict, Optional, Any, List # Adicionado List
from datetime import datetime, timezone # Adicionado timezone
from collections import deque

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default

logger = setup_logger("bollinger_fade_strategy") # Nome do logger específico

class BollingerFadeStrategy(BaseStrategy):
    """
    Estratégia de "fade" (reversão à média) usando Bandas de Bollinger.
    Entra contra o movimento quando o preço toca ou ultrapassa as bandas,
    com confirmações de %B, RSI e outros.
    """

    def __init__(self):
        super().__init__("BollingerBandFadeReversal") # Nome da estratégia mais descritivo
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY] # Onde fades são mais prováveis
        self.min_time_between_signals_sec = 180  # 3 minutos

        # Histórico para análise de padrões (ex: múltiplos toques, squeeze)
        self.band_touch_events: deque[Dict[str, Any]] = deque(maxlen=20) # Renomeado e aumentado buffer
        self.squeeze_active_history: deque[bool] = deque(maxlen=50) # Renomeado

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Parâmetros das Bandas de Bollinger
            'bb_period': 20,
            'bb_std_dev_entry': 2.0, # Desvio padrão para entrada
            'bb_std_dev_extreme_sl': 2.5, # Desvio padrão para SL ou entrada mais agressiva

            # Limites de Largura da Banda (em % do preço médio)
            'min_band_width_percent': 0.08,  # 0.08% do preço médio (ex: ~8 pips para EURUSD 1.0)
            'max_band_width_percent': 0.30,  # 0.30% do preço médio

            # Parâmetros de %B (Percent B)
            'percent_b_oversold_threshold': 0.05, # Abaixo de 0.05 (ou 0.0)
            'percent_b_overbought_threshold': 0.95, # Acima de 0.95 (ou 1.0)

            # Squeeze de Volatilidade
            'use_bollinger_squeeze_logic': True, # Renomeado
            'squeeze_lookback_periods': 20, # Períodos para calcular largura de Keltner/BB para squeeze
            'keltner_atr_period_squeeze': 10,
            'keltner_atr_multiplier_squeeze': 1.5,
            # 'squeeze_threshold_ratio': 0.8, # BBWidth < KeltnerWidth * ratio

            # Confirmações e Filtros
            'min_consecutive_touches_for_signal': 1, # Renomeado
            'rsi_period_confirm': 14, # Renomeado
            'rsi_oversold_confirm': 30, # Renomeado
            'rsi_overbought_confirm': 70, # Renomeado
            'atr_period_stops': 14,
            'atr_multiplier_sl_fade': 1.0, # SL mais apertado para fades
            'target_risk_reward_ratio_fade': 1.5, # R:R para fades pode ser menor
            'exit_on_middle_band_touch': True, # Renomeado
            'min_price_momentum_against_touch_pct': 0.05 # Ex: 0.05% de momentum contra a banda tocada
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula Bandas de Bollinger, RSI, ATR e outros indicadores relevantes."""
        recent_ticks_list = market_context.get('recent_ticks', [])
        # Precisa de dados suficientes para o período BB + período RSI + período ATR + lookback do squeeze
        min_data_len = max(self.parameters['bb_period'],
                           self.parameters['rsi_period_confirm'],
                           self.parameters['atr_period_stops'],
                           self.parameters.get('squeeze_lookback_periods', 20) # Default se não existir
                           ) + 20 # Buffer adicional

        if not recent_ticks_list or len(recent_ticks_list) < min_data_len:
            self.logger.debug(f"Dados insuficientes para indicadores da BollingerFade ({len(recent_ticks_list)}/{min_data_len}).")
            self.current_indicators = {}
            return

        # Preparar arrays de preço para TA-Lib
        close_prices_arr = self._get_prices_from_context(market_context, 'mid') # Usar helper da BaseStrategy
        high_prices_arr = self._get_prices_from_context(market_context, 'high')
        low_prices_arr = self._get_prices_from_context(market_context, 'low')

        if not (len(close_prices_arr) >= min_data_len and \
                len(high_prices_arr) >= min_data_len and \
                len(low_prices_arr) >= min_data_len) :
            self.current_indicators = {}
            return


        # Bollinger Bands (Padrão e Extrema)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices_arr,
                                                     timeperiod=self.parameters['bb_period'],
                                                     nbdevup=self.parameters['bb_std_dev_entry'],
                                                     nbdevdn=self.parameters['bb_std_dev_entry'],
                                                     matype=0) # SMA
        
        bb_upper_extreme, _, bb_lower_extreme = talib.BBANDS(close_prices_arr, # _ para middle não usado
                                                             timeperiod=self.parameters['bb_period'],
                                                             nbdevup=self.parameters['bb_std_dev_extreme_sl'],
                                                             nbdevdn=self.parameters['bb_std_dev_extreme_sl'],
                                                             matype=0)
        # Pegar os últimos valores
        up_band, mid_band, low_band = bb_upper[-1], bb_middle[-1], bb_lower[-1]
        up_extreme_band, low_extreme_band = bb_upper_extreme[-1], bb_lower_extreme[-1]


        # RSI
        rsi_val = talib.RSI(close_prices_arr, timeperiod=self.parameters['rsi_period_confirm'])[-1]

        # ATR
        atr_val_price = talib.ATR(high_prices_arr, low_prices_arr, close_prices_arr,
                                  timeperiod=self.parameters['atr_period_stops'])[-1]
        pip_size_indic = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01
        atr_pips_val = atr_val_price / pip_size_indic if atr_val_price > 0 and pip_size_indic > 0 else 0.0


        # Band Width e %B
        band_width_price = up_band - low_band # Renomeado
        band_width_pct_val = (band_width_price / (mid_band + 1e-9)) * 100 if mid_band > 0 else 0.0 # Renomeado
        percent_b_val = (close_prices_arr[-1] - low_band) / (band_width_price + 1e-9) if band_width_price > 0 else 0.5 # Renomeado

        # Squeeze (Bollinger Bands dentro de Keltner Channels)
        is_in_squeeze = False # Renomeado
        if self.parameters['use_bollinger_squeeze_logic']:
            keltner_atr = talib.ATR(high_prices_arr, low_prices_arr, close_prices_arr,
                                    timeperiod=self.parameters['keltner_atr_period_squeeze'])
            keltner_mult = self.parameters['keltner_atr_multiplier_squeeze']
            keltner_upper = talib.EMA(close_prices_arr, self.parameters['bb_period']) + (keltner_atr * keltner_mult) # Keltner usa EMA
            keltner_lower = talib.EMA(close_prices_arr, self.parameters['bb_period']) - (keltner_atr * keltner_mult)
            
            # Squeeze se BB estiverem dentro dos Keltner Channels
            is_in_squeeze = (up_band <= keltner_upper[-1] and low_band >= keltner_lower[-1]) if len(keltner_upper)>0 else False

        self.squeeze_active_history.append(is_in_squeeze)


        # Análise de toques nas bandas
        current_price_val = close_prices_arr[-1] # Renomeado
        band_touch_type = self._get_band_touch_type(current_price_val, up_band, low_band, up_extreme_band, low_extreme_band) # Renomeado
        if band_touch_type:
            self.band_touch_events.append({
                'type': band_touch_type, 'price': current_price_val, 'timestamp': datetime.now(timezone.utc)
            })
        
        consecutive_upper_touches = self._count_recent_consecutive_touches('upper') # Renomeado
        consecutive_lower_touches = self._count_recent_consecutive_touches('lower') # Renomeado

        # Momentum de preço (ROC para os últimos 5 períodos)
        price_mom_pct = talib.ROC(close_prices_arr, timeperiod=5)[-1] / 100.0 if len(close_prices_arr) >= 6 else 0.0 # Renomeado


        self.current_indicators = {
            'upper_band_entry': up_band, 'middle_band': mid_band, 'lower_band_entry': low_band, # Renomeado
            'upper_band_extreme': up_extreme_band, 'lower_band_extreme': low_extreme_band, # Renomeado
            'band_width_price': band_width_price,
            'band_width_percent': band_width_pct_val,
            'percent_b': percent_b_val,
            'is_in_squeeze': is_in_squeeze,
            'squeeze_periods_count': sum(self.squeeze_active_history), # Renomeado
            'rsi_value': rsi_val, # Renomeado
            'atr_pips': atr_pips_val,
            'current_price_mid': current_price_val,
            'price_momentum_short_term_pct': price_mom_pct, # Renomeado
            'consecutive_upper_band_touches': consecutive_upper_touches, # Renomeado
            'consecutive_lower_band_touches': consecutive_lower_touches, # Renomeado
            'last_band_touch_type': band_touch_type, # Renomeado
            'spread_pips': market_context.get('spread', 0.0) / pip_size_indic
        }

    def _get_band_touch_type(self, price: float, upper: float, lower: float, upper_ext: float, lower_ext: float) -> Optional[str]: # Renomeado
        """Analisa se o preço tocou ou ultrapassou bandas, retornando o tipo de toque."""
        # Adicionar uma pequena tolerância em pips para o toque
        pip_size_touch = 0.0001 if "JPY" not in CONFIG.SYMBOL.upper() else 0.01
        touch_tolerance = 0.2 * pip_size_touch # 0.2 pips de tolerância

        if price >= upper_ext - touch_tolerance: return 'upper_extreme_touch'
        if price >= upper - touch_tolerance: return 'upper_entry_touch' # Renomeado
        if price <= lower_ext + touch_tolerance: return 'lower_extreme_touch'
        if price <= lower + touch_tolerance: return 'lower_entry_touch' # Renomeado
        return None

    def _count_recent_consecutive_touches(self, band_prefix: str) -> int: # Renomeado
        """Conta toques consecutivos recentes em uma banda (upper ou lower)."""
        if not self.band_touch_events: return 0
        count = 0
        for touch_event in reversed(self.band_touch_events): # Renomeado
            if touch_event['type'] and touch_event['type'].startswith(band_prefix):
                count += 1
            else:
                break # Sequência quebrada
        return count


    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinais de fade (reversão) nas Bandas de Bollinger."""
        indic = self.current_indicators
        if not indic or not indic.get('upper_band_entry'): # Checar se indicadores foram calculados
            return None

        # Filtro de largura da banda (em %)
        min_width_pct = self.parameters['min_band_width_percent']
        max_width_pct = self.parameters['max_band_width_percent']
        current_bw_pct = indic.get('band_width_percent', 0.0)

        if not (min_width_pct <= current_bw_pct <= max_width_pct):
            self.logger.debug(f"Largura da banda ({current_bw_pct:.2f}%) fora dos limites [{min_width_pct:.2f}% - {max_width_pct:.2f}%]. Sem sinal.")
            return None

        signal_side_to_gen_bb: Optional[str] = None # Renomeado
        base_signal_strength = 0.0 # Renomeado

        # Lógica de Squeeze Breakout (se habilitada e squeeze ativo por N períodos)
        # Esta é uma lógica de BREAKOUT, não FADE. Se for mantida, precisa ser clara.
        # if self.parameters['use_bollinger_squeeze_logic'] and indic.get('squeeze_periods_count',0) > 10:
        #     if indic.get('percent_b', 0.5) > 1.05: # Breakout para cima (acima da banda superior)
        #         signal_side_to_gen_bb = 'buy'
        #         base_signal_strength = 0.7 # Confiança base para breakout de squeeze
        #     elif indic.get('percent_b', 0.5) < -0.05: # Breakout para baixo (abaixo da banda inferior)
        #         signal_side_to_gen_bb = 'sell'
        #         base_signal_strength = 0.7
        #     if signal_side_to_gen_bb:
        #         self.logger.info(f"Sinal de SQUEEZE BREAKOUT {signal_side_to_gen_bb.upper()} detectado.")
        #         # Para breakout, o SL/TP seria diferente da lógica de fade.
        #         # Esta parte precisa ser separada ou removida se a estratégia é puramente fade.
        #         # Por ora, vou focar na lógica de FADE.

        # Lógica de FADE (Reversão)
        last_touch = indic.get('last_band_touch_type') # Renomeado
        if last_touch and not signal_side_to_gen_bb: # Apenas se não for breakout de squeeze
            current_price = indic['current_price_mid']
            percent_b_val_gs = indic.get('percent_b', 0.5) # Renomeado
            rsi_val_gs = indic.get('rsi_value', 50.0) # Renomeado
            consecutive_touches_val: int # Adicionada tipagem

            if 'upper' in last_touch: # Toque na banda superior -> potencial VENDA (fade)
                base_signal_strength = 0.6
                consecutive_touches_val = indic.get('consecutive_upper_band_touches', 0)
                if percent_b_val_gs >= self.parameters['percent_b_overbought_threshold']: base_signal_strength += 0.1
                if rsi_val_gs > self.parameters['rsi_overbought_confirm']: base_signal_strength += 0.1
                if consecutive_touches_val >= self.parameters['min_consecutive_touches_for_signal']: base_signal_strength += 0.1
                if 'extreme' in last_touch: base_signal_strength += 0.15 # Toque na banda extrema
                # Checar se o preço começou a reverter (momentum contra o toque)
                if indic.get('price_momentum_short_term_pct', 0.0) < -self.parameters['min_price_momentum_against_touch_pct']:
                    base_signal_strength += 0.1
                
                if base_signal_strength >= 0.75: # Limiar de força para gerar sinal
                    signal_side_to_gen_bb = 'sell'

            elif 'lower' in last_touch: # Toque na banda inferior -> potencial COMPRA (fade)
                base_signal_strength = 0.6
                consecutive_touches_val = indic.get('consecutive_lower_band_touches', 0)
                if percent_b_val_gs <= self.parameters['percent_b_oversold_threshold']: base_signal_strength += 0.1
                if rsi_val_gs < self.parameters['rsi_oversold_confirm']: base_signal_strength += 0.1
                if consecutive_touches_val >= self.parameters['min_consecutive_touches_for_signal']: base_signal_strength += 0.1
                if 'extreme' in last_touch: base_signal_strength += 0.15
                if indic.get('price_momentum_short_term_pct', 0.0) > self.parameters['min_price_momentum_against_touch_pct']:
                    base_signal_strength += 0.1

                if base_signal_strength >= 0.75:
                    signal_side_to_gen_bb = 'buy'

        if signal_side_to_gen_bb:
            final_confidence = round(np.clip(base_signal_strength, 0.5, 0.95), 4)
            return self._create_bollinger_fade_signal(signal_side_to_gen_bb, indic, final_confidence, market_context) # Renomeado
        return None


    def _create_bollinger_fade_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                                final_confidence: float, market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a estratégia Bollinger Fade."""
        current_price_entry = indicators_dict['current_price_mid'] # Renomeado
        atr_pips_val_sl = indicators_dict.get('atr_pips', 10.0) # Renomeado
        if atr_pips_val_sl == 0.0: atr_pips_val_sl = 10.0 # Evitar ATR zero
        pip_size_create = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        sl_atr_mult_fade = self.parameters['atr_multiplier_sl_fade'] # Renomeado
        sl_distance_pips = atr_pips_val_sl * sl_atr_mult_fade

        stop_loss_val_create: float # Adicionada tipagem
        take_profit_val_create: float # Adicionada tipagem

        if signal_side == 'buy': # Fade de toque na banda inferior
            # SL um pouco abaixo da banda inferior extrema ou ATR
            stop_loss_val_create = min(indicators_dict['lower_band_extreme'] - (pip_size_create * 2), # 2 pips abaixo da extrema
                                 current_price_entry - (sl_distance_pips * pip_size_create))
            # TP na média móvel central ou R:R
            if self.parameters['exit_on_middle_band_touch']:
                take_profit_val_create = indicators_dict['middle_band']
            else:
                risk_pips_val_create = (current_price_entry - stop_loss_val_create) / pip_size_create # Renomeado
                take_profit_val_create = current_price_entry + (risk_pips_val_create * self.parameters['target_risk_reward_ratio_fade'] * pip_size_create)
        
        else:  # signal_side == 'sell', Fade de toque na banda superior
            stop_loss_val_create = max(indicators_dict['upper_band_extreme'] + (pip_size_create * 2),
                                 current_price_entry + (sl_distance_pips * pip_size_create))
            if self.parameters['exit_on_middle_band_touch']:
                take_profit_val_create = indicators_dict['middle_band']
            else:
                risk_pips_val_create = (stop_loss_val_create - current_price_entry) / pip_size_create
                take_profit_val_create = current_price_entry - (risk_pips_val_create * self.parameters['target_risk_reward_ratio_fade'] * pip_size_create)

        # Garantir que TP não seja igual ou pior que entry
        if signal_side == 'buy' and take_profit_val_create <= current_price_entry:
            take_profit_val_create = current_price_entry + (atr_pips_val_sl * 0.5 * pip_size_create) # Mínimo TP
        elif signal_side == 'sell' and take_profit_val_create >= current_price_entry:
            take_profit_val_create = current_price_entry - (atr_pips_val_sl * 0.5 * pip_size_create)


        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=final_confidence,
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val_create, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val_create, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"Bollinger Fade {signal_side.upper()}. %B: {indicators_dict.get('percent_b', 0.0):.2f}, RSI: {indicators_dict.get('rsi_value', 0.0):.1f}",
            metadata={
                'percent_b': indicators_dict.get('percent_b'),
                'band_width_percent': indicators_dict.get('band_width_percent'),
                'rsi_at_signal': indicators_dict.get('rsi_value'),
                'last_touch_type': indicators_dict.get('last_band_touch_type'),
                'consecutive_touches': indicators_dict.get(f'consecutive_{indicators_dict.get("last_band_touch_type","none").split("_")[0]}_band_touches',0),
                'atr_pips_at_signal': atr_pips_val_sl
            }
        )

    # _calculate_bollinger_confidence foi integrado na lógica de final_confidence em generate_signal

    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Condições de saída para a estratégia Bollinger Fade."""
        indic = self.current_indicators
        if not indic or not indic.get('middle_band'): # Checar se 'middle_band' existe
            return None

        current_price_exit = market_context['tick'].mid # Renomeado

        # 1. Saída na Média Móvel Central (se configurado)
        if self.parameters['exit_on_middle_band_touch']:
            if open_position.side.lower() == 'buy' and current_price_exit >= indic['middle_band']:
                return ExitSignal(position_id_to_close=open_position.id, reason="Preço atingiu a banda média (TP)")
            elif open_position.side.lower() == 'sell' and current_price_exit <= indic['middle_band']:
                return ExitSignal(position_id_to_close=open_position.id, reason="Preço atingiu a banda média (TP)")

        # 2. Saída se tocar a banda oposta (se não saiu na média)
        #    Isso pode ser um stop ou um sinal de que a reversão falhou e o preço continuou.
        last_touch_exit = indic.get('last_band_touch_type') # Renomeado
        if last_touch_exit:
            if open_position.side.lower() == 'buy' and 'upper' in last_touch_exit: # Comprou no toque inferior, mas agora tocou superior
                return ExitSignal(position_id_to_close=open_position.id, reason="Preço tocou a banda oposta (superior).")
            elif open_position.side.lower() == 'sell' and 'lower' in last_touch_exit: # Vendeu no toque superior, mas agora tocou inferior
                return ExitSignal(position_id_to_close=open_position.id, reason="Preço tocou a banda oposta (inferior).")

        # 3. Saída se %B "normalizar" demais contra a direção da posição
        #    Ex: Se comprou (%B baixo), sair se %B subir muito (ex: > 0.8) sem atingir TP.
        percent_b_exit = indic.get('percent_b', 0.5) # Renomeado
        if open_position.side.lower() == 'buy' and percent_b_exit >= self.parameters.get('percent_b_exit_buy_threshold', 0.85): # Ex: 0.85
            return ExitSignal(position_id_to_close=open_position.id, reason=f"%B ({percent_b_exit:.2f}) normalizou contra posição de compra.")
        elif open_position.side.lower() == 'sell' and percent_b_exit <= self.parameters.get('percent_b_exit_sell_threshold', 0.15): # Ex: 0.15
            return ExitSignal(position_id_to_close=open_position.id, reason=f"%B ({percent_b_exit:.2f}) normalizou contra posição de venda.")
        
        # Adicionar saída por tempo se a posição não se mover
        max_hold_seconds_bb = self.parameters.get('max_hold_duration_seconds_bb', 4 * 3600) # Ex: 4 horas
        time_held_bb = (datetime.now(timezone.utc) - open_position.open_time).total_seconds()
        if time_held_bb > max_hold_seconds_bb:
             return ExitSignal(position_id_to_close=open_position.id, reason=f"Tempo máximo de holding ({max_hold_seconds_bb/3600:.0f}h) para BollingerFade atingido.")


        return None