# strategies/momentum/ichimoku_kumo.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados
import talib # Para ATR e outros indicadores se necessário
from typing import Dict, Optional, Any, List, Tuple # Adicionado List, Tuple
from datetime import datetime, timezone # Adicionado timezone

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default, etc.

logger = setup_logger("ichimoku_kumo_strategy_logger") # Nome do logger específico

class IchimokuKumoStrategy(BaseStrategy):
    """
    Estratégia de trading usando a Nuvem Ichimoku (Kumo) e seus componentes.
    Sinais são gerados com base na posição do preço em relação à nuvem,
    cruzamentos Tenkan-sen/Kijun-sen (TK cross), e confirmação da Chikou Span.
    """

    def __init__(self):
        super().__init__("IchimokuKumoTrend") # Nome da estratégia
        self.suitable_regimes = [MarketRegime.TREND] # Ichimoku é primariamente uma estratégia de tendência
        self.min_time_between_signals_sec = 600  # 10 minutos

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Períodos do Ichimoku
            'tenkan_sen_period': 9, # Renomeado
            'kijun_sen_period': 26,  # Renomeado
            'senkou_span_b_period': 52, # Renomeado
            'chikou_span_displacement': 26, # Renomeado (deslocamento para trás do Chikou)
            'kumo_displacement': 26, # Renomeado (deslocamento para frente da Nuvem)

            # Filtros e Condições
            'min_kumo_thickness_pips': 5.0, # Espessura mínima da nuvem em pips
            'use_kumo_breakout_signal': True,
            'use_tk_cross_confirmation': True, # Renomeado
            'use_chikou_span_confirmation': True, # Renomeado
            'filter_on_kumo_twist': True, # Evitar trades durante Kumo Twist

            # Gestão de Risco
            'atr_period_ichimoku': 14, # Renomeado
            'atr_multiplier_sl_ichimoku': 2.0, # Renomeado
            'target_risk_reward_ratio_ichimoku': 2.5, # Renomeado (TP = Risco * RR_ratio)
            'stop_loss_on_opposite_kumo_line': True, # Usar Kijun ou linha oposta da nuvem para SL
            'max_spread_pips_ichimoku_entry': 1.5,
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula todos os componentes do Ichimoku Kinko Hyo e indicadores auxiliares."""
        self.current_indicators = {} # Resetar

        recent_ticks_list = market_context.get('recent_ticks', [])
        # Lookback necessário: max(senkou_b, kijun + kumo_displacement, chikou_displacement)
        # Mais um buffer para TA-Lib e cálculos de médias.
        min_data_len = max(self.parameters['senkou_span_b_period'] + self.parameters['kumo_displacement'],
                           self.parameters['kijun_sen_period'] + self.parameters['kumo_displacement'],
                           self.parameters['chikou_span_displacement'] + self.parameters['kumo_displacement']
                           ) + 20 # Buffer

        if not recent_ticks_list or len(recent_ticks_list) < min_data_len:
            self.logger.debug(f"Dados insuficientes para Ichimoku ({len(recent_ticks_list)}/{min_data_len}).")
            return

        high_prices = self._get_prices_from_context(market_context, 'high')
        low_prices = self._get_prices_from_context(market_context, 'low')
        close_prices = self._get_prices_from_context(market_context, 'mid') # Usar mid como close

        if not (len(close_prices) >= min_data_len and \
                len(high_prices) >= min_data_len and \
                len(low_prices) >= min_data_len) :
            self.logger.debug(f"Arrays de preço insuficientes após extração para Ichimoku.")
            return

        # Calcular Linhas Ichimoku (manualmente, pois TA-Lib não tem Ichimoku completo)
        # Tenkan-sen (Linha de Conversão): (Máxima dos últimos N1 períodos + Mínima dos últimos N1 períodos) / 2
        tenkan_period = self.parameters['tenkan_sen_period']
        tenkan_sen_series = (pd.Series(high_prices).rolling(window=tenkan_period).max() +
                             pd.Series(low_prices).rolling(window=tenkan_period).min()) / 2.0

        # Kijun-sen (Linha Base): (Máxima dos últimos N2 períodos + Mínima dos últimos N2 períodos) / 2
        kijun_period = self.parameters['kijun_sen_period']
        kijun_sen_series = (pd.Series(high_prices).rolling(window=kijun_period).max() +
                            pd.Series(low_prices).rolling(window=kijun_period).min()) / 2.0

        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotado N3 períodos à frente
        # Para decisão no presente, usamos os valores atuais de Tenkan/Kijun para Senkou A que corresponderia ao presente
        # se a nuvem fosse plotada. O valor que se refere ao *presente* é o calculado N3 períodos *atrás*.
        # A nuvem "futura" é usada para identificar suporte/resistência futuros.
        # Para os sinais de "preço vs nuvem", usamos a nuvem que está alinhada com o preço atual.
        # Esta nuvem foi formada por Tenkan/Kijun de `kumo_displacement` períodos atrás.

        kumo_disp = self.parameters['kumo_displacement'] # Renomeado
        
        # Senkou Span A para a nuvem atual (alinhada com o preço atual)
        # (Tenkan de kumo_disp períodos atrás + Kijun de kumo_disp períodos atrás) / 2
        if len(tenkan_sen_series) >= kumo_disp and len(kijun_sen_series) >= kumo_disp and \
           not np.isnan(tenkan_sen_series.iloc[-kumo_disp]) and not np.isnan(kijun_sen_series.iloc[-kumo_disp]):
            senkou_a_current_cloud = (tenkan_sen_series.iloc[-kumo_disp] + kijun_sen_series.iloc[-kumo_disp]) / 2.0
        else: senkou_a_current_cloud = np.nan


        # Senkou Span B para a nuvem atual: (Máxima dos últimos N4 períodos + Mínima dos últimos N4 períodos) / 2,
        # calculado kumo_disp períodos atrás.
        senkou_b_period_val = self.parameters['senkou_span_b_period'] # Renomeado
        if len(high_prices) >= senkou_b_period_val + kumo_disp and len(low_prices) >= senkou_b_period_val + kumo_disp:
            highs_for_sb = pd.Series(high_prices).rolling(window=senkou_b_period_val).max()
            lows_for_sb = pd.Series(low_prices).rolling(window=senkou_b_period_val).min()
            # Pegar o valor de kumo_disp períodos atrás
            senkou_b_current_cloud = (highs_for_sb.iloc[-kumo_disp] + lows_for_sb.iloc[-kumo_disp]) / 2.0 if not (np.isnan(highs_for_sb.iloc[-kumo_disp]) or np.isnan(lows_for_sb.iloc[-kumo_disp])) else np.nan
        else: senkou_b_current_cloud = np.nan


        # Chikou Span (Lagging Span): Preço de fechamento atual, plotado N3 períodos para trás.
        # Para confirmação, comparamos o preço atual (Chikou "real") com o preço de N3 períodos atrás.
        chikou_disp = self.parameters['chikou_span_displacement'] # Renomeado
        chikou_span_current_price = close_prices[-1] # Preço atual
        chikou_span_compare_price = close_prices[-1 - chikou_disp] if len(close_prices) > chikou_disp else np.nan # Preço de N3 períodos atrás

        # Últimos valores de Tenkan e Kijun (para o presente)
        tenkan_val = tenkan_sen_series.iloc[-1] # Renomeado
        kijun_val = kijun_sen_series.iloc[-1]   # Renomeado

        # Nuvem (Kumo) atual
        kumo_top_current, kumo_bottom_current = np.nan, np.nan # Renomeado
        if not np.isnan(senkou_a_current_cloud) and not np.isnan(senkou_b_current_cloud):
            kumo_top_current = max(senkou_a_current_cloud, senkou_b_current_cloud)
            kumo_bottom_current = min(senkou_a_current_cloud, senkou_b_current_cloud)
        
        kumo_thickness_price = kumo_top_current - kumo_bottom_current if not (np.isnan(kumo_top_current) or np.isnan(kumo_bottom_current)) else 0.0 # Renomeado
        pip_size_ichi = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado
        kumo_thickness_pips_val = kumo_thickness_price / pip_size_ichi if kumo_thickness_price > 0 and pip_size_ichi > 0 else 0.0 # Renomeado

        # Direção da Nuvem Atual e Futura (para Kumo Twist)
        kumo_is_bullish_current = senkou_a_current_cloud > senkou_b_current_cloud if not (np.isnan(senkou_a_current_cloud) or np.isnan(senkou_b_current_cloud)) else None # Renomeado
        
        # Nuvem Futura (a que é plotada à frente do preço atual)
        senkou_a_future_cloud = (tenkan_val + kijun_val) / 2.0 if not (np.isnan(tenkan_val) or np.isnan(kijun_val)) else np.nan
        senkou_b_future_cloud_series = (pd.Series(high_prices).rolling(window=senkou_b_period_val).max() +
                                        pd.Series(low_prices).rolling(window=senkou_b_period_val).min()) / 2.0
        senkou_b_future_cloud = senkou_b_future_cloud_series.iloc[-1] if not senkou_b_future_cloud_series.empty and not np.isnan(senkou_b_future_cloud_series.iloc[-1]) else np.nan

        kumo_is_bullish_future = senkou_a_future_cloud > senkou_b_future_cloud if not (np.isnan(senkou_a_future_cloud) or np.isnan(senkou_b_future_cloud)) else None # Renomeado
        kumo_twist_imminent = kumo_is_bullish_current is not None and kumo_is_bullish_future is not None and kumo_is_bullish_current != kumo_is_bullish_future # Renomeado


        # ATR
        atr_price_val_ichi = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.parameters['atr_period_ichimoku'])[-1] # Renomeado
        atr_pips_val_ichi = atr_price_val_ichi / pip_size_ichi if atr_price_val_ichi > 0 and pip_size_ichi > 0 else 0.0 # Renomeado

        # Posição do preço em relação à nuvem atual
        current_price_val_ichi = close_prices[-1] # Renomeado
        price_cloud_position_str: str # Renomeado
        if np.isnan(kumo_top_current) or np.isnan(kumo_bottom_current): price_cloud_position_str = "undefined_kumo"
        elif current_price_val_ichi > kumo_top_current: price_cloud_position_str = 'above_kumo'
        elif current_price_val_ichi < kumo_bottom_current: price_cloud_position_str = 'below_kumo'
        else: price_cloud_position_str = 'inside_kumo'


        self.current_indicators = {
            'tenkan_sen': tenkan_val,
            'kijun_sen': kijun_val,
            'senkou_span_a_current': senkou_a_current_cloud, # Para a nuvem alinhada com o preço
            'senkou_span_b_current': senkou_b_current_cloud, # Para a nuvem alinhada com o preço
            'kumo_top_current': kumo_top_current,
            'kumo_bottom_current': kumo_bottom_current,
            'kumo_thickness_pips': kumo_thickness_pips_val,
            'kumo_is_bullish_current': kumo_is_bullish_current,
            'kumo_is_bullish_future': kumo_is_bullish_future, # Para a nuvem que será plotada à frente
            'is_kumo_twist_imminent': kumo_twist_imminent, # Renomeado
            'chikou_span_price': chikou_span_current_price, # Preço atual (Chikou real)
            'chikou_span_compare_price': chikou_span_compare_price, # Preço de X períodos atrás para comparação
            'current_price_mid': current_price_val_ichi,
            'price_vs_kumo_position': price_cloud_position_str, # Renomeado
            'tk_cross_value': tenkan_val - kijun_val if not (np.isnan(tenkan_val) or np.isnan(kijun_val)) else 0.0, # Renomeado
            'atr_pips': atr_pips_val_ichi,
            'spread_pips': market_context.get('spread', 0.0) / pip_size_ichi
        }

    # _calculate_midpoint, _calculate_future_senkou_a, _calculate_future_senkou_b
    # foram integrados ou substituídos pela lógica em calculate_indicators.

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinais de trading baseados nas condições do Ichimoku."""
        indic = self.current_indicators
        if not indic or np.isnan(indic.get('tenkan_sen', np.nan)) or np.isnan(indic.get('kumo_top_current', np.nan)):
            self.logger.debug("Indicadores Ichimoku não disponíveis ou NaN. Sem sinal.")
            return None

        # Filtro de espessura mínima da nuvem
        if indic.get('kumo_thickness_pips', 0.0) < self.parameters['min_kumo_thickness_pips']:
            self.logger.debug(f"Nuvem Ichimoku muito fina ({indic.get('kumo_thickness_pips'):.1f} pips). Sem sinal.")
            return None

        # Filtro de Kumo Twist Iminente
        if self.parameters['filter_on_kumo_twist'] and indic.get('is_kumo_twist_imminent', False):
            self.logger.debug("Kumo Twist iminente detectado. Sem sinal para evitar instabilidade.")
            return None
        
        # Filtro de Spread
        max_spread = self.parameters['max_spread_pips_ichimoku_entry']
        if indic.get('spread_pips', float('inf')) > max_spread:
            self.logger.debug(f"Spread ({indic.get('spread_pips'):.1f} pips) muito alto para Ichimoku (Max: {max_spread}).")
            return None


        signal_side_to_gen_ichi: Optional[str] = None # Renomeado
        signal_strength_score = 0.0 # Renomeado

        price_pos = indic.get('price_vs_kumo_position', '') # Renomeado
        tk_cross = indic.get('tk_cross_value', 0.0)
        chikou_confirms_bullish = not np.isnan(indic.get('chikou_span_price',np.nan)) and \
                                  not np.isnan(indic.get('chikou_span_compare_price',np.nan)) and \
                                  indic['chikou_span_price'] > indic['chikou_span_compare_price']
        chikou_confirms_bearish = not np.isnan(indic.get('chikou_span_price',np.nan)) and \
                                  not np.isnan(indic.get('chikou_span_compare_price',np.nan)) and \
                                  indic['chikou_span_price'] < indic['chikou_span_compare_price']


        # Sinais de COMPRA (Long)
        # Condição Principal: Preço acima da Nuvem E Nuvem é de Alta (bullish)
        if price_pos == 'above_kumo' and indic.get('kumo_is_bullish_current', False):
            signal_strength_score = 1.0 # Condição base forte
            # Confirmação 1: TK Cross Bullish (Tenkan > Kijun)
            if self.parameters['use_tk_cross_confirmation'] and tk_cross > 0:
                signal_strength_score += 0.5
            # Confirmação 2: Chikou Span Bullish (Chikou acima do preço de X períodos atrás)
            if self.parameters['use_chikou_span_confirmation'] and chikou_confirms_bullish:
                signal_strength_score += 0.5
            
            if signal_strength_score >= 1.5: # Limiar de força para compra (ex: base + pelo menos uma confirmação)
                signal_side_to_gen_ichi = 'buy'

        # Sinais de VENDA (Short)
        # Condição Principal: Preço abaixo da Nuvem E Nuvem é de Baixa (bearish)
        elif price_pos == 'below_kumo' and indic.get('kumo_is_bullish_current', True) is False: # Kumo bearish
            signal_strength_score = 1.0
            if self.parameters['use_tk_cross_confirmation'] and tk_cross < 0: # TK Cross Bearish
                signal_strength_score += 0.5
            if self.parameters['use_chikou_span_confirmation'] and chikou_confirms_bearish:
                signal_strength_score += 0.5

            if signal_strength_score >= 1.5:
                signal_side_to_gen_ichi = 'sell'

        # (Opcional) Sinais de Breakout da Nuvem
        elif self.parameters['use_kumo_breakout_signal'] and price_pos == 'inside_kumo':
            # Esta lógica pode ser complexa: precisa de um estado anterior para saber de onde o preço entrou na nuvem.
            # Ex: Se preço estava ABAIXO e agora entrou e ROMPEU PARA CIMA o Kumo_Top -> Compra
            # Ex: Se preço estava ACIMA e agora entrou e ROMPEU PARA BAIXO o Kumo_Bottom -> Venda
            # Por simplicidade, esta parte é omitida, focando nos sinais mais claros.
            pass


        if signal_side_to_gen_ichi:
            return self._create_ichimoku_trade_signal(signal_side_to_gen_ichi, indic, signal_strength_score, market_context) # Renomeado
        return None


    def _create_ichimoku_trade_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                               signal_strength: float, market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a estratégia Ichimoku."""
        current_price_create_ichi = indicators_dict['current_price_mid'] # Renomeado
        atr_pips_create_ichi = indicators_dict.get('atr_pips', 10.0) # Renomeado
        if atr_pips_create_ichi == 0.0: atr_pips_create_ichi = 10.0
        pip_size_create_ichi = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        sl_atr_mult_ichi = self.parameters['atr_multiplier_sl_ichimoku'] # Renomeado
        rr_ratio_ichi = self.parameters['target_risk_reward_ratio_ichimoku'] # Renomeado
        
        stop_loss_val_ichi: float # Adicionada tipagem
        take_profit_val_ichi: float # Adicionada tipagem

        if signal_side == 'buy':
            # SL pode ser abaixo da Kijun-sen ou da linha inferior da Nuvem (Senkou Span B se bullish, A se bearish)
            sl_ref_kijun = indicators_dict.get('kijun_sen', current_price_create_ichi - (atr_pips_create_ichi * pip_size_create_ichi))
            sl_ref_kumo_line = indicators_dict.get('kumo_bottom_current', sl_ref_kijun) # Linha inferior da nuvem atual
            stop_loss_candidate = min(sl_ref_kijun, sl_ref_kumo_line) - (atr_pips_create_ichi * 0.25 * pip_size_create_ichi) # Buffer abaixo
            
            sl_atr_based = current_price_create_ichi - (atr_pips_create_ichi * sl_atr_mult_ichi * pip_size_create_ichi)
            stop_loss_val_ichi = min(stop_loss_candidate, sl_atr_based) if self.parameters['stop_loss_on_opposite_kumo_line'] else sl_atr_based


            risk_pips_val_ichi = (current_price_create_ichi - stop_loss_val_ichi) / pip_size_create_ichi # Renomeado
            take_profit_val_ichi = current_price_create_ichi + (risk_pips_val_ichi * rr_ratio_ichi * pip_size_create_ichi)
        
        else: # signal_side == 'sell'
            sl_ref_kijun = indicators_dict.get('kijun_sen', current_price_create_ichi + (atr_pips_create_ichi * pip_size_create_ichi))
            sl_ref_kumo_line = indicators_dict.get('kumo_top_current', sl_ref_kijun)
            stop_loss_candidate = max(sl_ref_kijun, sl_ref_kumo_line) + (atr_pips_create_ichi * 0.25 * pip_size_create_ichi)

            sl_atr_based = current_price_create_ichi + (atr_pips_create_ichi * sl_atr_mult_ichi * pip_size_create_ichi)
            stop_loss_val_ichi = max(stop_loss_candidate, sl_atr_based) if self.parameters['stop_loss_on_opposite_kumo_line'] else sl_atr_based

            risk_pips_val_ichi = (stop_loss_val_ichi - current_price_create_ichi) / pip_size_create_ichi
            take_profit_val_ichi = current_price_create_ichi - (risk_pips_val_ichi * rr_ratio_ichi * pip_size_create_ichi)

        # Confiança baseada na força do sinal (quantas confirmações ativaram)
        # signal_strength já vem como float (ex: 1.0, 1.5, 2.0, 2.5)
        # Normalizar para 0.5 - 1.0
        # Max score possível é 1.0 (base) + 0.5 (TK) + 0.5 (Chikou) + 0.5 (nuvem na direção) = 2.5 no código original.
        # A lógica de signal_strength no generate_signal foi ajustada, então aqui usamos o valor passado.
        # Max score agora pode ser 1 (base) + 0.5 (TK) + 0.5 (Chikou) = 2.0 (se nuvem já está na condição base)
        # Mapear score de 1.0-2.5 para confiança de 0.5-0.95
        confidence_val_ichi = 0.5 + (min(signal_strength, 2.5) - 1.0) / (2.5 - 1.0) * 0.45 if signal_strength > 1.0 else 0.5 # Renomeado

        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=round(np.clip(confidence_val_ichi, 0.5, 0.95), 4),
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val_ichi, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val_ichi, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"Ichimoku {signal_side.upper()}. Força: {signal_strength:.1f}. Preço vs Nuvem: {indicators_dict.get('price_vs_kumo_position')}",
            metadata={
                'signal_strength_score': signal_strength,
                'price_vs_kumo_pos': indicators_dict.get('price_vs_kumo_position'),
                'kumo_thickness_pips': indicators_dict.get('kumo_thickness_pips'),
                'tk_cross_val': indicators_dict.get('tk_cross_value'),
                'kumo_is_bullish_now': indicators_dict.get('kumo_is_bullish_current'),
                'chikou_confirms_bullish': chikou_confirms_bullish, # Adicionado
                'chikou_confirms_bearish': chikou_confirms_bearish, # Adicionado
                'atr_pips_at_signal': atr_pips_create_ichi
            }
        )

    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Define condições de saída para a estratégia Ichimoku."""
        indic = self.current_indicators
        if not indic or np.isnan(indic.get('kumo_top_current', np.nan)): # Checar se nuvem foi calculada
            return None

        current_price_exit_ichi = market_context['tick'].mid # Renomeado

        # 1. Saída se preço cruzar para o lado OPOSTO da nuvem atual
        price_pos_exit = indic.get('price_vs_kumo_position', '') # Renomeado
        if open_position.side.lower() == 'buy' and price_pos_exit == 'below_kumo':
            return ExitSignal(position_id_to_close=open_position.id, reason="Saída Ichimoku: Preço cruzou abaixo da Nuvem Kumo.")
        elif open_position.side.lower() == 'sell' and price_pos_exit == 'above_kumo':
            return ExitSignal(position_id_to_close=open_position.id, reason="Saída Ichimoku: Preço cruzou acima da Nuvem Kumo.")

        # 2. Saída se TK Cross (Tenkan-Kijun) virar contra a posição
        tk_cross_exit = indic.get('tk_cross_value', 0.0) # Renomeado
        # Adicionar um pequeno buffer para evitar saídas por ruído no cruzamento exato
        tk_buffer = indic.get('atr_pips', 1.0) * 0.05 * (0.0001 if "JPY" not in CONFIG.SYMBOL else 0.01) # 5% do ATR em valor de preço
        if open_position.side.lower() == 'buy' and tk_cross_exit < -tk_buffer: # TK Cross virou bearish
            return ExitSignal(position_id_to_close=open_position.id, reason=f"Saída Ichimoku: TK Cross Bearish (Valor: {tk_cross_exit:.5f}).")
        elif open_position.side.lower() == 'sell' and tk_cross_exit > tk_buffer: # TK Cross virou bullish
            return ExitSignal(position_id_to_close=open_position.id, reason=f"Saída Ichimoku: TK Cross Bullish (Valor: {tk_cross_exit:.5f}).")

        # 3. (Opcional) Trailing stop usando Kijun-sen
        # Esta lógica é mais para ATUALIZAR o SL da posição, não para gerar um ExitSignal direto,
        # a menos que a estratégia queira fechar IMEDIATAMENTE se Kijun for violada.
        # Se o SL da posição for dinamicamente atualizado para Kijun pela BaseStrategy/ExecutionEngine,
        # o sistema de stops normal lidará com isso.
        # if self.parameters.get('use_kijun_trailing_stop', True):
        #     kijun_val_exit = indic.get('kijun_sen')
        #     if kijun_val_exit is not None:
        #         if open_position.side.lower() == 'buy' and current_price_exit_ichi < kijun_val_exit:
        #             return ExitSignal(position_id_to_close=open_position.id, reason="Saída Ichimoku: Preço abaixo da Kijun-sen (Trailing).")
        #         elif open_position.side.lower() == 'sell' and current_price_exit_ichi > kijun_val_exit:
        #             return ExitSignal(position_id_to_close=open_position.id, reason="Saída Ichimoku: Preço acima da Kijun-sen (Trailing).")

        return None