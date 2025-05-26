# strategies/momentum/ema_stack.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados
import talib # Para EMA, ATR, RSI, ADX
from typing import Dict, Optional, Any, List # Adicionado List
from datetime import datetime, timezone # Adicionado timezone

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default, etc.
# Importar TickData se for usado explicitamente nos tipos de market_context
from api.ticktrader_ws import TickData

logger = setup_logger("ema_stack_strategy_logger") # Nome do logger específico

class EMAStackStrategy(BaseStrategy):
    """
    Estratégia de momentum usando um conjunto (stack) de Médias Móveis Exponenciais (EMAs).
    Entra quando as EMAs estão alinhadas (ex: EMA curta > EMA média > EMA longa para compra)
    e o preço confirma a direção do momentum.
    """

    def __init__(self):
        super().__init__("EMAStackTrendFollow_8_21_50") # Nome mais descritivo
        self.suitable_regimes = [MarketRegime.TREND] # Ideal para mercados em tendência
        self.min_time_between_signals_sec = 300  # 5 minutos

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Períodos das EMAs
            'ema_fast_period': 8, # Renomeado
            'ema_medium_period': 21, # Renomeado
            'ema_slow_period': 50, # Renomeado

            # Gestão de Risco e TP
            'atr_period_stops_ema': 14, # Renomeado
            'atr_multiplier_sl_ema': 2.0, # Renomeado
            'target_risk_reward_ratio_ema': 2.5, # Renomeado (TP = Risco * RR_ratio)

            # Filtros de Confirmação
            'min_ema_stack_separation_pct': 0.0002, # Ex: 0.02% de separação mínima entre EMAs (2 pips em 1.0000)
            'min_adx_trend_strength_ema': 25, # Renomeado
            'rsi_filter_enabled_ema': True, # Renomeado
            'rsi_max_entry_bullish_ema': 70, # Não entrar comprado se RSI > 70
            'rsi_min_entry_bearish_ema': 30, # Não entrar vendido se RSI < 30
            # 'min_volume_ratio_ema': 0.8, # Se for usar filtro de volume (requer dados de volume)
            'max_spread_pips_ema_entry': 1.2, # Renomeado

            # Gestão de Posição
            'use_ema_fast_as_trailing_stop': True, # Usar EMA rápida como trailing stop dinâmico
            # 'trailing_stop_activation_pips_ema': 20, # Se não usar EMA como TS
            # 'exit_on_ema_cross_against': True, # Sair se EMA rápida cruzar a média contra a posição
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula EMAs, ATR, RSI, ADX e verifica condições de alinhamento."""
        self.current_indicators = {} # Resetar

        recent_ticks_list = market_context.get('recent_ticks', [])
        # Lookback necessário é o da EMA mais longa + ADX/RSI
        min_data_len = max(self.parameters['ema_slow_period'],
                           self.parameters['atr_period_stops_ema'],
                           self.parameters.get('rsi_period_ema', 14), # Default para RSI
                           self.parameters.get('adx_period_ema', 14)  # Default para ADX
                           ) + 50 # Buffer para aquecimento

        if not recent_ticks_list or len(recent_ticks_list) < min_data_len:
            self.logger.debug(f"Dados insuficientes para indicadores EMA_Stack ({len(recent_ticks_list)}/{min_data_len}).")
            return

        close_prices = self._get_prices_from_context(market_context, 'mid')
        high_prices = self._get_prices_from_context(market_context, 'high')
        low_prices = self._get_prices_from_context(market_context, 'low')

        if not (len(close_prices) >= min_data_len and \
                len(high_prices) >= min_data_len and \
                len(low_prices) >= min_data_len) :
            self.logger.debug(f"Arrays de preço insuficientes após extração para EMA_Stack.")
            return

        # Calcular EMAs
        ema_fast_val = talib.EMA(close_prices, timeperiod=self.parameters['ema_fast_period'])[-1] # Renomeado
        ema_medium_val = talib.EMA(close_prices, timeperiod=self.parameters['ema_medium_period'])[-1] # Renomeado
        ema_slow_val = talib.EMA(close_prices, timeperiod=self.parameters['ema_slow_period'])[-1] # Renomeado

        # Inclinação das EMAs (diferença simples dos últimos N valores da EMA)
        # TA-Lib não calcula inclinação diretamente. Pode ser feito com diff em pd.Series.
        # Exemplo para ema_fast_slope:
        ema_fast_series = talib.EMA(close_prices, timeperiod=self.parameters['ema_fast_period'])
        ema_fast_slope_val = (ema_fast_series[-1] - ema_fast_series[-6]) / 5.0 if len(ema_fast_series) >= 6 and not np.isnan(ema_fast_series[-1]) and not np.isnan(ema_fast_series[-6]) else 0.0 # Renomeado

        # ATR
        atr_price_val_ema = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.parameters['atr_period_stops_ema'])[-1] # Renomeado
        pip_size_ema = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado
        atr_pips_val_ema = atr_price_val_ema / pip_size_ema if atr_price_val_ema > 0 and pip_size_ema > 0 else 0.0 # Renomeado
        
        # RSI
        rsi_val_ema = talib.RSI(close_prices, timeperiod=self.parameters.get('rsi_period_ema', 14))[-1] # Renomeado
        if np.isnan(rsi_val_ema): rsi_val_ema = 50.0

        # ADX
        adx_val_ema = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.parameters.get('adx_period_ema', 14))[-1] # Renomeado
        if np.isnan(adx_val_ema): adx_val_ema = 20.0 # Default se NaN (ADX baixo)

        # Verificar alinhamento e separação das EMAs
        emas_aligned_bullish = False
        emas_aligned_bearish = False
        min_sep_abs = self.parameters['min_ema_stack_separation_pct'] * close_prices[-1] # Separação em valor de preço

        if not np.isnan(ema_fast_val) and not np.isnan(ema_medium_val) and not np.isnan(ema_slow_val):
            if ema_fast_val > ema_medium_val + min_sep_abs and ema_medium_val > ema_slow_val + min_sep_abs:
                emas_aligned_bullish = True
            if ema_fast_val < ema_medium_val - min_sep_abs and ema_medium_val < ema_slow_val - min_sep_abs:
                emas_aligned_bearish = True


        self.current_indicators = {
            'ema_fast': ema_fast_val,
            'ema_medium': ema_medium_val,
            'ema_slow': ema_slow_val,
            'ema_fast_slope_5p': ema_fast_slope_val, # Inclinação da EMA rápida nos últimos 5 períodos
            'emas_aligned_bullish': emas_aligned_bullish,
            'emas_aligned_bearish': emas_aligned_bearish,
            'current_price_mid': close_prices[-1],
            'atr_pips': atr_pips_val_ema,
            'rsi_value': rsi_val_ema,
            'adx_value': adx_val_ema,
            'spread_pips': market_context.get('spread', 0.0) / pip_size_ema
        }

    # _calculate_adx foi removido, usar TA-Lib.

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de trading com base no alinhamento das EMAs e filtros."""
        indic = self.current_indicators
        if not indic or np.isnan(indic.get('ema_fast', np.nan)): # Checar se EMAs foram calculadas
            return None

        # Aplicar filtros básicos de entrada
        if not self._passes_ema_stack_entry_filters(indic, market_context.get('regime')): # Renomeado
            return None

        current_price = indic['current_price_mid']
        signal_side_to_gen_es: Optional[str] = None # Renomeado

        # Setup de COMPRA (Long)
        if indic.get('emas_aligned_bullish', False) and current_price > indic['ema_fast']: # Preço acima da EMA mais rápida
            if self.parameters['rsi_filter_enabled_ema']:
                if indic.get('rsi_value', 50.0) >= self.parameters['rsi_max_entry_bullish_ema']: # RSI já sobrecomprado
                    self.logger.debug(f"EMA_Stack BUY ignorado: RSI ({indic.get('rsi_value'):.1f}) muito alto.")
                    return None
            signal_side_to_gen_es = 'buy'

        # Setup de VENDA (Short)
        elif indic.get('emas_aligned_bearish', False) and current_price < indic['ema_fast']: # Preço abaixo da EMA mais rápida
            if self.parameters['rsi_filter_enabled_ema']:
                if indic.get('rsi_value', 50.0) <= self.parameters['rsi_min_entry_bearish_ema']: # RSI já sobrevendido
                    self.logger.debug(f"EMA_Stack SELL ignorado: RSI ({indic.get('rsi_value'):.1f}) muito baixo.")
                    return None
            signal_side_to_gen_es = 'sell'

        if signal_side_to_gen_es:
            return self._create_ema_stack_signal(signal_side_to_gen_es, indic, market_context) # Renomeado
        return None


    def _passes_ema_stack_entry_filters(self, indicators_dict: Dict[str, Any], current_market_regime: Optional[str]) -> bool: # Renomeado
        """Verifica filtros básicos (spread, ADX, regime) antes de gerar sinal EMA Stack."""
        # Spread máximo
        max_spread = self.parameters['max_spread_pips_ema_entry']
        if indicators_dict.get('spread_pips', float('inf')) > max_spread:
            self.logger.debug(f"Filtro Spread EMA_Stack não passou: {indicators_dict.get('spread_pips'):.1f} > {max_spread:.1f} pips.")
            return False

        # ADX mínimo (força da tendência)
        min_adx = self.parameters['min_adx_trend_strength_ema']
        if indicators_dict.get('adx_value', 0.0) < min_adx:
            self.logger.debug(f"Filtro ADX EMA_Stack não passou: {indicators_dict.get('adx_value'):.1f} < {min_adx:.1f}.")
            return False

        # Regime de mercado (deve ser TREND)
        if current_market_regime not in self.suitable_regimes: # suitable_regimes definido no __init__
            self.logger.debug(f"Filtro Regime EMA_Stack não passou: Regime atual '{current_market_regime}' não é adequado.")
            return False
        
        # (Opcional) Filtro de horário global da BaseStrategy pode ser chamado aqui se necessário
        # if not self.get_time_filter_for_strategy(datetime.now(timezone.utc).hour):
        #     return False
        return True


    def _create_ema_stack_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                               market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a estratégia EMA Stack."""
        current_price_cs = indicators_dict['current_price_mid'] # Renomeado
        atr_pips_cs_ema = indicators_dict.get('atr_pips', 10.0) # Renomeado
        if atr_pips_cs_ema == 0.0: atr_pips_cs_ema = 10.0
        pip_size_cs_ema = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        sl_atr_mult_ema = self.parameters['atr_multiplier_sl_ema'] # Renomeado
        rr_ratio_ema = self.parameters['target_risk_reward_ratio_ema'] # Renomeado

        sl_distance_pips = atr_pips_cs_ema * sl_atr_mult_ema
        tp_distance_pips = sl_distance_pips * rr_ratio_ema
        
        stop_loss_val_ema: float # Adicionada tipagem
        take_profit_val_ema: float # Adicionada tipagem

        if signal_side == 'buy':
            stop_loss_val_ema = current_price_cs - (sl_distance_pips * pip_size_cs_ema)
            # Opcional: SL pode ser abaixo da EMA lenta também
            # stop_loss_val_ema = min(stop_loss_val_ema, indicators_dict['ema_slow'] - (atr_pips_cs_ema * 0.5 * pip_size_cs_ema))
            take_profit_val_ema = current_price_cs + (tp_distance_pips * pip_size_cs_ema)
        else: # sell
            stop_loss_val_ema = current_price_cs + (sl_distance_pips * pip_size_cs_ema)
            # stop_loss_val_ema = max(stop_loss_val_ema, indicators_dict['ema_slow'] + (atr_pips_cs_ema * 0.5 * pip_size_cs_ema))
            take_profit_val_ema = current_price_cs - (tp_distance_pips * pip_size_cs_ema)

        confidence_score_ema = self._calculate_ema_stack_signal_confidence(indicators_dict, signal_side) # Renomeado

        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=confidence_score_ema,
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val_ema, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val_ema, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"EMA Stack {signal_side.upper()}. ADX: {indicators_dict.get('adx_value',0.0):.1f}, EMA Fast Slope: {indicators_dict.get('ema_fast_slope_5p',0.0):.5f}",
            metadata={
                'atr_pips_at_signal': atr_pips_cs_ema,
                'rsi_at_signal': indicators_dict.get('rsi_value'),
                'adx_at_signal': indicators_dict.get('adx_value'),
                'ema_fast_val': indicators_dict.get('ema_fast'),
                'ema_medium_val': indicators_dict.get('ema_medium'),
                'ema_slow_val': indicators_dict.get('ema_slow'),
                'sl_pips_calculated': sl_distance_pips
            }
        )

    def _calculate_ema_stack_signal_confidence(self, indicators_dict: Dict[str, Any], signal_side: str) -> float: # Renomeado
        """Calcula a confiança do sinal EMA Stack (0.5 a 1.0)."""
        confidence = 0.60 # Confiança base

        # Força do ADX
        adx_conf = indicators_dict.get('adx_value', 0.0) # Renomeado
        if adx_conf > 35: confidence += 0.15 # ADX bem forte
        elif adx_conf > self.parameters['min_adx_trend_strength_ema']: confidence += 0.05

        # Inclinação da EMA rápida (se calculada e positiva/negativa na direção do trade)
        slope_conf = indicators_dict.get('ema_fast_slope_5p', 0.0) # Renomeado
        if signal_side == 'buy' and slope_conf > 0.00005: confidence += 0.05 # Ex: inclinação positiva
        elif signal_side == 'sell' and slope_conf < -0.00005: confidence += 0.05

        # RSI não extremo (espaço para o preço correr)
        rsi_conf = indicators_dict.get('rsi_value', 50.0) # Renomeado
        if signal_side == 'buy' and 40 < rsi_conf < 60 : confidence += 0.05 # RSI neutro/subindo
        elif signal_side == 'sell' and 40 < rsi_conf < 60 : confidence += 0.05 # RSI neutro/caindo

        return round(np.clip(confidence, 0.5, 0.95), 4)


    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Avalia condições de saída para a estratégia EMA Stack."""
        indic = self.current_indicators
        if not indic or np.isnan(indic.get('ema_fast', np.nan)):
            return None

        current_price_exit_es = market_context['tick'].mid # Renomeado

        # 1. Saída se EMAs desalinharem (ex: EMA rápida cruza a média contra a posição)
        ema_fast_exit = indic['ema_fast'] # Renomeado
        ema_medium_exit = indic['ema_medium'] # Renomeado

        if open_position.side.lower() == 'buy':
            if ema_fast_exit < ema_medium_exit: # Cruzamento bearish
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída EMA Stack: EMA Rápida cruzou abaixo da Média.")
            # (Opcional) Se usar EMA rápida como trailing stop e preço fechar abaixo dela
            if self.parameters['use_ema_fast_as_trailing_stop'] and current_price_exit_es < ema_fast_exit:
                # Esta lógica é mais para definir um NOVO SL. Atingir o SL é tratado pelo sistema.
                # Mas se quiser forçar saída aqui:
                # return ExitSignal(position_id_to_close=open_position.id, reason="Saída EMA Stack: Preço fechou abaixo da EMA Rápida (Trailing).")
                pass # Deixar o SL dinâmico ser atualizado pela BaseStrategy ou ExecutionEngine

        elif open_position.side.lower() == 'sell':
            if ema_fast_exit > ema_medium_exit: # Cruzamento bullish
                return ExitSignal(position_id_to_close=open_position.id, reason="Saída EMA Stack: EMA Rápida cruzou acima da Média.")
            if self.parameters['use_ema_fast_as_trailing_stop'] and current_price_exit_es > ema_fast_exit:
                # return ExitSignal(position_id_to_close=open_position.id, reason="Saída EMA Stack: Preço fechou acima da EMA Rápida (Trailing).")
                pass


        # 2. Saída se ADX indicar perda de força da tendência
        if indic.get('adx_value', 100.0) < self.parameters['min_adx_trend_strength_ema'] - 5: # ADX caiu bem
            return ExitSignal(position_id_to_close=open_position.id, reason=f"Saída EMA Stack: ADX ({indic.get('adx_value'):.1f}) indica perda de força da tendência.")

        # 3. (Removido) Saída parcial por lucro - geralmente gerenciado pelo Orchestrator ou config de TP.

        return None