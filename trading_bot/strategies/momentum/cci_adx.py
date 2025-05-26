# strategies/momentum/cci_adx.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados
import talib # Para CCI, ADX, ATR
from typing import Dict, Optional, Any, List, Tuple # Adicionado List, Tuple
from datetime import datetime, timezone # Adicionado timezone
from collections import deque # Não usado diretamente aqui, mas pode ser útil para buffers

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default, etc.

logger = setup_logger("cci_adx_strategy_logger") # Nome do logger específico

class CCIADXStrategy(BaseStrategy):
    """
    Estratégia que combina o Commodity Channel Index (CCI) com o Average Directional Index (ADX).
    O CCI é usado para identificar condições de sobrecompra/sobrevenda ou início de novos movimentos.
    O ADX é usado para confirmar a força da tendência direcional.
    Entradas podem ser em pullbacks dentro de tendências fortes ou em cruzamentos do CCI.
    """

    def __init__(self):
        super().__init__("CCI_ADX_TrendPullback") # Nome mais descritivo
        self.suitable_regimes = [MarketRegime.TREND] # Primariamente para tendências
        self.min_time_between_signals_sec = 180  # 3 minutos

        # Buffers para cálculos (não eram usados no original, mas podem ser se necessário para estado)
        # self.typical_price_buffer: deque[float] = deque(maxlen=200) # Exemplo

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Parâmetros CCI
            'cci_period': 20,
            'cci_overbought_level': 100, # Renomeado de cci_overbought
            'cci_oversold_level': -100,  # Renomeado de cci_oversold
            'cci_extreme_overbought_level': 150, # Renomeado e ajustado (original era 200)
            'cci_extreme_oversold_level': -150, # Renomeado e ajustado

            # Parâmetros ADX
            'adx_period': 14,
            'adx_min_strength_threshold': 25, # Renomeado de adx_threshold
            'adx_strong_trend_threshold': 35, # Renomeado de adx_strong (ajustado)
            'di_plus_minus_min_diff': 5, # Diferença mínima entre DI+ e DI- para confirmar direção

            # Lógica de Entrada
            'use_cci_divergence_entry': True, # Renomeado
            'use_cci_zero_line_cross_entry': True, # Renomeado

            # Gestão de Risco
            'atr_period_for_sl_tp': 14, # Renomeado
            'atr_multiplier_sl_cci_adx': 1.5, # Renomeado
            'target_risk_reward_ratio_cci_adx': 2.0, # Renomeado (TP será SL_dist * RR_ratio)

            # Filtros
            'min_bars_in_cci_zone_for_signal': 2, # Mínimo de barras na zona OB/OS antes do sinal
            'max_spread_pips_cci_adx_entry': 1.5,
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula CCI, ADX, ATR e outros indicadores necessários."""
        self.current_indicators = {} # Resetar

        recent_ticks_list = market_context.get('recent_ticks', [])
        # Determinar o lookback necessário para todos os indicadores
        min_data_len = max(self.parameters['cci_period'],
                           self.parameters['adx_period'],
                           self.parameters['atr_period_for_sl_tp']
                           ) + 50  # Buffer para aquecimento dos indicadores

        if not recent_ticks_list or len(recent_ticks_list) < min_data_len:
            self.logger.debug(f"Dados insuficientes para indicadores CCI_ADX ({len(recent_ticks_list)}/{min_data_len}).")
            return

        # Preparar arrays de preço para TA-Lib
        # Assumindo que BaseStrategy._get_prices_from_context está disponível
        # ou implementar aqui
        close_prices = self._get_prices_from_context(market_context, 'mid')
        high_prices = self._get_prices_from_context(market_context, 'high')
        low_prices = self._get_prices_from_context(market_context, 'low')

        if not (len(close_prices) >= min_data_len and \
                len(high_prices) >= min_data_len and \
                len(low_prices) >= min_data_len):
            self.logger.debug(f"Arrays de preço insuficientes após extração para CCI_ADX.")
            return


        # CCI (Commodity Channel Index) - requer high, low, close
        cci_val = talib.CCI(high_prices, low_prices, close_prices, timeperiod=self.parameters['cci_period'])[-1] # Renomeado

        # ADX (Average Directional Index) e DI+/-
        adx_val = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.parameters['adx_period'])[-1] # Renomeado
        plus_di_val = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.parameters['adx_period'])[-1] # Renomeado
        minus_di_val = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=self.parameters['adx_period'])[-1] # Renomeado

        # ATR (Average True Range)
        atr_price_val = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.parameters['atr_period_for_sl_tp'])[-1] # Renomeado
        pip_size_val = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado
        atr_pips_val = atr_price_val / pip_size_val if atr_price_val > 0 and pip_size_val > 0 else 0.0 # Renomeado

        # CCI Rate of Change (para detectar divergências, exemplo simplificado)
        cci_series = talib.CCI(high_prices, low_prices, close_prices, timeperiod=self.parameters['cci_period'])
        cci_roc_val = 0.0 # Renomeado
        if len(cci_series) >= 10 and not np.isnan(cci_series[-1]) and not np.isnan(cci_series[-10]):
            cci_roc_val = cci_series[-1] - cci_series[-10]

        # Preço Rate of Change (para divergência)
        price_roc_val = talib.ROC(close_prices, timeperiod=10)[-1] if len(close_prices) >=11 else 0.0 # Renomeado

        # Determinar Zona CCI
        cci_current_zone = 'neutral' # Renomeado
        if cci_val > self.parameters['cci_extreme_overbought_level']: cci_current_zone = 'extreme_overbought'
        elif cci_val > self.parameters['cci_overbought_level']: cci_current_zone = 'overbought'
        elif cci_val < self.parameters['cci_extreme_oversold_level']: cci_current_zone = 'extreme_oversold'
        elif cci_val < self.parameters['cci_oversold_level']: cci_current_zone = 'oversold'


        self.current_indicators = {
            'cci_value': cci_val,
            'cci_zone': cci_current_zone,
            'cci_roc_10p': cci_roc_val, # CCI RoC 10 períodos
            'adx_value': adx_val,
            'plus_di_value': plus_di_val,
            'minus_di_value': minus_di_val,
            'di_difference': plus_di_val - minus_di_val,
            'atr_pips': atr_pips_val,
            'current_price_mid': close_prices[-1],
            'price_roc_10p': price_roc_val, # Preço RoC 10 períodos
            'spread_pips': market_context.get('spread', 0.0) / pip_size_val,
            # 'typical_price': (high_prices[-1] + low_prices[-1] + close_prices[-1]) / 3.0 # Se necessário
        }
    
    # _calculate_cci, _calculate_adx_full, _wilders_smoothing removidos pois TA-Lib é usado.

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinais de entrada baseados na combinação de CCI e ADX."""
        indic = self.current_indicators
        if not indic or np.isnan(indic.get('cci_value', np.nan)) or np.isnan(indic.get('adx_value', np.nan)):
            self.logger.debug("Indicadores CCI/ADX não disponíveis ou NaN. Sem sinal.")
            return None

        # Filtro de Spread
        max_spread = self.parameters['max_spread_pips_cci_adx_entry']
        if indic.get('spread_pips', float('inf')) > max_spread:
            self.logger.debug(f"Spread ({indic.get('spread_pips'):.1f} pips) muito alto para CCI_ADX (Max: {max_spread}).")
            return None

        # Condição primária: Força da tendência (ADX)
        if indic['adx_value'] < self.parameters['adx_min_strength_threshold']:
            self.logger.debug(f"ADX ({indic['adx_value']:.1f}) abaixo do limiar ({self.parameters['adx_min_strength_threshold']}). Sem tendência forte.")
            return None

        signal_side_to_gen_ca: Optional[str] = None # Renomeado
        signal_reason_text = "" # Renomeado

        # Estratégia 1: Pullback em Tendência Forte (CCI em extremos)
        is_strong_trend = indic['adx_value'] >= self.parameters['adx_strong_trend_threshold']
        di_confirm_bullish = indic['di_difference'] > self.parameters['di_plus_minus_min_diff']
        di_confirm_bearish = indic['di_difference'] < -self.parameters['di_plus_minus_min_diff']

        if is_strong_trend:
            if indic['cci_zone'] in ['oversold', 'extreme_oversold'] and di_confirm_bullish:
                signal_side_to_gen_ca = 'buy'
                signal_reason_text = f"Pullback Compra CCI ({indic['cci_value']:.0f}) em Tendência Alta Forte (ADX {indic['adx_value']:.0f})"
            elif indic['cci_zone'] in ['overbought', 'extreme_overbought'] and di_confirm_bearish:
                signal_side_to_gen_ca = 'sell'
                signal_reason_text = f"Pullback Venda CCI ({indic['cci_value']:.0f}) em Tendência Baixa Forte (ADX {indic['adx_value']:.0f})"

        # Estratégia 2: Cruzamento da Linha Zero do CCI com confirmação de ADX/DI
        if not signal_side_to_gen_ca and self.parameters['use_cci_zero_line_cross_entry']:
            # Checar se CCI cruzou zero recentemente (ex: CCI[-2] < 0 and CCI[-1] > 0)
            # Isso requereria guardar o CCI anterior no self.internal_state
            # Para simplificar, usar CCI_RoC e a posição atual do CCI.
            if indic['cci_value'] > 0 and indic.get('cci_roc_10p',0) > 20 and di_confirm_bullish: # Cruzou para cima com ímpeto
                signal_side_to_gen_ca = 'buy'
                signal_reason_text = f"CCI Cruzou Linha Zero para Cima (ADX {indic['adx_value']:.0f})"
            elif indic['cci_value'] < 0 and indic.get('cci_roc_10p',0) < -20 and di_confirm_bearish: # Cruzou para baixo com ímpeto
                signal_side_to_gen_ca = 'sell'
                signal_reason_text = f"CCI Cruzou Linha Zero para Baixo (ADX {indic['adx_value']:.0f})"

        # Estratégia 3: Divergência CCI vs Preço (se habilitado)
        if not signal_side_to_gen_ca and self.parameters['use_cci_divergence_entry']:
            divergence_type = self._check_cci_price_divergence(indic) # Renomeado
            if divergence_type == 'bullish' and di_confirm_bullish:
                signal_side_to_gen_ca = 'buy'
                signal_reason_text = f"Divergência Bullish CCI-Preço (ADX {indic['adx_value']:.0f})"
            elif divergence_type == 'bearish' and di_confirm_bearish:
                signal_side_to_gen_ca = 'sell'
                signal_reason_text = f"Divergência Bearish CCI-Preço (ADX {indic['adx_value']:.0f})"
        
        if signal_side_to_gen_ca:
            return self._create_cci_adx_signal(signal_side_to_gen_ca, indic, signal_reason_text, market_context) # Renomeado
        return None


    def _check_cci_price_divergence(self, indicators_dict: Dict[str, Any]) -> Optional[str]: # Renomeado
        """Verifica divergências entre preço (ROC) e CCI (ROC)."""
        price_roc = indicators_dict.get('price_roc_10p', 0.0)
        cci_roc = indicators_dict.get('cci_roc_10p', 0.0)

        # Divergência Bullish: Preço fazendo mínimas mais baixas (price_roc < 0), CCI fazendo mínimas mais altas (cci_roc > 0)
        # Esta é uma simplificação. Divergência real compara picos/vales.
        if price_roc < -0.05 and cci_roc > 5: # Ex: Preço caiu 0.05%, CCI subiu 5 pontos
            return 'bullish'
        # Divergência Bearish: Preço fazendo máximas mais altas (price_roc > 0), CCI fazendo máximas mais baixas (cci_roc < 0)
        elif price_roc > 0.05 and cci_roc < -5:
            return 'bearish'
        return None


    def _create_cci_adx_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                          signal_reason: str, market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal com SL/TP baseados em ATR e R:R."""
        current_price_cs = indicators_dict['current_price_mid'] # Renomeado
        atr_pips_cs = indicators_dict.get('atr_pips', 10.0) # Renomeado
        if atr_pips_cs == 0: atr_pips_cs = 10.0 # Default se ATR for zero
        pip_size_cs = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        sl_multiplier_cs = self.parameters['atr_multiplier_sl_cci_adx'] # Renomeado
        rr_ratio_cs = self.parameters['target_risk_reward_ratio_cci_adx'] # Renomeado

        # Ajustar SL/TP com base na força do ADX (opcional)
        adx_strength_factor = 1.0
        if indicators_dict['adx_value'] > self.parameters['adx_strong_trend_threshold']:
            adx_strength_factor = 0.8 # SL mais apertado em tendência muito forte (cuidado com volatilidade)
        elif indicators_dict['adx_value'] < self.parameters['adx_min_strength_threshold'] + 5: # ADX fraco mas acima do limiar
            adx_strength_factor = 1.2 # SL mais largo

        sl_distance_pips = atr_pips_cs * sl_multiplier_cs * adx_strength_factor
        tp_distance_pips = sl_distance_pips * rr_ratio_cs

        stop_loss_cs: float # Adicionada tipagem
        take_profit_cs: float # Adicionada tipagem

        if signal_side == 'buy':
            stop_loss_cs = current_price_cs - (sl_distance_pips * pip_size_cs)
            take_profit_cs = current_price_cs + (tp_distance_pips * pip_size_cs)
        else: # sell
            stop_loss_cs = current_price_cs + (sl_distance_pips * pip_size_cs)
            take_profit_cs = current_price_cs - (tp_distance_pips * pip_size_cs)

        confidence_score_cs = self._calculate_cci_adx_signal_confidence(indicators_dict, signal_side) # Renomeado

        return Signal(
            timestamp=datetime.now(timezone.utc), # Usar UTC
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL), # Adicionar símbolo
            side=signal_side,
            confidence=confidence_score_cs,
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_cs, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_cs, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=signal_reason,
            metadata={
                'cci_at_signal': indicators_dict.get('cci_value'),
                'cci_zone_at_signal': indicators_dict.get('cci_zone'),
                'adx_at_signal': indicators_dict.get('adx_value'),
                'di_diff_at_signal': indicators_dict.get('di_difference'),
                'atr_pips_at_signal': atr_pips_cs,
                'sl_pips_calculated': sl_distance_pips
            }
        )

    def _calculate_cci_adx_signal_confidence(self, indicators_dict: Dict[str, Any], signal_side: str) -> float: # Renomeado
        """Calcula a confiança do sinal CCI/ADX."""
        confidence = 0.60 # Confiança base

        # Força do ADX
        adx_val_conf = indicators_dict.get('adx_value', 0.0) # Renomeado
        if adx_val_conf > self.parameters['adx_strong_trend_threshold']: confidence += 0.15
        elif adx_val_conf > self.parameters['adx_min_strength_threshold'] + 5: confidence += 0.05 # Pouco acima do limiar

        # CCI em zona extrema ou confirmando bem
        cci_val_conf = indicators_dict.get('cci_value', 0.0) # Renomeado
        if abs(cci_val_conf) > self.parameters.get('cci_extreme_overbought_level', 150) * 0.8 : # Perto do extremo
            confidence += 0.10
        
        # Confirmação DI+/-
        di_diff_conf = indicators_dict.get('di_difference', 0.0) # Renomeado
        min_di_diff_conf = self.parameters['di_plus_minus_min_diff'] # Renomeado
        if signal_side == 'buy' and di_diff_conf > min_di_diff_conf * 2: # DI+ bem acima de DI-
            confidence += 0.05
        elif signal_side == 'sell' and di_diff_conf < -min_di_diff_conf * 2: # DI- bem acima de DI+
            confidence += 0.05
        
        return round(np.clip(confidence, 0.5, 0.95), 4)


    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Define condições de saída para a estratégia CCI/ADX."""
        indic = self.current_indicators
        if not indic or np.isnan(indic.get('cci_value', np.nan)) or np.isnan(indic.get('adx_value', np.nan)):
            return None # Sem indicadores, não avaliar saída

        current_price_exit_ca = market_context['tick'].mid # Renomeado
        
        # 1. Saída se CCI cruzar para zona oposta de forma significativa
        cci_val_exit = indic.get('cci_value', 0.0) # Renomeado
        if open_position.side.lower() == 'buy':
            # Se estava comprado (CCI baixo), e agora CCI está bem sobrecomprado
            if cci_val_exit > self.parameters['cci_overbought_level'] + 20: # Um pouco além do nível OB
                return ExitSignal(position_id_to_close=open_position.id, reason=f"Saída CCI: Atingiu zona oposta sobrecomprada ({cci_val_exit:.0f})")
        elif open_position.side.lower() == 'sell':
            # Se estava vendido (CCI alto), e agora CCI está bem sobrevendido
            if cci_val_exit < self.parameters['cci_oversold_level'] - 20:
                return ExitSignal(position_id_to_close=open_position.id, reason=f"Saída CCI: Atingiu zona oposta sobrevendida ({cci_val_exit:.0f})")

        # 2. Saída se a força da tendência (ADX) diminuir drasticamente ou DI cruzar contra
        adx_val_exit = indic.get('adx_value', 0.0) # Renomeado
        di_diff_exit = indic.get('di_difference', 0.0) # Renomeado
        if adx_val_exit < self.parameters['adx_min_strength_threshold'] - 5: # ADX caiu bem abaixo do limiar de entrada
            return ExitSignal(position_id_to_close=open_position.id, reason=f"Saída ADX: Tendência enfraqueceu (ADX {adx_val_exit:.0f})")
        
        if open_position.side.lower() == 'buy' and di_diff_exit < -self.parameters['di_plus_minus_min_diff'] / 2.0 : # DI- cruza DI+
            return ExitSignal(position_id_to_close=open_position.id, reason="Saída DI: Cruzamento DI- sobre DI+")
        elif open_position.side.lower() == 'sell' and di_diff_exit > self.parameters['di_plus_minus_min_diff'] / 2.0 : # DI+ cruza DI-
            return ExitSignal(position_id_to_close=open_position.id, reason="Saída DI: Cruzamento DI+ sobre DI-")

        return None