# strategies/breakout/donchian_breakout.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados se necessário
import talib # Para ADX e ATR
from typing import Dict, Optional, Any, List # Adicionado List
from datetime import datetime, timedelta, timezone # Adicionado timezone

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para configurações globais como SYMBOL

logger = setup_logger("donchian_breakout_strategy") # Nome do logger específico

class DonchianBreakoutStrategy(BaseStrategy):
    """
    Estratégia de breakout usando Donchian Channel.
    Entra em rompimentos de máximas/mínimas de N períodos,
    confirmados por volume (estimado), momentum e força da tendência (ADX).
    """

    def __init__(self):
        super().__init__("DonchianBreakout_55_v2") # Nome pode ser ajustado com versão
        self.suitable_regimes = [MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY]
        self.min_time_between_signals_sec = 600  # 10 minutos (herdado de BaseStrategy, pode ser sobrescrito)

        # Rastreamento de estado específico da estratégia (se necessário)
        # self.last_breakout_level_high: Optional[float] = None
        # self.last_breakout_level_low: Optional[float] = None
        # O original tinha last_high, last_low, breakout_confirmed, mas a lógica de reset
        # em calculate_indicators não os usava de forma clara para prevenir reentradas imediatas.
        # A lógica de cooldown de BaseStrategy (min_time_between_signals_sec) já ajuda com isso.

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Parâmetros Donchian
            'donchian_lookback_period': 55, # Renomeado de 'channel_period'
            'donchian_exit_lookback_period': 20, # Renomeado de 'exit_period'
            'use_middle_channel_exit': True, # Renomeado de 'use_middle_exit'

            # Filtros de Confirmação de Breakout
            'min_breakout_distance_pips': 5.0,  # Mínimo de pips para considerar um breakout válido
            'volume_surge_ratio_required': 1.5,  # Ex: 1.5 significa 150% do volume médio
            'adx_confirmation_required': True, # Renomeado de 'momentum_confirmation' (ADX é melhor para força)
            'min_adx_for_breakout': 20, # Limiar mínimo do ADX
            # 'confirmation_bars_count': 2, # Renomeado (lógica de confirmação pode ser mais complexa)

            # Gestão de Risco
            'atr_period_for_stops': 14, # Período do ATR para cálculo de stops
            'atr_multiplier_sl': 2.0,
            'target_risk_reward_ratio': 2.5, # Renomeado de 'atr_multiplier_tp'
            'use_atr_trailing_stop': True, # Renomeado de 'use_trailing_stop'
            'trailing_stop_activation_atr_multiple': 1.0,  # Ativar TS quando em lucro de X * ATR
            'trailing_stop_distance_atr_multiple': 1.5, # Distância do TS em múltiplos de ATR

            # Filtros Gerais
            'min_atr_value_pips': 3.0, # ATR mínimo em pips para operar
            'max_spread_pips_entry': 1.5, # Spread máximo em pips para entrar # Renomeado de 'max_spread'
            # 'avoid_news_window_minutes': 30, # Herdado de GLOBAL_SETTINGS em BaseStrategy se enable_news_filter=True
            'apply_session_time_filter': True, # Renomeado 'use_time_filter', BaseStrategy pode ter um filtro global
                                              # Este seria um filtro adicional específico da estratégia
            # 'london_open_filter': True, # Exemplo de filtros de sessão específicos, se necessário
            # 'ny_open_filter': True,

            # Gestão de Posição (se aplicável diretamente na estratégia)
            # 'scale_out_enabled': True,
            # 'scale_out_levels_r': [(0.5, 0.5), (1.0, 0.3), (1.5, 0.2)], # (R múltiplo, % saída)
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula Donchian Channel, ATR, ADX e outros indicadores auxiliares."""
        # Extrair dados de ticks recentes do market_context
        # BaseStrategy._get_prices_from_context pode ser usado
        recent_ticks_data = market_context.get('recent_ticks', []) # Lista de objetos TickData
        if not recent_ticks_data or len(recent_ticks_data) < self.parameters['donchian_lookback_period'] + 1: # +1 para diffs
            self.logger.debug("Dados insuficientes para calcular indicadores Donchian.")
            self.current_indicators = {} # Limpar indicadores se não puder calcular
            return

        # Converter para arrays numpy para TA-Lib
        # Usar high e low reais para Donchian, e close (mid) para outros cálculos
        # O ideal é que TickData tenha 'high' e 'low' do período do tick (se for barra),
        # ou usar 'ask' e 'bid' como proxy se forem ticks puros.
        # Para Donchian, geralmente se usa as máximas dos asks e mínimas dos bids.
        # Assumindo que market_context['recent_ticks'] são ticks e usamos ask/bid.
        
        # Se os ticks tiverem apenas bid/ask, high = ask, low = bid.
        # Se forem barras OHLC, usar high e low da barra.
        # Para simplificar, se 'recent_ticks' forem objetos TickData:
        high_prices = np.array([tick.ask for tick in recent_ticks_data if hasattr(tick, 'ask')])
        low_prices = np.array([tick.bid for tick in recent_ticks_data if hasattr(tick, 'bid')])
        close_prices = np.array([tick.mid for tick in recent_ticks_data if hasattr(tick, 'mid')])
        # Volume estimado (simplificado)
        volumes_est = np.array([(getattr(tick, 'bid_volume',0) + getattr(tick, 'ask_volume',0))/2 for tick in recent_ticks_data])
        volumes_est = np.maximum(volumes_est, 1.0) # Evitar volume zero

        if not (len(high_prices) > 0 and len(low_prices) > 0 and len(close_prices) > 0):
            self.logger.warning("Arrays de high, low ou close vazios após extração de ticks.")
            self.current_indicators = {}
            return


        lookback = self.parameters['donchian_lookback_period']
        exit_lookback = self.parameters['donchian_exit_lookback_period']
        atr_period = self.parameters['atr_period_for_stops']

        # Garantir dados suficientes para TA-Lib (alguns indicadores precisam de mais que 'period' para aquecer)
        min_len_for_talib = max(lookback, exit_lookback, atr_period) + 20 # Buffer adicional

        if len(close_prices) < min_len_for_talib:
            self.logger.debug(f"Dados insuficientes ({len(close_prices)}/{min_len_for_talib}) para cálculos TA-Lib.")
            self.current_indicators = {}
            return

        # Donchian Channel (calculado manualmente, pois TA-Lib não tem Donchian direto)
        # Usar slicing para pegar os últimos 'lookback' períodos
        upper_channel_val = np.max(high_prices[-lookback:]) if len(high_prices) >= lookback else np.nan # Renomeado
        lower_channel_val = np.min(low_prices[-lookback:]) if len(low_prices) >= lookback else np.nan # Renomeado
        middle_channel_val = (upper_channel_val + lower_channel_val) / 2.0 if not (np.isnan(upper_channel_val) or np.isnan(lower_channel_val)) else np.nan # Renomeado

        exit_upper_val = np.max(high_prices[-exit_lookback:]) if len(high_prices) >= exit_lookback else np.nan # Renomeado
        exit_lower_val = np.min(low_prices[-exit_lookback:]) if len(low_prices) >= exit_lookback else np.nan # Renomeado

        # Preço atual
        current_ask_price = high_prices[-1] # Usar ask para breakout de alta
        current_bid_price = low_prices[-1]  # Usar bid para breakout de baixa
        current_mid_price = close_prices[-1]

        # ATR
        atr_val = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)[-1] # Renomeado

        # ADX para força da tendência (TA-Lib)
        adx_val = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1] # Usar período padrão ou configurável # Renomeado

        # Volume (estimado, como ratio da média)
        # TA-Lib não tem volume ratio direto, calcular manualmente
        avg_volume_val = np.mean(volumes_est[-20:]) if len(volumes_est) >= 20 else np.mean(volumes_est) if len(volumes_est)>0 else 1.0 # Renomeado
        current_volume_val = volumes_est[-1] if len(volumes_est)>0 else 1.0 # Renomeado
        volume_surge_ratio_val = current_volume_val / (avg_volume_val + 1e-9) # Evitar divisão por zero # Renomeado

        # Momentum (ROC - Rate of Change)
        momentum_roc_val = talib.ROC(close_prices, timeperiod=10)[-1] # Ex: ROC de 10 períodos # Renomeado


        # Breakout detection (baseado nos preços atuais vs canais)
        breakout_up_flag = not np.isnan(upper_channel_val) and current_ask_price > upper_channel_val # Renomeado
        breakout_down_flag = not np.isnan(lower_channel_val) and current_bid_price < lower_channel_val # Renomeado

        breakout_dist_pips = 0.0 # Renomeado
        pip_size = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01
        if breakout_up_flag:
            breakout_dist_pips = (current_ask_price - upper_channel_val) / pip_size
        elif breakout_down_flag:
            breakout_dist_pips = (lower_channel_val - current_bid_price) / pip_size


        self.current_indicators = {
            'upper_channel': upper_channel_val,
            'lower_channel': lower_channel_val,
            'middle_channel': middle_channel_val,
            'channel_width_pips': (upper_channel_val - lower_channel_val) / pip_size if not (np.isnan(upper_channel_val) or np.isnan(lower_channel_val)) else 0.0,
            'exit_upper_channel': exit_upper_val, # Renomeado
            'exit_lower_channel': exit_lower_val, # Renomeado
            'current_ask': current_ask_price, # Renomeado
            'current_bid': current_bid_price, # Renomeado
            'current_mid': current_mid_price, # Renomeado
            'is_breakout_up': breakout_up_flag, # Renomeado
            'is_breakout_down': breakout_down_flag, # Renomeado
            'breakout_distance_pips': breakout_dist_pips,
            'atr_pips': atr_val / pip_size if atr_val is not None else 0.0, # Converter ATR para pips
            'volume_surge_ratio': volume_surge_ratio_val,
            'momentum_roc': momentum_roc_val,
            'adx_value': adx_val, # Renomeado
            'spread_pips': market_context.get('spread', 0.0) / pip_size
        }

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de breakout se as condições forem atendidas."""
        indic = self.current_indicators # Renomeado para 'indic' para brevidade
        if not indic: # Se indicadores não foram calculados
            return None

        # Aplicar filtros de entrada básicos
        if not self._passes_entry_filters(indic): # Renomeado
            return None

        signal_side_to_gen: Optional[str] = None # Renomeado

        # Breakout de Alta
        if indic.get('is_breakout_up', False):
            if self._is_valid_breakout_confirmation('up', indic): # Renomeado
                signal_side_to_gen = 'buy'

        # Breakout de Baixa
        elif indic.get('is_breakout_down', False):
            if self._is_valid_breakout_confirmation('down', indic): # Renomeado
                signal_side_to_gen = 'sell'

        if signal_side_to_gen:
            return self._create_donchian_breakout_signal(signal_side_to_gen, indic, market_context) # Renomeado
        return None


    def _passes_entry_filters(self, indicators_dict: Dict[str, Any]) -> bool: # Renomeado
        """Verifica filtros básicos de entrada (ATR, spread, horário)."""
        min_atr_pips = self.parameters['min_atr_value_pips']
        if indicators_dict.get('atr_pips', 0.0) < min_atr_pips:
            self.logger.debug(f"Filtro ATR não passou: {indicators_dict.get('atr_pips', 0.0):.1f} < {min_atr_pips:.1f} pips.")
            return False

        max_spread_val = self.parameters['max_spread_pips_entry'] # Renomeado
        if indicators_dict.get('spread_pips', float('inf')) > max_spread_val:
            self.logger.debug(f"Filtro Spread não passou: {indicators_dict.get('spread_pips', 0.0):.1f} > {max_spread_val:.1f} pips.")
            return False

        if self.parameters['apply_session_time_filter']:
            current_utc_hour = datetime.now(timezone.utc).hour
            # Usar o filtro de tempo da BaseStrategy ou um específico aqui
            if not self.get_time_filter_for_strategy(current_utc_hour): # Assumindo que este método existe e é relevante
                self.logger.debug(f"Filtro de horário da estratégia não passou para hora {current_utc_hour} UTC.")
                return False
        return True


    def _is_valid_breakout_confirmation(self, direction: str, indicators_dict: Dict[str, Any]) -> bool: # Renomeado
        """Confirma se o breakout é válido baseado em distância, volume, ADX."""
        # Distância mínima do breakout
        if indicators_dict.get('breakout_distance_pips', 0.0) < self.parameters['min_breakout_distance_pips']:
            self.logger.debug(f"Breakout {direction} não confirmado: distância pequena ({indicators_dict.get('breakout_distance_pips',0.0):.1f} pips).")
            return False

        # Surge de Volume
        if indicators_dict.get('volume_surge_ratio', 0.0) < self.parameters['volume_surge_ratio_required']:
            self.logger.debug(f"Breakout {direction} não confirmado: volume baixo (ratio {indicators_dict.get('volume_surge_ratio',0.0):.2f}).")
            return False

        # Confirmação de ADX (força da tendência)
        if self.parameters['adx_confirmation_required']:
            if indicators_dict.get('adx_value', 0.0) < self.parameters['min_adx_for_breakout']:
                self.logger.debug(f"Breakout {direction} não confirmado: ADX baixo ({indicators_dict.get('adx_value',0.0):.1f}).")
                return False
        
        # (Opcional) Confirmação de Momentum (ROC) - se ROC positivo para compra, negativo para venda
        # roc_val = indicators_dict.get('momentum_roc', 0.0)
        # if direction == 'up' and roc_val <= 0: return False
        # if direction == 'down' and roc_val >= 0: return False

        self.logger.info(f"BREAKOUT {direction.upper()} CONFIRMADO: Dist={indicators_dict.get('breakout_distance_pips',0.0):.1f} pips, VolRatio={indicators_dict.get('volume_surge_ratio',0.0):.2f}, ADX={indicators_dict.get('adx_value',0.0):.1f}")
        return True


    def _create_donchian_breakout_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                               market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal com SL/TP calculados."""
        # Preço de entrada: para breakout, geralmente é o nível do canal rompido ou o preço atual.
        # Usar o preço atual de ask (para compra) ou bid (para venda) como referência de entrada.
        entry_ref_price = indicators_dict['current_ask'] if signal_side == 'buy' else indicators_dict['current_bid']
        atr_pips_val = indicators_dict.get('atr_pips', self.parameters['min_atr_value_pips']) # Renomeado
        pip_sz = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        sl_atr_mult = self.parameters['atr_multiplier_sl'] # Renomeado
        stop_loss_pips_val = atr_pips_val * sl_atr_mult # Renomeado
        
        stop_loss_price_val: float # Adicionada tipagem
        take_profit_price_val: float # Adicionada tipagem

        if signal_side == 'buy':
            # SL abaixo do canal inferior ou ATR
            stop_loss_price_val = min(indicators_dict['lower_channel'] - (atr_pips_val * 0.5 * pip_sz), # Um pouco abaixo do canal
                                 entry_ref_price - (stop_loss_pips_val * pip_sz))
            risk_pips = (entry_ref_price - stop_loss_price_val) / pip_sz
            take_profit_pips = risk_pips * self.parameters['target_risk_reward_ratio']
            take_profit_price_val = entry_ref_price + (take_profit_pips * pip_sz)
        else: # sell
            stop_loss_price_val = max(indicators_dict['upper_channel'] + (atr_pips_val * 0.5 * pip_sz),
                                 entry_ref_price + (stop_loss_pips_val * pip_sz))
            risk_pips = (stop_loss_price_val - entry_ref_price) / pip_sz
            take_profit_pips = risk_pips * self.parameters['target_risk_reward_ratio']
            take_profit_price_val = entry_ref_price - (take_profit_pips * pip_sz)


        confidence_score = self._calculate_breakout_signal_confidence(indicators_dict, signal_side) # Renomeado

        return Signal(
            timestamp=datetime.now(timezone.utc), # Usar UTC
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL), # Adicionar símbolo
            side=signal_side,
            confidence=confidence_score,
            entry_price=None, # Entrada a mercado, preço será o de fill
            stop_loss=round(stop_loss_price_val, 5 if "JPY" not in CONFIG.SYMBOL else 3), # Arredondar para precisão do par
            take_profit=round(take_profit_price_val, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market", # Tipo de ordem
            reason=f"Donchian Breakout {signal_side.upper()}. ADX: {indicators_dict.get('adx_value',0.0):.1f}, BreakDist: {indicators_dict.get('breakout_distance_pips',0.0):.1f} pips",
            metadata={
                'channel_width_pips': indicators_dict.get('channel_width_pips',0.0),
                'breakout_distance_pips': indicators_dict.get('breakout_distance_pips',0.0),
                'volume_surge_ratio': indicators_dict.get('volume_surge_ratio',0.0),
                'momentum_roc': indicators_dict.get('momentum_roc',0.0),
                'atr_pips': atr_pips_val,
                'adx_value': indicators_dict.get('adx_value',0.0),
                'stop_loss_pips_calc': stop_loss_pips_val # Guardar SL em pips para referência
            }
        )

    def _calculate_breakout_signal_confidence(self, indicators_dict: Dict[str, Any], signal_side: str) -> float: # Renomeado
        """Calcula a confiança do sinal de breakout (0.5 a 1.0)."""
        confidence = 0.60  # Confiança base

        # ADX forte
        adx = indicators_dict.get('adx_value', 0.0)
        if adx > 30: confidence += 0.15
        elif adx > self.parameters['min_adx_for_breakout']: confidence += 0.05

        # Volume alto
        vol_ratio = indicators_dict.get('volume_surge_ratio', 0.0) # Renomeado
        if vol_ratio > self.parameters['volume_surge_ratio_required'] * 1.5: # Significativamente maior que o threshold
            confidence += 0.10
        elif vol_ratio > self.parameters['volume_surge_ratio_required']:
            confidence += 0.05
            
        # Distância do breakout (maior distância, mais significativo)
        break_dist_pips = indicators_dict.get('breakout_distance_pips', 0.0) # Renomeado
        if break_dist_pips > self.parameters['min_breakout_distance_pips'] * 2.0:
            confidence += 0.10
        
        # Momentum (ROC) na direção do breakout
        roc = indicators_dict.get('momentum_roc', 0.0)
        if signal_side == 'buy' and roc > 0.05 : confidence += 0.05 # Ex: ROC > 0.05%
        elif signal_side == 'sell' and roc < -0.05 : confidence += 0.05

        return round(np.clip(confidence, 0.5, 0.95), 4) # Limitar entre 0.5 e 0.95


    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Define condições de saída para a posição de breakout."""
        # Indicadores são recalculados em BaseStrategy.check_exit_for_position antes de chamar este método
        indic = self.current_indicators
        if not indic: return None

        current_price_for_exit = market_context['tick'].mid # Usar mid para avaliação de saída

        # 1. Saída por canal oposto ou médio (Donchian Exit)
        use_mid = self.parameters['use_middle_channel_exit'] # Renomeado
        exit_level_val: Optional[float] = None # Adicionada tipagem e Optional

        if open_position.side.lower() == 'buy':
            exit_target_price = indic.get('exit_lower_channel') if not use_mid else indic.get('middle_channel') # Renomeado
            if exit_target_price is not None and current_price_for_exit <= exit_target_price:
                exit_level_val = exit_target_price
                reason = f"Donchian Exit - Preço cruzou {'Canal Médio' if use_mid else 'Canal de Saída Inferior'} ({exit_level_val:.5f})"
                return ExitSignal(position_id_to_close=open_position.id, reason=reason)
        
        elif open_position.side.lower() == 'sell':
            exit_target_price = indic.get('exit_upper_channel') if not use_mid else indic.get('middle_channel')
            if exit_target_price is not None and current_price_for_exit >= exit_target_price:
                exit_level_val = exit_target_price
                reason = f"Donchian Exit - Preço cruzou {'Canal Médio' if use_mid else 'Canal de Saída Superior'} ({exit_level_val:.5f})"
                return ExitSignal(position_id_to_close=open_position.id, reason=reason)


        # 2. Trailing Stop baseado em ATR (se habilitado na posição e nos parâmetros)
        # A lógica de atualização do SL da posição pelo trailing stop é feita na BaseStrategy ou ExecutionEngine.
        # Aqui, a estratégia apenas CALCULA o novo nível de SL se o trailing estiver ativo.
        # Se o novo SL calculado for atingido, o sistema de stops normal fechará a posição.
        # Esta função é mais para *sinais de saída discricionários* da estratégia.
        # No entanto, se a estratégia quiser forçar uma saída por uma lógica de trailing stop específica:
        if self.parameters['use_atr_trailing_stop'] and open_position.metadata.get('trailing_stop_active', False):
            # O open_position.stop_loss já deve refletir o trailing stop price atualizado.
            # A checagem se o preço atual atingiu o SL é feita pelo loop de monitoramento principal.
            # Não precisa ser replicada aqui, a menos que seja uma condição de saída *adicional*.
            pass


        # 3. (Opcional) Lógica de Scale Out (saídas parciais)
        # A lógica de scale_out do original foi movida para BaseStrategy ou Orchestrator,
        # pois é mais uma técnica de gestão de posição do que um sinal de saída completo.
        # Se a estratégia precisar de uma lógica de scale-out específica, pode ser implementada aqui.

        return None # Nenhuma condição de saída específica da estratégia foi atingida


    # _calculate_adx foi removido, pois BaseStrategy agora usa TA-Lib ou
    # as estratégias chamam talib.ADX diretamente.
    # _calculate_r_multiple e _already_scaled_out também foram removidos,
    # pois scale-out é mais genérico e pode ser tratado em BaseStrategy ou Orchestrator.