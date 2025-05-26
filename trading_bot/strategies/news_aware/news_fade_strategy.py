# strategies/news_aware/news_fade_strategy.py
import numpy as np
import pandas as pd # Adicionado para manipulação de dados
import talib # Para ATR
from typing import Dict, Optional, Any, List, Tuple # Adicionado Tuple
from datetime import datetime, timedelta, timezone # Adicionado timezone
import asyncio
# import aiohttp # Removido, pois a busca de notícias não está implementada aqui
# Se for usar API externa para notícias, aiohttp seria necessário.

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger
from config.settings import CONFIG # Para símbolo default, etc.
# Importar TickData se for usado explicitamente nos tipos de market_context
from api.ticktrader_ws import TickData


logger = setup_logger("news_fade_strategy_logger") # Nome do logger específico

class NewsFadeStrategy(BaseStrategy):
    """
    Estratégia que opera "fade" (reversão) de spikes de preço induzidos por notícias.
    Detecta movimentos exagerados causados por eventos econômicos e tenta entrar
    contra esse movimento, esperando uma correção quando a volatilidade normaliza.
    """

    def __init__(self):
        super().__init__("NewsSpikeFadeReversal") # Nome da estratégia mais descritivo
        self.suitable_regimes = [MarketRegime.HIGH_VOLATILITY, MarketRegime.TREND] # Notícias podem ocorrer em tendências também
        self.min_time_between_signals_sec = 1800  # 30 minutos (para evitar overtrading em torno de um evento)

        # Cache de notícias e calendário
        # self.news_cache: List[Dict[str, Any]] = [] # Não usado no código original
        # self.last_news_check_time: Optional[datetime] = None # Renomeado
        self.upcoming_economic_events: List[Dict[str, Any]] = [] # Renomeado
        self.last_calendar_load_time: Optional[datetime] = None # Para recarregar periodicamente

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Configurações de Notícias e Calendário
            'enable_economic_calendar_check': True, # Renomeado de 'check_calendar'
            'filter_by_high_impact_news_only': True, # Renomeado
            'news_event_monitor_window_before_mins': 30, # Janela ANTES do evento # Renomeado
            'news_event_monitor_window_after_mins': 60,  # Janela APÓS o evento # Renomeado
            'min_price_spike_pips_for_fade': 15.0, # Renomeado

            # Critérios de Entrada para Fade
            'spike_min_velocity_pips_per_sec': 15.0,  # Velocidade mínima do spike (pips/segundo) # Renomeado e ajustado
            'spike_volatility_expansion_ratio': 2.5,  # Volatilidade atual vs normal (ex: 2.5x) # Renomeado
            'delay_after_spike_peak_seconds': 20,    # Aguardar N segundos após o pico do spike antes de entrar # Renomeado

            # Gestão de Risco para Trades de Notícia
            'atr_period_news_stops': 14,
            'atr_multiplier_sl_news': 2.5,  # SL mais largo devido à volatilidade de notícias # Renomeado
            'fixed_tp_pips_news_fade': 20.0, # TP fixo em pips
            'max_position_hold_time_seconds': 1800,  # 30 minutos

            # Filtros Adicionais
            # 'avoid_trading_certain_major_pairs': ['USDJPY', 'GBPUSD'], # Renomeado
            'min_spread_quality_for_news_trade': 0.4, # Aceitar spread pior durante notícias (0.0 a 1.0)
            'require_retracement_before_entry': True,
            'min_retracement_percentage_of_spike': 0.25, # Ex: 25% de retração do spike # Renomeado

            # Configuração de API de Calendário (placeholders, precisam ser preenchidos)
            'calendar_api_url': "YOUR_CALENDAR_API_URL_HERE", # Ex: de ForexFactory, Myfxbook, etc.
            'calendar_api_key': "YOUR_API_KEY_HERE",
            'calendar_reload_interval_hours': 4,
        }

    async def initialize_strategy(self): # Renomeado de initialize
        """Inicializa a estratégia e carrega o calendário econômico pela primeira vez."""
        await super().initialize_strategy()
        if self.parameters['enable_economic_calendar_check']:
            await self._load_and_cache_economic_calendar() # Renomeado

    async def _load_and_cache_economic_calendar(self): # Renomeado
        """
        Carrega e armazena em cache eventos do calendário econômico para o dia/semana.
        NECESSITA DE IMPLEMENTAÇÃO REAL com uma API de calendário.
        """
        # Placeholder: Em produção, usar uma API real (ForexFactory, Investing.com, Myfxbook, etc.)
        # Esta função deve ser chamada periodicamente (ex: a cada poucas horas) para atualizar os eventos.
        
        now_utc = datetime.now(timezone.utc)
        if self.last_calendar_load_time and \
           (now_utc - self.last_calendar_load_time) < timedelta(hours=self.parameters['calendar_reload_interval_hours']):
            self.logger.debug("Calendário econômico já carregado recentemente.")
            return

        self.logger.info("Carregando/Atualizando calendário econômico...")
        self.upcoming_economic_events = [] # Limpar eventos antigos

        # --- INÍCIO DA LÓGICA DE BUSCA DE API (EXEMPLO SIMULADO) ---
        # try:
        #     async with aiohttp.ClientSession() as session:
        #         # Montar URL com data de início e fim (ex: próximos 7 dias)
        #         start_date_str = now_utc.strftime('%Y-%m-%d')
        #         end_date_str = (now_utc + timedelta(days=7)).strftime('%Y-%m-%d')
        #         api_url = f"{self.parameters['calendar_api_url']}?from={start_date_str}&to={end_date_str}&apikey={self.parameters['calendar_api_key']}"
        #         async with session.get(api_url) as response:
        #             if response.status == 200:
        #                 events_data = await response.json()
        #                 # Processar events_data para o formato esperado por self.upcoming_economic_events
        #                 # Ex: {'time': datetime_obj_utc, 'currency': 'USD', 'event': 'Non-Farm Payrolls', 'impact': 'high'}
        #             else:
        #                 self.logger.error(f"Erro ao buscar calendário econômico: {response.status}")
        # except Exception as e_cal:
        #     self.logger.exception(f"Exceção ao carregar calendário econômico: {e_cal}")
        # --- FIM DA LÓGICA DE BUSCA DE API ---

        # Usando dados simulados como no original, mas com timestamps UTC corretos
        self.upcoming_economic_events = [
            {
                'time': now_utc + timedelta(hours=2, minutes=np.random.randint(0,59)), # Adicionar alguma variação
                'currency': 'USD', 'event': 'Non-Farm Payrolls', 'impact': 'high',
                'forecast': '180K', 'previous': '175K'
            },
            {
                'time': now_utc + timedelta(hours=4, minutes=np.random.randint(0,59)),
                'currency': 'EUR', 'event': 'ECB Interest Rate Decision', 'impact': 'high',
                'forecast': '4.50%', 'previous': '4.50%'
            },
            {
                'time': now_utc + timedelta(days=1, hours=1, minutes=np.random.randint(0,59)),
                'currency': 'GBP', 'event': 'BoE MPC Vote', 'impact': 'medium',
                'forecast': '7-0-2', 'previous': '7-0-2'
            }
        ]
        # Ordenar eventos por tempo
        self.upcoming_economic_events.sort(key=lambda x: x['time'])
        self.last_calendar_load_time = now_utc
        self.logger.info(f"Calendário econômico carregado/atualizado: {len(self.upcoming_economic_events)} eventos futuros relevantes.")


    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula indicadores para detecção de spikes de notícias e volatilidade."""
        self.current_indicators = {} # Resetar
        
        # Recarregar calendário se necessário
        if self.parameters['enable_economic_calendar_check']:
            await self._load_and_cache_economic_calendar() # Chamada async

        recent_ticks_list = market_context.get('recent_ticks', []) # Lista de TickData
        # Precisa de dados suficientes para ATR e para detectar spikes
        min_len_ci_nf = max(self.parameters.get('atr_period_news_stops', 14), 100) + 1 # Buffer para spikes e ATR

        if not recent_ticks_list or len(recent_ticks_list) < min_len_ci_nf:
            self.logger.debug(f"Dados insuficientes para indicadores NewsFade ({len(recent_ticks_list)}/{min_len_ci_nf}).")
            return

        # Preparar arrays de preço
        close_prices = self._get_prices_from_context(market_context, 'mid')
        high_prices = self._get_prices_from_context(market_context, 'high')
        low_prices = self._get_prices_from_context(market_context, 'low')
        # Timestamps para cálculo de duração de spike
        timestamps_pd = pd.to_datetime([tick.timestamp for tick in recent_ticks_list if hasattr(tick, 'timestamp')], utc=True)


        if not (len(close_prices) >= min_len_ci_nf and len(timestamps_pd) == len(close_prices)):
            self.logger.debug("Arrays de preço ou timestamps insuficientes após extração para NewsFade.")
            return

        # ATR para stops e referência de volatilidade
        atr_val_price_nf = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.parameters['atr_period_news_stops'])[-1] # Renomeado
        pip_size_nf = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado
        atr_pips_nf = atr_val_price_nf / pip_size_nf if atr_val_price_nf > 0 and pip_size_nf > 0 else 0.0 # Renomeado

        # Detectar spike de preço
        spike_is_detected, spike_details_dict = self._find_price_spike(close_prices, timestamps_pd, atr_pips_nf) # Renomeado

        # Verificar proximidade de eventos de notícias
        is_near_news_event, relevant_news_event_info = self._get_nearest_relevant_news_event() # Renomeado

        # Volatilidade atual vs. normal (ex: usando std dev de retornos ou largura de BBands)
        # Usando std dev de retornos dos últimos N ticks vs. M ticks anteriores
        returns_short_term = pd.Series(close_prices).pct_change().iloc[-20:].std() # Ex: últimos 20 ticks
        returns_longer_term = pd.Series(close_prices).pct_change().iloc[-100:-20].std() # Ex: 80 ticks antes disso
        vol_expansion_ratio = returns_short_term / (returns_longer_term + 1e-9) if not np.isnan(returns_short_term) and not np.isnan(returns_longer_term) else 1.0

        # Retração após o spike
        retracement_pct_val = 0.0 # Renomeado
        if spike_is_detected and spike_details_dict:
            retracement_pct_val = self._calculate_spike_retracement(close_prices[-1], spike_details_dict)


        self.current_indicators = {
            'is_price_spike_detected': spike_is_detected, # Renomeado
            'price_spike_details': spike_details_dict, # Renomeado
            'is_near_economic_event': is_near_news_event, # Renomeado
            'active_news_event_info': relevant_news_event_info, # Renomeado
            'volatility_expansion_ratio': vol_expansion_ratio,
            'atr_pips': atr_pips_nf,
            'current_spike_retracement_pct': retracement_pct_val,
            'current_price_mid': close_prices[-1],
            'current_spread_pips': market_context.get('spread', 0.0) / pip_size_nf,
            # 'spread_widening_ratio': ..., # Se for calcular em relação à média
        }

    def _find_price_spike(self, prices_arr: np.ndarray, timestamps_pd_series: pd.Series, # Renomeado
                        current_atr_pips: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Detecta um spike de preço significativo nos dados recentes."""
        # Olhar para diferentes janelas de tempo para o spike (ex: últimos 5s, 10s, 30s)
        # Esta lógica é complexa e depende de como se define um "spike".
        # A original iterava de 5 a 20 ticks. Vamos tentar uma abordagem baseada em tempo e ATR.
        
        min_spike_pips_val = self.parameters['min_price_spike_pips_for_fade']
        min_velocity_val = self.parameters['spike_min_velocity_pips_per_sec']

        # Analisar os últimos 60 segundos, por exemplo
        lookback_duration_for_spike = timedelta(seconds=60)
        relevant_indices = timestamps_pd_series >= (timestamps_pd_series.iloc[-1] - lookback_duration_for_spike)
        
        if np.sum(relevant_indices) < 5 : return False, None # Pelo menos 5 pontos de dados

        window_prices = prices_arr[relevant_indices]
        window_timestamps = timestamps_pd_series[relevant_indices]

        price_change_in_window = abs(window_prices[-1] - window_prices[0])
        duration_in_seconds = (window_timestamps.iloc[-1] - window_timestamps.iloc[0]).total_seconds()
        if duration_in_seconds < 1e-3: return False, None # Evitar divisão por zero

        price_change_pips = price_change_in_window / (0.0001 if "JPY" not in CONFIG.SYMBOL else 0.01)
        velocity_pips_s = price_change_pips / duration_in_seconds

        if price_change_pips >= min_spike_pips_val and velocity_pips_s >= min_velocity_val:
            # Verificar se o spike é > X vezes o ATR atual (ex: 3x ATR)
            if current_atr_pips > 0 and price_change_pips < (current_atr_pips * 1.5): # Spike pequeno demais em relação à vol atual
                 return False, None

            spike_direction = 'up' if window_prices[-1] > window_prices[0] else 'down'
            # O "pico" do spike é o máximo/mínimo na janela, não necessariamente o último preço
            peak_price_of_spike = np.max(window_prices) if spike_direction == 'up' else np.min(window_prices)
            
            spike_details = {
                'start_price_of_move': window_prices[0],
                'end_price_of_move': window_prices[-1], # Preço no fim da janela de detecção
                'peak_price_of_spike': peak_price_of_spike,
                'direction_of_spike': spike_direction, # Renomeado
                'velocity_pips_s': round(velocity_pips_s, 2),
                'magnitude_pips': round(price_change_pips, 2),
                'duration_seconds': round(duration_in_seconds, 2),
                'timestamp_of_detection': window_timestamps.iloc[-1].to_pydatetime(warn=False) # Timestamp da detecção
            }
            return True, spike_details
        return False, None


    def _get_nearest_relevant_news_event(self) -> Tuple[bool, Optional[Dict[str, Any]]]: # Renomeado
        """Verifica se há um evento econômico relevante próximo (antes ou depois)."""
        if not self.parameters['enable_economic_calendar_check'] or not self.upcoming_economic_events:
            return False, None

        now_utc_check = datetime.now(timezone.utc) # Renomeado
        min_before = self.parameters['news_event_monitor_window_before_mins']
        min_after = self.parameters['news_event_monitor_window_after_mins']

        for event_item in self.upcoming_economic_events: # Renomeado
            event_time_utc = event_item['time'] # Já deve ser datetime UTC
            if not isinstance(event_time_utc, datetime): continue # Pular se não for datetime

            time_to_event_minutes = (event_time_utc - now_utc_check).total_seconds() / 60.0

            # Checar se estamos na janela de monitoramento do evento
            # (Evento está para acontecer OU acabou de acontecer)
            if -min_after <= time_to_event_minutes <= min_before:
                # Filtrar por impacto e moeda (EUR ou USD para EURUSD)
                if self.parameters['filter_by_high_impact_news_only'] and event_item.get('impact', '').lower() != 'high':
                    continue
                if event_item.get('currency', '').upper() not in ['EUR', 'USD', CONFIG.SYMBOL[:3], CONFIG.SYMBOL[3:]]: # Checar ambas as moedas do par
                    continue
                
                # Evento relevante encontrado
                event_info_to_return: Dict[str, Any] = event_item.copy() # Renomeado
                event_info_to_return['time_to_event_minutes'] = round(time_to_event_minutes, 1)
                event_info_to_return['is_post_event_window'] = time_to_event_minutes < 0 # Renomeado
                return True, event_info_to_return
        
        return False, None


    def _calculate_spike_retracement(self, current_price_retracement: float, # Renomeado
                                   detected_spike_info: Dict[str, Any]) -> float:
        """Calcula o percentual de retração do preço atual em relação ao spike detectado."""
        if not detected_spike_info: return 0.0

        peak_price = detected_spike_info['peak_price_of_spike']
        start_price = detected_spike_info['start_price_of_move'] # Usar o início do movimento que gerou o spike
        spike_direction = detected_spike_info['direction_of_spike']

        total_spike_range = abs(peak_price - start_price)
        if total_spike_range < 1e-9: return 0.0 # Evitar divisão por zero

        retracement_amount_price: float # Adicionada tipagem
        if spike_direction == 'up': # Spike de alta, retração é queda a partir do pico
            retracement_amount_price = peak_price - current_price_retracement
        else: # Spike de baixa, retração é alta a partir do vale (pico aqui é o mínimo)
            retracement_amount_price = current_price_retracement - peak_price
        
        # Retração percentual do range total do spike
        retracement_percentage = retracement_amount_price / total_spike_range
        # Limitar entre 0 (sem retração ou movimento continuou) e 1 (retração completa)
        # Se current_price_retracement foi além do start_price na direção da retração, pct > 1.
        # Se current_price_retracement continuou na direção do spike, pct < 0.
        return round(np.clip(retracement_percentage, 0.0, 2.0), 4) # Permitir retração até 200%

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de fade (reversão) se um spike de notícia válido for detectado."""
        indic = self.current_indicators
        if not indic or not indic.get('is_price_spike_detected') or not indic.get('is_near_economic_event'):
            # Precisa de um spike E estar perto de uma notícia relevante
            return None

        spike_info_gs = indic['price_spike_details'] # Renomeado
        news_info_gs = indic['active_news_event_info'] # Renomeado

        # Filtro de volatilidade: Spike deve ter causado expansão significativa
        if indic.get('volatility_expansion_ratio', 1.0) < self.parameters['spike_volatility_expansion_ratio']:
            self.logger.debug(f"NewsFade ignorado: Expansão de volatilidade ({indic.get('volatility_expansion_ratio'):.1f}x) insuficiente.")
            return None

        # Atraso após o PICO do spike antes de entrar no fade
        # Precisamos do timestamp do pico do spike. spike_info_gs['timestamp_of_detection'] é o fim da janela.
        # Esta lógica precisa ser mais robusta para identificar o "pico" real e o tempo desde então.
        # Por ora, vamos assumir que a detecção é rápida e 'duration_seconds' do spike_info_gs é relevante.
        # Se o spike acabou de acontecer (ex: duração < delay_after_spike_peak_seconds), esperar.
        # if spike_info_gs['duration_seconds'] < self.parameters['delay_after_spike_peak_seconds']: # Esta lógica é falha.
        # Correção: Precisamos saber quando o PICO ocorreu. Se não temos essa info, não podemos usar esse delay.
        # Vamos assumir que o sinal só é gerado se o spike_info é do tick atual.
        # Se o Orchestrator chama calculate_indicators e generate_signal no mesmo tick,
        # então o delay pode ser implementado esperando N ticks *após* a detecção inicial do pico.
        # Essa lógica de "espera" geralmente é gerenciada pelo estado interno da estratégia.
        # Por enquanto, vamos pular o delay exato e focar na retração.

        # Filtro de Retração (se habilitado)
        if self.parameters['require_retracement_before_entry']:
            if indic.get('current_spike_retracement_pct', 0.0) < self.parameters['min_retracement_percentage_of_spike']:
                self.logger.debug(f"NewsFade ignorado: Retração do spike ({indic.get('current_spike_retracement_pct'):.1%}) insuficiente.")
                return None
        
        # Determinar direção do FADE (oposta ao spike)
        signal_side_fade: Optional[str] = None # Renomeado
        if spike_info_gs['direction_of_spike'] == 'up':
            signal_side_fade = 'sell' # Vender o fade da alta
        elif spike_info_gs['direction_of_spike'] == 'down':
            signal_side_fade = 'buy'  # Comprar o fade da baixa
        
        if not signal_side_fade: return None

        # Aumentar confiança se for PÓS-notícia (mais confiável que pré-notícia)
        confidence_adj_factor = 0.0 # Renomeado
        if news_info_gs.get('is_post_event_window', False):
            confidence_adj_factor = 0.1

        return self._create_news_fade_trade_signal( # Renomeado
            signal_side_fade, indic, confidence_adj_factor, market_context
        )


    def _create_news_fade_trade_signal(self, signal_side: str, indicators_dict: Dict[str, Any], # Renomeado
                                confidence_adjustment: float, market_context: Dict[str, Any]) -> Signal:
        """Cria o objeto Signal para a estratégia NewsFade."""
        current_price_cs_nf = indicators_dict['current_price_mid'] # Renomeado
        atr_pips_cs_nf = indicators_dict.get('atr_pips', 15.0) # Usar ATR maior como default para notícias # Renomeado
        if atr_pips_cs_nf == 0.0: atr_pips_cs_nf = 15.0
        pip_size_cs_nf = 0.0001 if "JPY" not in market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL).upper() else 0.01 # Renomeado

        spike_details_cs = indicators_dict['price_spike_details'] # Renomeado
        news_details_cs = indicators_dict['active_news_event_info'] # Renomeado

        # SL é crucial em trades de notícia. Colocar além do pico/vale do spike + buffer ATR.
        sl_atr_mult_nf = self.parameters['atr_multiplier_sl_news'] # Renomeado
        stop_loss_val_nf: float # Adicionada tipagem

        if signal_side == 'buy': # Fade de spike de baixa -> Comprar
            # SL abaixo do mínimo do spike (peak_price_of_spike é o mínimo aqui)
            stop_loss_val_nf = spike_details_cs['peak_price_of_spike'] - (atr_pips_cs_nf * sl_atr_mult_nf * 0.5 * pip_size_cs_nf) # Buffer de 0.5 * ATR mult
            take_profit_val_nf = current_price_cs_nf + (self.parameters['fixed_tp_pips_news_fade'] * pip_size_cs_nf)
        else: # signal_side == 'sell', Fade de spike de alta -> Vender
            # SL acima do máximo do spike
            stop_loss_val_nf = spike_details_cs['peak_price_of_spike'] + (atr_pips_cs_nf * sl_atr_mult_nf * 0.5 * pip_size_cs_nf)
            take_profit_val_nf = current_price_cs_nf - (self.parameters['fixed_tp_pips_news_fade'] * pip_size_cs_nf)

        # Calcular confiança (0.5 a 1.0)
        # Usar a confiança ajustada passada e adicionar mais fatores
        base_conf_nf = 0.55 + confidence_adjustment # Renomeado
        if spike_details_cs['magnitude_pips'] > self.parameters['min_price_spike_pips_for_fade'] * 1.5 : base_conf_nf += 0.10 # Spike maior
        if spike_details_cs['velocity_pips_s'] > self.parameters['spike_min_velocity_pips_per_sec'] * 1.5 : base_conf_nf += 0.10 # Spike mais rápido
        if indicators_dict.get('current_spike_retracement_pct',0.0) > self.parameters['min_retracement_percentage_of_spike'] * 1.5 : base_conf_nf += 0.05 # Retração maior

        final_confidence_nf = round(np.clip(base_conf_nf, 0.5, 0.90), 4) # Limitar confiança máxima para trades de notícia

        return Signal(
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
            side=signal_side,
            confidence=final_confidence_nf,
            entry_price=None, # Entrada a mercado
            stop_loss=round(stop_loss_val_nf, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            take_profit=round(take_profit_val_nf, 5 if "JPY" not in CONFIG.SYMBOL else 3),
            order_type="Market",
            reason=f"NewsFade {signal_side.upper()} após spike de {spike_details_cs['magnitude_pips']:.1f} pips. Evento: {news_details_cs.get('event', 'N/A')}.",
            metadata={
                'spike_info': spike_details_cs,
                'news_event_info': news_details_cs,
                'volatility_expansion_ratio': indicators_dict.get('volatility_expansion_ratio'),
                'retracement_pct_at_entry': indicators_dict.get('current_spike_retracement_pct'),
                'atr_pips_at_entry': atr_pips_cs_nf
            }
        )

    # _calculate_news_fade_confidence foi integrado em _create_news_fade_trade_signal

    async def evaluate_exit_conditions(self, open_position: Position, # Renomeado
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Condições de saída para a estratégia NewsFade."""
        indic = self.current_indicators
        if not indic: return None # Sem indicadores, não fazer nada

        current_price_exit_nf = market_context['tick'].mid # Renomeado

        # 1. Saída se volatilidade normalizar significativamente
        if indic.get('volatility_expansion_ratio', 10.0) < self.parameters.get('exit_volatility_ratio_threshold', 1.2): # Ex: se ratio < 1.2
            return ExitSignal(position_id_to_close=open_position.id, reason="Volatilidade normalizou, fechando trade de notícia.")

        # 2. Saída por tempo máximo de holding
        time_held_exit_s = (datetime.now(timezone.utc) - open_position.open_time).total_seconds() # Renomeado
        if time_held_exit_s > self.parameters['max_position_hold_time_seconds']:
            return ExitSignal(position_id_to_close=open_position.id, reason=f"Tempo máximo ({self.parameters['max_position_hold_time_seconds']/60:.0f} min) para trade de notícia atingido.")

        # 3. Saída se um NOVO spike ocorrer na mesma direção da posição (indicando continuação, não fade)
        #    Isso requer que 'is_price_spike_detected' e 'price_spike_details' sejam do *tick atual*.
        if indic.get('is_price_spike_detected', False) and indic.get('price_spike_details'):
            new_spike_details = indic['price_spike_details'] # Renomeado
            # Checar se este é um spike *diferente* do que originou a trade (ex: por timestamp ou magnitude)
            # Para simplificar, se um novo spike na direção da posição é detectado:
            if (open_position.side.lower() == 'buy' and new_spike_details['direction_of_spike'] == 'up' and new_spike_details['magnitude_pips'] > 5) or \
               (open_position.side.lower() == 'sell' and new_spike_details['direction_of_spike'] == 'down' and new_spike_details['magnitude_pips'] > 5):
                return ExitSignal(position_id_to_close=open_position.id, reason="Novo spike detectado na direção da posição (fade falhou).")

        # 4. Lógica de Trailing Stop Agressivo (ex: mover para breakeven rápido)
        #    Esta lógica é melhor tratada pela BaseStrategy ou ExecutionEngine com base nos parâmetros.
        #    Mas a estratégia pode *sinalizar* para ativar um trailing stop mais agressivo.
        #    Exemplo: Se PnL > X pips, instruir para mover SL para Breakeven + Y pips.
        #    Isso requer que o metadata da posição seja atualizável ou que o ExitSignal possa conter instruções de modificação.
        #    Por ora, não implementado aqui como ExitSignal.

        return None