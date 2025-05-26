# backtest/engine.py
"""Motor de backtesting para avaliação de estratégias de trading."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone # Adicionado timezone
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict # Não usado diretamente, mas pode ser útil
import time # Para medir tempo de execução do backtest

from strategies.base_strategy import BaseStrategy, Signal, Position as StrategyPosition # Renomeado Position
# Importar BacktestResults e BacktestTrade daqui mesmo se definidos aqui, para clareza.
# Ou movê-los para um arquivo de modelos de dados.
from utils.logger import setup_logger
from utils.helpers import convert_price_diff_to_pips # Renomeado (era price_to_pips)
# Importar PerformanceMetrics para usar como tipo de retorno consistente
from optimization.scoring import PerformanceMetrics, StrategyScorer # Para calcular métricas finais
from config.settings import CONFIG # Para configurações globais

logger = setup_logger("backtest_engine_logger") # Nome do logger específico

@dataclass
class BacktestTrade:
    """Estrutura de dados para um trade individual no backtest."""
    id: str # ID único do trade
    strategy_name: str # Nome da estratégia que gerou o trade # Renomeado de strategy
    symbol: str
    side: str  # 'buy' ou 'sell'
    entry_timestamp_utc: datetime # Renomeado de entry_time e especificado UTC
    entry_price_actual: float # Renomeado de entry_price (preço real de entrada com slippage)
    size_in_lots: float # Renomeado de size
    stop_loss_initial_price: float # Renomeado de stop_loss
    take_profit_initial_price: float # Renomeado de take_profit

    exit_timestamp_utc: Optional[datetime] = None # Renomeado e UTC
    exit_price_actual: Optional[float] = None # Renomeado
    pnl_currency: float = 0.0 # PnL na moeda da conta, líquido de comissões # Renomeado de pnl
    pnl_in_pips: float = 0.0 # Renomeado de pnl_pips
    commission_total_currency: float = 0.0 # Renomeado de commission
    exit_trigger_reason: str = "" # Renomeado de exit_reason
    duration_in_seconds: int = 0 # Renomeado

    # Métricas intra-trade
    max_favorable_excursion_pnl: float = 0.0 # MFE em PnL $ # Renomeado de max_profit
    max_adverse_excursion_pnl: float = 0.0 # MAE em PnL $ (valor negativo) # Renomeado de max_loss
    
    # Metadados do sinal original ou outros
    signal_metadata: Dict[str, Any] = field(default_factory=dict) # Renomeado e usando Any

    def to_dict(self) -> Dict[str, Any]: # Adicionado para facilitar conversão para scorer
        # Converter datetimes para ISO string para serialização
        data = self.__dict__.copy()
        if data['entry_timestamp_utc']: data['entry_timestamp_utc'] = data['entry_timestamp_utc'].isoformat()
        if data['exit_timestamp_utc']: data['exit_timestamp_utc'] = data['exit_timestamp_utc'].isoformat()
        return data


@dataclass
class BacktestResults:
    """
    Estrutura para armazenar os resultados completos de um backtest.
    Muitas métricas são agora encapsuladas em PerformanceMetrics.
    """
    # Configuração do Backtest
    strategy_name_tested: str # Renomeado
    symbol_tested: str # Renomeado
    start_date_data: datetime # Renomeado
    end_date_data: datetime # Renomeado
    initial_balance_set: float # Renomeado
    
    # Resultados Chave
    final_balance_achieved: float = 0.0 # Renomeado
    performance_metrics_calculated: Optional[PerformanceMetrics] = None # Armazena o objeto PerformanceMetrics

    # Dados brutos para análise posterior
    executed_trades_list: List[BacktestTrade] = field(default_factory=list) # Renomeado
    full_equity_curve: List[Tuple[datetime, float]] = field(default_factory=list) # (Timestamp, Equity) # Renomeado
    # drawdown_curve_pct: List[float] = field(default_factory=list) # Agora em PerformanceMetrics

    # Metadados da execução do backtest
    backtest_run_timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc)) # Renomeado
    backtest_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]: # Para consistência com PerformanceMetrics
        """Converte resultados para dicionário, útil para logs ou relatórios."""
        res_dict = {
            'strategy_name_tested': self.strategy_name_tested,
            'symbol_tested': self.symbol_tested,
            'start_date_data': self.start_date_data.isoformat() if self.start_date_data else None,
            'end_date_data': self.end_date_data.isoformat() if self.end_date_data else None,
            'initial_balance_set': self.initial_balance_set,
            'final_balance_achieved': self.final_balance_achieved,
            'performance_metrics_calculated': self.performance_metrics_calculated.to_dict() if self.performance_metrics_calculated else None,
            'executed_trades_count': len(self.executed_trades_list),
            'backtest_run_timestamp_utc': self.backtest_run_timestamp_utc.isoformat(),
            'backtest_duration_seconds': self.backtest_duration_seconds
        }
        # Opcional: adicionar lista de trades ou curva de equity se necessário para o dict
        # res_dict['executed_trades_list'] = [t.to_dict() for t in self.executed_trades_list]
        return res_dict


class BacktestEngine:
    """Motor de backtesting robusto para estratégias de trading, operando sobre dados históricos."""

    def __init__(self):
        self._current_open_positions: Dict[str, BacktestTrade] = {} # Renomeado
        self._all_closed_trades: List[BacktestTrade] = [] # Renomeado
        self._equity_curve_history: List[Tuple[datetime, float]] = [] # (Timestamp, Equity) # Renomeado
        self._current_high_water_mark: float = 0.0 # Renomeado
        self._trade_id_counter: int = 0 # Renomeado

        # Configurações do backtest (podem ser passadas no método run)
        self.current_balance: float = 0.0 # Renomeado
        self.initial_balance_bt: float = 0.0 # Renomeado
        self.commission_per_trade_pct: float = 0.0 # Renomeado
        self.slippage_one_side_price: float = 0.0 # Renomeado

        self.allow_multiple_simultaneous_positions: bool = True # Renomeado
        # self.use_tick_level_data: bool = True # Implícito pelos dados de entrada # Renomeado
        # self.calculate_intrabar_trade_metrics: bool = True # Para MFE/MAE # Renomeado

        self._scorer = StrategyScorer() # Para calcular métricas finais


    def _reset_backtest_state(self): # Renomeado
        """Reseta o estado interno do motor para um novo backtest."""
        self._current_open_positions.clear()
        self._all_closed_trades.clear()
        self._equity_curve_history.clear()
        self._current_high_water_mark = 0.0
        self._trade_id_counter = 0
        self.current_balance = 0.0
        self.initial_balance_bt = 0.0
        logger.debug("Estado do BacktestEngine resetado.")


    def run(self, strategy_to_test: BaseStrategy, # Renomeado
            historical_price_data: pd.DataFrame, # Renomeado
            initial_account_balance: float = 10000.0, # Renomeado
            commission_rate_pct: float = 0.00002,  # 0.002% = 0.2 pips (0.00002 * 10000 pips = 0.2 pips)
            slippage_per_side_price_val: float = 0.00001,    # 0.1 pip em valor de preço
            symbol_name: str = CONFIG.SYMBOL) -> BacktestResults: # Renomeado
        """
        Executa um backtest síncrono para a estratégia fornecida.
        `historical_price_data` deve ser um DataFrame com colunas:
        'timestamp' (datetime UTC), 'bid', 'ask', 'mid' (opcional),
        e opcionalmente 'bid_volume', 'ask_volume', 'high', 'low'.
        """
        self._reset_backtest_state()
        logger.info(f"Iniciando backtest para '{strategy_to_test.name}' em '{symbol_name}'. "
                   f"Período: {historical_price_data.index.min()} a {historical_price_data.index.max() if isinstance(historical_price_data.index, pd.DatetimeIndex) else 'N/A'}.")

        self.initial_balance_bt = initial_account_balance
        self.current_balance = initial_account_balance
        self._current_high_water_mark = initial_account_balance
        self.commission_per_trade_pct = commission_rate_pct # Comission rate como percentual do valor nocional
        self.slippage_one_side_price = slippage_per_side_price_val
        
        # Preparar dados: garantir colunas e índice de timestamp
        data_for_backtest = historical_price_data.copy() # Renomeado
        if not isinstance(data_for_backtest.index, pd.DatetimeIndex):
            if 'timestamp' in data_for_backtest.columns:
                data_for_backtest['timestamp'] = pd.to_datetime(data_for_backtest['timestamp'], utc=True)
                data_for_backtest.set_index('timestamp', inplace=True)
            else:
                logger.error("Dados históricos não possuem índice de timestamp nem coluna 'timestamp'. Backtest abortado.")
                return BacktestResults(strategy_name_tested=strategy_to_test.name, symbol_tested=symbol_name,
                                       start_date_data=datetime.now(timezone.utc), end_date_data=datetime.now(timezone.utc), # Placeholder
                                       initial_balance_set=initial_account_balance)

        # Garantir colunas 'mid', 'high', 'low', 'spread'
        if 'mid' not in data_for_backtest.columns: data_for_backtest['mid'] = (data_for_backtest['bid'] + data_for_backtest['ask']) / 2.0
        if 'high' not in data_for_backtest.columns: data_for_backtest['high'] = data_for_backtest['ask'] # Proxy se não houver OHLC
        if 'low' not in data_for_backtest.columns: data_for_backtest['low'] = data_for_backtest['bid']   # Proxy
        if 'spread' not in data_for_backtest.columns: data_for_backtest['spread'] = data_for_backtest['ask'] - data_for_backtest['bid']

        self._equity_curve_history.append((data_for_backtest.index[0], self.initial_balance_bt)) # Ponto inicial da equity

        total_data_points = len(data_for_backtest) # Renomeado
        processed_data_points = 0 # Renomeado
        backtest_start_time = time.perf_counter() # Renomeado

        # Iterar sobre cada linha (tick ou barra) dos dados históricos
        for current_timestamp, current_row_data in data_for_backtest.iterrows(): # Renomeado idx, row
            # Construir contexto de mercado atual para a estratégia
            # Passar a fatia de dados até o ponto atual para cálculo de indicadores
            # O current_pos é o índice inteiro da linha atual no DataFrame original
            current_pos_idx = data_for_backtest.index.get_loc(current_timestamp) # Renomeado
            
            # A fatia para 'recent_ticks' deve incluir a linha atual
            # Ex: data_for_backtest.iloc[max(0, current_pos_idx - N + 1) : current_pos_idx + 1]
            # Para BaseStrategy, ela espera uma lista de objetos TickData.
            # Esta conversão deve ser feita em _create_market_context_for_strategy.
            market_context_bt = self._create_market_context_for_strategy(data_for_backtest, current_pos_idx, current_row_data, symbol_name) # Renomeado

            # 1. Verificar SL/TP para posições abertas com os preços atuais
            self._check_and_process_stops_for_open_positions(current_row_data, current_timestamp) # Renomeado

            # 2. (Opcional) Verificar sinais de saída discricionários da estratégia
            for pos_id_exit, open_pos_exit in list(self._current_open_positions.items()): # Renomeado
                if open_pos_exit.strategy_name == strategy_to_test.name: # Apenas para posições desta estratégia
                    exit_sig: Optional[ExitSignal] = asyncio.run(strategy_to_test.check_exit_for_position(open_pos_exit, market_context_bt)) # Rodar async
                    if exit_sig:
                        self._close_open_position( # Renomeado
                            exit_sig.position_id_to_close,
                            exit_sig.exit_price if exit_sig.exit_price else current_row_data['mid'], # Usar mid se preço de saída não especificado
                            current_timestamp,
                            exit_sig.reason,
                            partial_close_size=exit_sig.exit_size_lots # Adicionado para fechamento parcial
                        )


            # 3. Processar o tick/barra pela estratégia para NOVOS sinais de entrada
            # A estratégia é ativada/desativada externamente, aqui assumimos que ela decide se opera
            if strategy_to_test.active: # Checar se a estratégia está ativa
                entry_signal_obj: Optional[Signal] = asyncio.run(strategy_to_test.on_tick(market_context_bt)) # Rodar async # Renomeado
                if entry_signal_obj:
                    self._process_and_execute_entry_signal(entry_signal_obj, current_row_data, current_timestamp, symbol_name) # Renomeado

            # 4. Atualizar curva de equity
            self._update_equity_curve_and_hwm(current_row_data, current_timestamp) # Renomeado

            # Log de progresso
            processed_data_points += 1
            if processed_data_points % (total_data_points // 20 or 1) == 0: # Logar a cada 5%
                progress_pct = (processed_data_points / total_data_points) * 100 # Renomeado
                elapsed_bt_time = time.perf_counter() - backtest_start_time # Renomeado
                processing_speed_dps = processed_data_points / elapsed_bt_time if elapsed_bt_time > 0 else 0 # Renomeado (data points per second)
                logger.info(f"Backtest Progresso: {progress_pct:.0f}% ({processed_data_points}/{total_data_points}) | "
                           f"Velocidade: {processing_speed_dps:.0f} dps | Trades: {len(self._all_closed_trades)}")


        # Fim do loop de dados: fechar quaisquer posições remanescentes
        if self._current_open_positions:
            last_data_row = data_for_backtest.iloc[-1]
            last_timestamp = data_for_backtest.index[-1]
            logger.info(f"Fechando {len(self._current_open_positions)} posições abertas no final do backtest.")
            for pos_id_final in list(self._current_open_positions.keys()): # Renomeado
                self._close_open_position(pos_id_final, last_data_row['mid'], last_timestamp, "Fim do período de backtest")

        backtest_total_duration_s = time.perf_counter() - backtest_start_time # Renomeado
        logger.info(f"Backtest para '{strategy_to_test.name}' concluído em {backtest_total_duration_s:.2f}s. "
                   f"Velocidade média: {total_data_points / backtest_total_duration_s if backtest_total_duration_s > 0 else 0:.0f} dps. "
                   f"Total de trades fechados: {len(self._all_closed_trades)}")

        # Preparar e retornar resultados finais
        final_results_obj = self._compile_final_backtest_results(strategy_to_test.name, symbol_name, data_for_backtest.index[0], data_for_backtest.index[-1], backtest_total_duration_s) # Renomeado
        return final_results_obj


    async def run_async(self, strategy: BaseStrategy, data: pd.DataFrame,
                       initial_balance: float = 10000.0,
                       commission_rate_pct: float = 0.00002, # Renomeado
                       slippage_price: float = 0.00001, # Renomeado
                       symbol: str = CONFIG.SYMBOL) -> BacktestResults: # Adicionado symbol
        """Versão assíncrona do backtest para compatibilidade com otimizadores asyncio."""
        # O motor de backtest em si é síncrono. Usar run_in_executor para chamá-lo de um contexto async.
        loop = asyncio.get_event_loop()
        # Passar os nomes corretos dos parâmetros para a função síncrona 'run'
        return await loop.run_in_executor(
            None, # Default ThreadPoolExecutor
            self.run, # Método síncrono
            strategy,
            data,
            initial_account_balance=initial_balance, # Nome do arg em run()
            commission_rate_pct=commission_rate_pct,   # Nome do arg em run()
            slippage_per_side_price_val=slippage_price, # Nome do arg em run()
            symbol_name=symbol                         # Nome do arg em run()
        )


    def _create_market_context_for_strategy(self, full_hist_data: pd.DataFrame, current_data_idx: int, # Renomeado
                                         current_row_series: pd.Series, symbol_ctx: str) -> Dict[str, Any]: # Renomeado
        """Cria o dicionário de contexto de mercado para a estratégia."""
        # Determinar lookback para 'recent_ticks' (ex: últimos 1000 pontos de dados)
        # Este lookback deve ser suficiente para o maior período de indicador da estratégia.
        # Pode ser pego de strategy.parameters se definido lá, ou um default grande.
        # strategy_instance = BaseStrategy() # Não instanciar BaseStrategy, usar a que foi passada
        # default_lookback = strategy_instance.parameters.get('context_lookback_ticks', 1000)
        # Por ora, usar um valor fixo ou o início do DataFrame.
        lookback_window_size = 1000 # Renomeado
        start_idx_slice = max(0, current_data_idx - lookback_window_size + 1) # Renomeado
        end_idx_slice = current_data_idx + 1 # Renomeado
        recent_data_slice_df = full_hist_data.iloc[start_idx_slice:end_idx_slice] # Renomeado

        # Converter a fatia do DataFrame para uma lista de objetos TickData (ou dicts compatíveis)
        recent_ticks_for_strat: List[TickData] = [] # Renomeado
        for ts_tick, row_tick in recent_data_slice_df.iterrows(): # Renomeado
            # Construir TickData (ou dict compatível)
            # O objeto TickData espera um dict no formato da API.
            # Aqui, estamos convertendo de um DataFrame de backtest.
            tick_dict_from_row = {
                'Symbol': symbol_ctx,
                'Timestamp': int(ts_tick.timestamp() * 1000), # Milissegundos
                'BestBid': {'Price': row_tick['bid'], 'Volume': row_tick.get('bid_volume', 100000.0)},
                'BestAsk': {'Price': row_tick['ask'], 'Volume': row_tick.get('ask_volume', 100000.0)},
                # Adicionar outros campos se TickData os usa (mid, spread são calculados no __init__ de TickData)
            }
            recent_ticks_for_strat.append(TickData(tick_dict_from_row))

        # Volatilidade (exemplo: ATR sobre os recent_ticks_for_strat)
        # Esta lógica pode ser mais sofisticada ou usar um helper.
        volatility_val = 0.0
        if len(recent_ticks_for_strat) > 14: # Mínimo para ATR(14)
            highs_vol = np.array([t.ask for t in recent_ticks_for_strat]) # Proxy
            lows_vol = np.array([t.bid for t in recent_ticks_for_strat])
            closes_vol = np.array([t.mid for t in recent_ticks_for_strat])
            try:
                atr_val_vol = talib.ATR(highs_vol, lows_vol, closes_vol, 14)[-1] # Renomeado
                # Normalizar ATR (ex: como % do preço)
                volatility_val = (atr_val_vol / (closes_vol[-1] + 1e-9)) if closes_vol[-1] > 0 else 0.0
            except Exception: # TA-Lib pode falhar com dados ruins
                volatility_val = 0.01 # Fallback


        return {
            'tick': recent_ticks_for_strat[-1] if recent_ticks_for_strat else None,
            'recent_ticks': recent_ticks_for_strat, # Lista de TickData
            'historical_data_slice_df': recent_data_slice_df, # DataFrame para estratégias que preferem
            'spread': current_row_series.get('spread', current_row_series['ask'] - current_row_series['bid']),
            'timestamp': current_row_series.name if isinstance(current_row_series.name, datetime) else pd.Timestamp(current_row_series.name, tz='UTC'), # Garantir que é datetime
            'regime': MarketRegime.TREND,  # Placeholder, regime real viria do RegimeDetector
            'volatility': volatility_val,
            'dom': None,  # DOM não é simulado neste backtester simples
            'session': self._get_simulated_trading_session(current_row_series.name), # Usar timestamp da linha
            'risk_available': self.current_balance * getattr(CONFIG, 'MAX_RISK_PER_TRADE', 0.01) # Simplificado
        }

    def _get_simulated_trading_session(self, timestamp_val: datetime) -> str: # Renomeado
        """Simula a sessão de trading para o backtest."""
        # Pode usar utils.helpers.get_forex_trading_session
        from utils.helpers import get_forex_trading_session # Import local
        return get_forex_trading_session(timestamp_val)


    def _process_and_execute_entry_signal(self, signal_obj: Signal, current_row: pd.Series, # Renomeado
                                       current_ts: datetime, symbol_signal: str): # Renomeado
        """Processa e executa um sinal de entrada."""
        # Verificar se múltiplas posições são permitidas ou se já existe uma para esta estratégia
        if not self.allow_multiple_simultaneous_positions and self._current_open_positions:
            # Se já houver qualquer posição aberta, não abrir outra
            # Ou, se for por estratégia:
            # if any(p.strategy_name == signal_obj.strategy_name for p in self._current_open_positions.values()):
            #     self.logger.debug(f"Sinal de {signal_obj.strategy_name} ignorado: Posição já aberta para esta estratégia.")
            #     return
            self.logger.debug(f"Sinal de {signal_obj.strategy_name} ignorado: Múltiplas posições não permitidas e já existe(m) posição(ões) aberta(s).")
            return


        # Aplicar slippage e determinar preço de entrada real
        entry_price_fill: float # Adicionada tipagem
        if signal_obj.side.lower() == 'buy':
            entry_price_fill = current_row['ask'] + self.slippage_one_side_price
        elif signal_obj.side.lower() == 'sell':
            entry_price_fill = current_row['bid'] - self.slippage_one_side_price
        else: # Lado inválido
            self.logger.error(f"Lado de sinal inválido '{signal_obj.side}' para {signal_obj.strategy_name}.")
            return

        # Calcular tamanho da posição (simplificado aqui, usar PositionSizer em produção)
        # Esta é uma simplificação extrema. O Orchestrator usaria o RiskManager/PositionSizer.
        # Para o backtest, podemos usar um tamanho fixo ou um cálculo de risco simples.
        # Assumindo que signal.stop_loss e signal.take_profit são preços absolutos.
        # Risco em pips:
        if signal_obj.stop_loss is None:
            self.logger.warning(f"Sinal de {signal_obj.strategy_name} sem stop_loss. Usando SL default alto.")
            # Aplicar um SL default muito amplo ou rejeitar o sinal.
            # Para este exemplo, vamos pular o trade se não houver SL.
            return

        pip_unit_exec = 0.0001 if "JPY" not in symbol_signal.upper() else 0.01 # Renomeado
        risk_pips_exec = abs(entry_price_fill - signal_obj.stop_loss) / pip_unit_exec # Renomeado
        if risk_pips_exec < 1.0 : # Risco muito pequeno (ex: < 1 pip), ajustar ou ignorar
            self.logger.debug(f"Risco em pips ({risk_pips_exec:.1f}) muito pequeno para sinal de {signal_obj.strategy_name}. Ignorando.")
            return

        # Usar PositionSizer simulado ou helper
        # position_size_lots_val = calculate_forex_position_size( # Usar helper
        #     account_balance=self.current_balance,
        #     risk_percent_of_balance=0.01, # Ex: 1% risco
        #     stop_loss_pips_val=risk_pips_exec,
        #     pip_value_per_lot_in_acct_ccy=10.0 # Assumindo conta USD e par como EURUSD (valor de $10/pip/lote)
        # )
        # Para este motor de backtest, vamos usar um tamanho fixo para simplificar:
        position_size_lots_val = getattr(CONFIG, 'BACKTEST_DEFAULT_LOT_SIZE', 0.1) # Renomeado

        if position_size_lots_val < 0.01:
            self.logger.warning(f"Tamanho de lote calculado ({position_size_lots_val:.2f}) muito pequeno. Usando mínimo 0.01.")
            position_size_lots_val = 0.01


        # Calcular comissão (por lado, total no fechamento)
        # Comissão = TamanhoLote * TamanhoContrato * PreçoEntrada * TaxaComissãoPercentual
        # Ou, se a taxa for por lote: TamanhoLote * TaxaPorLote
        contract_size_bt = getattr(CONFIG, 'CONTRACT_SIZE', 100000) # Renomeado
        commission_entry = position_size_lots_val * contract_size_bt * self.commission_per_trade_pct # Comissão sobre valor nocional
        # Se self.commission_per_trade_pct for por lote, então:
        # commission_entry = position_size_lots_val * (self.commission_per_trade_pct / 2.0) # Metade no entry

        self._trade_id_counter += 1
        new_trade = BacktestTrade(
            id=f"BTE_{self._trade_id_counter}", # ID do trade do Backtest Engine
            strategy_name=signal_obj.strategy_name,
            symbol=symbol_signal,
            side=signal_obj.side.lower(), # Padronizar para minúsculas
            entry_timestamp_utc=current_ts,
            entry_price_actual=entry_price_fill,
            size_in_lots=position_size_lots_val,
            stop_loss_initial_price=signal_obj.stop_loss, # SL original do sinal
            take_profit_initial_price=signal_obj.take_profit, # TP original do sinal
            commission_total_currency=commission_entry, # Comissão de entrada
            signal_metadata=signal_obj.metadata.copy() if signal_obj.metadata else {}
        )

        self._current_open_positions[new_trade.id] = new_trade
        self.current_balance -= commission_entry # Deduzir comissão do balanço
        self.logger.info(f"Trade ABERTO: {new_trade.id} | {new_trade.side.upper()} {new_trade.size_in_lots} {new_trade.symbol} @ {new_trade.entry_price_actual:.5f} | SL {new_trade.stop_loss_initial_price:.5f} TP {new_trade.take_profit_initial_price:.5f if new_trade.take_profit_initial_price else 'N/A'}")


    def _check_and_process_stops_for_open_positions(self, current_market_row: pd.Series, current_event_ts: datetime): # Renomeado
        """Verifica e processa Stop Loss e Take Profit para todas as posições abertas."""
        # Iterar sobre uma cópia dos IDs para permitir modificação (fechamento) de posições
        for position_id_check in list(self._current_open_positions.keys()): # Renomeado
            trade_to_check = self._current_open_positions.get(position_id_check) # Renomeado
            if not trade_to_check: continue # Posição pode ter sido fechada em iteração anterior

            # Usar bid para checar SL/TP de compras, ask para vendas
            price_to_check_sl_tp: float # Adicionada tipagem
            if trade_to_check.side == 'buy':
                price_to_check_sl_tp = current_market_row['bid'] # Comprador sai vendendo no BID
            else: # sell
                price_to_check_sl_tp = current_market_row['ask']  # Vendedor sai comprando no ASK

            # Atualizar PnL flutuante e MFE/MAE antes de checar stops
            self._update_intra_trade_metrics(trade_to_check, current_market_row['mid']) # Usar mid para PnL flutuante


            # Verificar Stop Loss
            if trade_to_check.stop_loss_initial_price is not None:
                if (trade_to_check.side == 'buy' and price_to_check_sl_tp <= trade_to_check.stop_loss_initial_price) or \
                   (trade_to_check.side == 'sell' and price_to_check_sl_tp >= trade_to_check.stop_loss_initial_price):
                    self.logger.debug(f"Stop Loss atingido para {trade_to_check.id} @ {trade_to_check.stop_loss_initial_price:.5f} (Preço Mkt: {price_to_check_sl_tp:.5f})")
                    self._close_open_position(position_id_check, trade_to_check.stop_loss_initial_price, current_event_ts, "Stop Loss Atingido")
                    continue # Pular para próxima posição, pois esta foi fechada

            # Verificar Take Profit
            if trade_to_check.take_profit_initial_price is not None:
                if (trade_to_check.side == 'buy' and price_to_check_sl_tp >= trade_to_check.take_profit_initial_price) or \
                   (trade_to_check.side == 'sell' and price_to_check_sl_tp <= trade_to_check.take_profit_initial_price):
                    self.logger.debug(f"Take Profit atingido para {trade_to_check.id} @ {trade_to_check.take_profit_initial_price:.5f} (Preço Mkt: {price_to_check_sl_tp:.5f})")
                    self._close_open_position(position_id_check, trade_to_check.take_profit_initial_price, current_event_ts, "Take Profit Atingido")
                    continue


    def _update_intra_trade_metrics(self, trade_obj: BacktestTrade, current_mid_price: float): # Renomeado
        """Atualiza MFE e MAE para um trade aberto."""
        # PnL flutuante atual (sem comissões ainda)
        unrealized_pnl_current: float # Adicionada tipagem
        if trade_obj.side == 'buy':
            unrealized_pnl_current = (current_mid_price - trade_obj.entry_price_actual) * trade_obj.size_in_lots * getattr(CONFIG, 'CONTRACT_SIZE', 100000)
        else: # sell
            unrealized_pnl_current = (trade_obj.entry_price_actual - current_mid_price) * trade_obj.size_in_lots * getattr(CONFIG, 'CONTRACT_SIZE', 100000)

        trade_obj.max_favorable_excursion_pnl = max(trade_obj.max_favorable_excursion_pnl, unrealized_pnl_current)
        trade_obj.max_adverse_excursion_pnl = min(trade_obj.max_adverse_excursion_pnl, unrealized_pnl_current) # MAE é o PnL mais negativo


    def _close_open_position(self, position_id_to_close: str, actual_exit_price: float, # Renomeado
                       exit_ts: datetime, reason_for_exit: str, # Renomeado
                       partial_close_size: Optional[float] = None):
        """Fecha uma posição aberta e registra o trade."""
        if position_id_to_close not in self._current_open_positions:
            logger.warning(f"Tentativa de fechar posição {position_id_to_close} que não está aberta ou já foi fechada.")
            return

        trade_being_closed = self._current_open_positions[position_id_to_close] # Renomeado

        # Aplicar slippage no preço de saída
        # Se comprando para fechar uma venda, usa ASK + slippage
        # Se vendendo para fechar uma compra, usa BID - slippage
        # O actual_exit_price já deve ser o preço de SL/TP, que é o "gatilho".
        # O preço de preenchimento real seria este preço + ou - slippage.
        fill_price_with_slippage: float # Adicionada tipagem
        if trade_being_closed.side == 'buy': # Para fechar uma compra, você VENDE
            fill_price_with_slippage = actual_exit_price - self.slippage_one_side_price
        else: # Para fechar uma venda, você COMPRA
            fill_price_with_slippage = actual_exit_price + self.slippage_one_side_price


        # Calcular PnL
        pnl_gross_currency: float # Adicionada tipagem
        contract_sz_close = getattr(CONFIG, 'CONTRACT_SIZE', 100000) # Renomeado
        size_to_close = partial_close_size if partial_close_size and partial_close_size <= trade_being_closed.size_in_lots else trade_being_closed.size_in_lots


        if trade_being_closed.side == 'buy':
            pnl_gross_currency = (fill_price_with_slippage - trade_being_closed.entry_price_actual) * size_to_close * contract_sz_close
        else: # sell
            pnl_gross_currency = (trade_being_closed.entry_price_actual - fill_price_with_slippage) * size_to_close * contract_sz_close

        # Calcular comissão de saída
        commission_on_exit = size_to_close * contract_sz_close * self.commission_per_trade_pct # Renomeado

        net_pnl_for_this_close = pnl_gross_currency - commission_on_exit # Renomeado

        # Atualizar trade (mesmo que parcial, registramos o PnL desta porção)
        trade_being_closed.exit_timestamp_utc = exit_ts
        trade_being_closed.exit_price_actual = fill_price_with_slippage # Preço de preenchimento real
        trade_being_closed.pnl_currency += net_pnl_for_this_close # Acumular PnL se parcial
        trade_being_closed.commission_total_currency += commission_on_exit # Acumular comissão
        trade_being_closed.exit_trigger_reason = reason_for_exit
        trade_being_closed.duration_in_seconds = int((exit_ts - trade_being_closed.entry_timestamp_utc).total_seconds())
        
        # Calcular PnL em pips para esta porção fechada
        pip_unit_close = 0.0001 if "JPY" not in trade_being_closed.symbol.upper() else 0.01 # Renomeado
        if trade_being_closed.side == 'buy':
            trade_being_closed.pnl_in_pips += (fill_price_with_slippage - trade_being_closed.entry_price_actual) / pip_unit_close
        else:
            trade_being_closed.pnl_in_pips += (trade_being_closed.entry_price_actual - fill_price_with_slippage) / pip_unit_close


        self.current_balance += net_pnl_for_this_close # Atualizar balanço da conta
        logger.info(f"Trade FECHADO: {trade_being_closed.id} ({reason_for_exit}) | Saída @ {fill_price_with_slippage:.5f} | PnL (líq): ${net_pnl_for_this_close:.2f} | Vol Fechado: {size_to_close}")


        if partial_close_size and size_to_close < trade_being_closed.size_in_lots:
            trade_being_closed.size_in_lots -= size_to_close
            # Manter a posição em _current_open_positions com tamanho reduzido
            # O PnL em BacktestTrade deve ser o PnL total realizado até agora para este ID.
            logger.info(f"Fechamento parcial para {trade_being_closed.id}. Tamanho restante: {trade_being_closed.size_in_lots:.2f} lotes.")
        else: # Fechamento total
            self._all_closed_trades.append(trade_being_closed)
            del self._current_open_positions[position_id_to_close]


    def _update_equity_curve_and_hwm(self, current_market_data_row: pd.Series, current_event_timestamp: datetime): # Renomeado
        """Atualiza a curva de equity e o high water mark."""
        current_floating_equity = self.current_balance # Começar com o balanço realizado
        contract_sz_equity = getattr(CONFIG, 'CONTRACT_SIZE', 100000) # Renomeado

        for open_trade_obj in self._current_open_positions.values(): # Renomeado
            current_mid_for_equity = current_market_data_row['mid'] # Renomeado
            unrealized_pnl_trade: float # Adicionada tipagem
            if open_trade_obj.side == 'buy':
                unrealized_pnl_trade = (current_mid_for_equity - open_trade_obj.entry_price_actual) * open_trade_obj.size_in_lots * contract_sz_equity
            else: # sell
                unrealized_pnl_trade = (open_trade_obj.entry_price_actual - current_mid_for_equity) * open_trade_obj.size_in_lots * contract_sz_equity
            current_floating_equity += unrealized_pnl_trade

        self._equity_curve_history.append((current_event_timestamp, current_floating_equity))

        if current_floating_equity > self._current_high_water_mark:
            self._current_high_water_mark = current_floating_equity


    def _compile_final_backtest_results(self, strat_name: str, sym_name: str, # Renomeado
                                       start_dt_final: datetime, end_dt_final: datetime, # Renomeado
                                       backtest_duration_s: float) -> BacktestResults: # Renomeado
        """Compila o objeto BacktestResults final com todas as métricas calculadas."""
        
        # Usar o StrategyScorer para calcular o objeto PerformanceMetrics
        # Converter List[BacktestTrade] para List[Dict] para o scorer
        trades_for_final_score = [t.to_dict() for t in self._all_closed_trades]
        
        total_period_days_final = (end_dt_final - start_dt_final).days if (end_dt_final - start_dt_final).days > 0 else 1.0 # Renomeado

        final_perf_metrics: PerformanceMetrics = self._scorer.calculate_all_performance_metrics( # Renomeado
            trades_list=trades_for_final_score,
            initial_balance=self.initial_balance_bt,
            total_duration_days=total_period_days_final, # Passar duração para anualização
            risk_free_rate_annual=getattr(CONFIG, 'RISK_FREE_RATE_ANNUAL', 0.02)
        )
        # Adicionar a equity curve correta ao objeto PerformanceMetrics se o scorer não o fizer
        # A equity curve do scorer pode ser baseada apenas nos PnLs dos trades.
        # A equity curve do engine (self._equity_curve_history) é baseada em cada tick/barra.
        # Para o drawdown, a curva de equity do engine é mais precisa.
        # Max Drawdown já está sendo calculado pelo scorer usando a equity_curve que ele constrói.
        # Se quisermos usar a nossa equity_curve_history mais granular:
        if self._equity_curve_history:
             equity_values_only = [eq_val for _, eq_val in self._equity_curve_history] # Renomeado
             max_dd_from_engine_equity, _, _ = calculate_max_drawdown(equity_values_only)
             final_perf_metrics.max_drawdown_pct = max_dd_from_engine_equity
             final_perf_metrics.max_drawdown_abs = max_dd_from_engine_equity * self._current_high_water_mark # Aproximação


        results_obj = BacktestResults( # Renomeado
            strategy_name_tested=strat_name,
            symbol_tested=sym_name,
            start_date_data=start_dt_final,
            end_date_data=end_dt_final,
            initial_balance_set=self.initial_balance_bt,
            final_balance_achieved=self.current_balance, # Balanço final realizado
            performance_metrics_calculated=final_perf_metrics, # Objeto completo
            executed_trades_list=list(self._all_closed_trades), # Cópia da lista
            full_equity_curve=list(self._equity_curve_history), # Cópia da lista
            backtest_duration_seconds=backtest_duration_s
        )
        return results_obj


    def plot_backtest_equity_curve(self, results_to_plot: BacktestResults, save_path: Optional[str] = None): # Renomeado
        """Plota a curva de equity do backtest (requer matplotlib)."""
        if not results_to_plot.full_equity_curve:
            logger.info("Curva de equity vazia, nada para plotar.")
            return
        try:
            import matplotlib.pyplot as plt
            # Desempacotar timestamps e valores de equity
            timestamps_plot, equity_values_plot = zip(*results_to_plot.full_equity_curve) # Renomeado

            plt.figure(figsize=(12, 6))
            plt.plot(timestamps_plot, equity_values_plot, label=f"Equity Curve - {results_to_plot.strategy_name_tested}")
            plt.title(f"Curva de Equity: {results_to_plot.strategy_name_tested} ({results_to_plot.symbol_tested})")
            plt.xlabel("Tempo")
            plt.ylabel(f"Equity ({getattr(CONFIG, 'CURRENCY', 'USD')})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_path:
                output_fig_path = Path(save_path) # Renomeado
                output_fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(output_fig_path))
                logger.info(f"Gráfico da curva de equity salvo em: {output_fig_path}")
            else:
                # plt.show() # Comentado para ambientes sem GUI
                logger.info("Plot da curva de equity gerado. Descomente plt.show() ou use save_path para visualizar/salvar.")
            plt.close() # Fechar figura

        except ImportError:
            logger.warning("Matplotlib não instalado. Não é possível plotar resultados do backtest.")
        except Exception as e_plot_equity: # Renomeado
            logger.exception("Erro ao plotar curva de equity do backtest:")