# backtest/engine.py
"""Motor de backtesting para estratégias"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict
import time

from strategies.base_strategy import BaseStrategy, Signal, Position
from utils.logger import setup_logger
from utils.helpers import calculate_pip_value, price_to_pips

logger = setup_logger("backtest_engine")

@dataclass
class BacktestTrade:
    """Estrutura de trade no backtest"""
    id: str
    strategy: str
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""
    duration_seconds: int = 0
    max_profit: float = 0.0
    max_loss: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class BacktestResults:
    """Resultados do backtest"""
    # Métricas básicas
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # PnL
    total_pnl: float = 0.0
    total_commission: float = 0.0
    net_pnl: float = 0.0
    
    # Estatísticas
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Ratios
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    
    # Duração
    avg_trade_duration: float = 0.0
    total_market_exposure: float = 0.0
    
    # Detalhes
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    # Metadados
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_balance: float = 0.0
    final_balance: float = 0.0
    
    def to_dict(self) -> Dict:
        """Converte resultados para dicionário"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'net_pnl': self.net_pnl,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_trade_duration': self.avg_trade_duration,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_return': (self.final_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        }

class BacktestEngine:
    """Motor de backtesting para estratégias de trading"""
    
    def __init__(self):
        self.current_positions: Dict[str, BacktestTrade] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve = []
        self.high_water_mark = 0
        self.trade_counter = 0
        
        # Configurações
        self.allow_multiple_positions = True
        self.use_tick_data = True
        self.calculate_intrabar_metrics = True
        
    def run(self, strategy: BaseStrategy, data: pd.DataFrame,
            initial_balance: float = 10000,
            commission: float = 0.00002,  # 0.2 pips
            slippage: float = 0.00001,    # 0.1 pip
            spread_column: Optional[str] = 'spread') -> BacktestResults:
        """
        Executa backtest síncrono
        
        Args:
            strategy: Estratégia a testar
            data: DataFrame com dados históricos
            initial_balance: Capital inicial
            commission: Comissão por trade (percentual)
            slippage: Slippage estimado
            spread_column: Coluna com spread (se disponível)
        
        Returns:
            Resultados do backtest
        """
        # Resetar estado
        self._reset()
        
        # Configurar
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.commission_rate = commission
        self.slippage = slippage
        
        # Preparar dados
        if 'timestamp' not in data.columns and data.index.name != 'timestamp':
            data = data.copy()
            data['timestamp'] = data.index
        
        # Verificar colunas necessárias
        required_columns = ['bid', 'ask', 'mid']
        for col in required_columns:
            if col not in data.columns:
                if col == 'mid' and 'bid' in data.columns and 'ask' in data.columns:
                    data['mid'] = (data['bid'] + data['ask']) / 2
                else:
                    raise ValueError(f"Coluna '{col}' não encontrada nos dados")
        
        # Adicionar spread se não existir
        if 'spread' not in data.columns:
            data['spread'] = data['ask'] - data['bid']
        
        # Estatísticas
        total_ticks = len(data)
        ticks_processed = 0
        start_time = time.time()
        
        logger.info(f"Iniciando backtest de {strategy.name} com {total_ticks} ticks")
        
        # Processar cada tick
        for idx, row in data.iterrows():
            # Criar contexto de mercado
            market_context = self._create_market_context(data, idx, row)
            
            # Verificar stops de posições abertas
            self._check_stops(row, market_context['timestamp'])
            
            # Processar estratégia
            signal = self._process_strategy_tick(strategy, market_context)
            
            # Executar sinal se houver
            if signal:
                self._execute_signal(signal, row, market_context['timestamp'])
            
            # Atualizar equity
            self._update_equity(row)
            
            # Progresso
            ticks_processed += 1
            if ticks_processed % 10000 == 0:
                progress = (ticks_processed / total_ticks) * 100
                elapsed = time.time() - start_time
                speed = ticks_processed / elapsed
                logger.info(f"Progresso: {progress:.1f}% | "
                           f"Velocidade: {speed:.0f} ticks/s | "
                           f"Trades: {len(self.closed_trades)}")
        
        # Fechar posições abertas
        if self.current_positions:
            last_row = data.iloc[-1]
            for position_id in list(self.current_positions.keys()):
                self._close_position(
                    position_id,
                    last_row['mid'],
                    last_row.name if hasattr(last_row, 'name') else last_row['timestamp'],
                    "Fim do backtest"
                )
        
        # Calcular resultados
        results = self._calculate_results(data)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Backtest concluído em {elapsed_time:.1f}s | "
                   f"Velocidade: {total_ticks/elapsed_time:.0f} ticks/s | "
                   f"Total de trades: {results.total_trades}")
        
        return results
    
    async def run_async(self, strategy: BaseStrategy, data: pd.DataFrame,
                       initial_balance: float = 10000,
                       commission: float = 0.00002,
                       slippage: float = 0.00001) -> BacktestResults:
        """Versão assíncrona do backtest para compatibilidade"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.run,
            strategy,
            data,
            initial_balance,
            commission,
            slippage
        )
    
    def _reset(self):
        """Reseta estado do motor"""
        self.current_positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.high_water_mark = 0
        self.trade_counter = 0
        self.balance = 0
        self.initial_balance = 0
    
    def _create_market_context(self, data: pd.DataFrame, idx: Any, row: pd.Series) -> Dict:
        """Cria contexto de mercado para estratégia"""
        # Índice da posição atual
        current_pos = data.index.get_loc(idx) if hasattr(data.index, 'get_loc') else idx
        
        # Ticks recentes (até 1000)
        lookback = min(current_pos, 1000)
        recent_data = data.iloc[max(0, current_pos - lookback):current_pos + 1]
        
        # Converter para formato de ticks
        recent_ticks = []
        for _, tick_row in recent_data.iterrows():
            tick = type('Tick', (), {
                'timestamp': tick_row.get('timestamp', tick_row.name),
                'bid': tick_row['bid'],
                'ask': tick_row['ask'],
                'mid': tick_row['mid'],
                'spread': tick_row.get('spread', tick_row['ask'] - tick_row['bid']),
                'bid_volume': tick_row.get('bid_volume', 100000),
                'ask_volume': tick_row.get('ask_volume', 100000)
            })
            recent_ticks.append(tick)
        
        # Contexto
        return {
            'tick': recent_ticks[-1] if recent_ticks else None,
            'recent_ticks': recent_ticks,
            'spread': row.get('spread', row['ask'] - row['bid']),
            'timestamp': row.get('timestamp', idx),
            'regime': 'backtest',  # Simplificado
            'volatility': self._calculate_volatility(recent_data),
            'dom': None,  # Não disponível em backtest
            'session': 'backtest',
            'risk_available': self.balance * 0.01  # 1% de risco
        }
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calcula volatilidade simples"""
        if len(data) < 20:
            return 0.01
        
        returns = data['mid'].pct_change().dropna()
        return returns.std() * np.sqrt(252 * 24 * 60 * 60)  # Anualizada
    
    def _process_strategy_tick(self, strategy: BaseStrategy, market_context: Dict) -> Optional[Signal]:
        """Processa tick pela estratégia"""
        try:
            # Calcular indicadores
            strategy.indicators = strategy.calculate_indicators(market_context)
            
            # Gerar sinal
            signal = strategy.generate_signal(market_context)
            
            return signal
            
        except Exception as e:
            logger.error(f"Erro ao processar estratégia: {e}")
            return None
    
    def _execute_signal(self, signal: Signal, row: pd.Series, timestamp: datetime):
        """Executa sinal de trading"""
        # Verificar se pode abrir posição
        if not self.allow_multiple_positions and self.current_positions:
            return
        
        # Aplicar slippage
        if signal.side == 'buy':
            entry_price = row['ask'] + self.slippage
        else:
            entry_price = row['bid'] - self.slippage
        
        # Calcular tamanho (simplificado)
        size = 0.1  # 0.1 lote padrão
        
        # Calcular comissão
        commission = size * 100000 * self.commission_rate
        
        # Criar trade
        self.trade_counter += 1
        trade = BacktestTrade(
            id=f"BT_{self.trade_counter}",
            strategy=signal.strategy_name,
            symbol='EURUSD',
            side=signal.side,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            commission=commission,
            metadata=signal.metadata or {}
        )
        
        # Adicionar à posições abertas
        self.current_positions[trade.id] = trade
        
        # Deduzir comissão do balanço
        self.balance -= commission
    
    def _check_stops(self, row: pd.Series, timestamp: datetime):
        """Verifica stop loss e take profit"""
        for position_id, trade in list(self.current_positions.items()):
            current_price = row['mid']
            
            # Para posições long
            if trade.side == 'buy':
                # Stop Loss
                if row['bid'] <= trade.stop_loss:
                    self._close_position(position_id, trade.stop_loss, timestamp, "Stop Loss")
                # Take Profit
                elif row['bid'] >= trade.take_profit:
                    self._close_position(position_id, trade.take_profit, timestamp, "Take Profit")
                else:
                    # Atualizar max profit/loss
                    profit = (current_price - trade.entry_price) * trade.size * 100000
                    trade.max_profit = max(trade.max_profit, profit)
                    trade.max_loss = min(trade.max_loss, profit)
            
            # Para posições short
            else:
                # Stop Loss
                if row['ask'] >= trade.stop_loss:
                    self._close_position(position_id, trade.stop_loss, timestamp, "Stop Loss")
                # Take Profit
                elif row['ask'] <= trade.take_profit:
                    self._close_position(position_id, trade.take_profit, timestamp, "Take Profit")
                else:
                    # Atualizar max profit/loss
                    profit = (trade.entry_price - current_price) * trade.size * 100000
                    trade.max_profit = max(trade.max_profit, profit)
                    trade.max_loss = min(trade.max_loss, profit)
    
    def _close_position(self, position_id: str, exit_price: float, 
                       timestamp: datetime, reason: str):
        """Fecha posição"""
        if position_id not in self.current_positions:
            return
        
        trade = self.current_positions[position_id]
        
        # Aplicar slippage na saída
        if trade.side == 'buy':
            exit_price -= self.slippage
        else:
            exit_price += self.slippage
        
        # Calcular PnL
        if trade.side == 'buy':
            pnl_pips = price_to_pips(exit_price - trade.entry_price)
            pnl = (exit_price - trade.entry_price) * trade.size * 100000
        else:
            pnl_pips = price_to_pips(trade.entry_price - exit_price)
            pnl = (trade.entry_price - exit_price) * trade.size * 100000
        
        # Deduzir comissão de saída
        commission_exit = trade.size * 100000 * self.commission_rate
        pnl -= commission_exit
        
        # Atualizar trade
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pips = pnl_pips
        trade.commission += commission_exit
        trade.exit_reason = reason
        trade.duration_seconds = int((trade.exit_time - trade.entry_time).total_seconds())
        
        # Mover para trades fechados
        self.closed_trades.append(trade)
        del self.current_positions[position_id]
        
        # Atualizar balanço
        self.balance += pnl
    
    def _update_equity(self, row: pd.Series):
        """Atualiza curva de equity"""
        # Calcular equity atual (balanço + posições abertas)
        equity = self.balance
        
        for trade in self.current_positions.values():
            current_price = row['mid']
            if trade.side == 'buy':
                unrealized_pnl = (current_price - trade.entry_price) * trade.size * 100000
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.size * 100000
            equity += unrealized_pnl
        
        self.equity_curve.append(equity)
        
        # Atualizar high water mark
        if equity > self.high_water_mark:
            self.high_water_mark = equity
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResults:
        """Calcula resultados finais do backtest"""
        results = BacktestResults()
        
        # Informações básicas
        results.start_date = data.index[0] if isinstance(data.index[0], datetime) else data['timestamp'].iloc[0]
        results.end_date = data.index[-1] if isinstance(data.index[-1], datetime) else data['timestamp'].iloc[-1]
        results.initial_balance = self.initial_balance
        results.final_balance = self.balance
        
        # Trades
        results.trades = self.closed_trades
        results.total_trades = len(self.closed_trades)
        
        if results.total_trades > 0:
            # Separar wins e losses
            wins = [t for t in self.closed_trades if t.pnl > 0]
            losses = [t for t in self.closed_trades if t.pnl <= 0]
            
            results.winning_trades = len(wins)
            results.losing_trades = len(losses)
            results.win_rate = results.winning_trades / results.total_trades
            
            # PnL
            results.total_pnl = sum(t.pnl for t in self.closed_trades)
            results.total_commission = sum(t.commission for t in self.closed_trades)
            results.net_pnl = results.total_pnl
            
            # Estatísticas de wins/losses
            if wins:
                results.average_win = np.mean([t.pnl for t in wins])
                results.largest_win = max(t.pnl for t in wins)
            
            if losses:
                results.average_loss = abs(np.mean([t.pnl for t in losses]))
                results.largest_loss = abs(min(t.pnl for t in losses))
            
            # Profit factor
            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
            results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Expectancy
            results.expectancy = results.total_pnl / results.total_trades
            
            # Duração média
            durations = [t.duration_seconds for t in self.closed_trades]
            results.avg_trade_duration = np.mean(durations) if durations else 0
            
            # Sharpe Ratio
            if len(self.equity_curve) > 1:
                equity_array = np.array(self.equity_curve)
                returns = np.diff(equity_array) / equity_array[:-1]
                
                if len(returns) > 0 and np.std(returns) > 0:
                    results.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
                
                # Drawdown
                drawdowns = []
                peak = self.initial_balance
                
                for equity in self.equity_curve:
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak if peak > 0 else 0
                    drawdowns.append(dd)
                
                results.max_drawdown = max(drawdowns) if drawdowns else 0
                results.average_drawdown = np.mean(drawdowns) if drawdowns else 0
                results.drawdown_curve = drawdowns
            
            # Curva de equity
            results.equity_curve = self.equity_curve
            
            # Retornos diários
            if len(self.equity_curve) > 1:
                equity_df = pd.DataFrame({
                    'equity': self.equity_curve,
                    'timestamp': pd.date_range(
                        start=results.start_date,
                        end=results.end_date,
                        periods=len(self.equity_curve)
                    )
                })
                equity_df.set_index('timestamp', inplace=True)
                
                daily_equity = equity_df.resample('D').last().dropna()
                
                if len(daily_equity) > 1:
                    daily_returns = daily_equity['equity'].pct_change().dropna()
                    results.daily_returns = daily_returns.tolist()
                    
                    # Sortino Ratio
                    downside_returns = daily_returns[daily_returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std()
                        if downside_std > 0:
                            results.sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252)
            
            # Calmar Ratio
            if results.max_drawdown > 0:
                annual_return = (results.final_balance / results.initial_balance - 1) * \
                               (252 / ((results.end_date - results.start_date).days or 1))
                results.calmar_ratio = annual_return / results.max_drawdown
        
        return results
    
    def plot_results(self, results: BacktestResults):
        """Plota resultados do backtest (requer matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Equity Curve
            ax1 = axes[0, 0]
            ax1.plot(results.equity_curve)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Ticks')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True)
            
            # Drawdown
            ax2 = axes[0, 1]
            ax2.fill_between(range(len(results.drawdown_curve)), 
                           [d * 100 for d in results.drawdown_curve], 
                           color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Ticks')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True)
            
            # Distribution of Returns
            ax3 = axes[1, 0]
            returns = [t.pnl for t in results.trades]
            if returns:
                ax3.hist(returns, bins=50, alpha=0.7)
                ax3.axvline(x=0, color='black', linestyle='--')
                ax3.set_title('Distribution of Trade Returns')
                ax3.set_xlabel('PnL ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True)
            
            # Trade Duration
            ax4 = axes[1, 1]
            durations = [t.duration_seconds / 3600 for t in results.trades]  # Em horas
            if durations:
                ax4.hist(durations, bins=50, alpha=0.7)
                ax4.set_title('Trade Duration Distribution')
                ax4.set_xlabel('Duration (hours)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib não instalado - não é possível plotar resultados")