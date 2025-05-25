# data/market_data.py
"""Processamento e agregação de dados de mercado"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import asyncio

from utils.logger import setup_logger

logger = setup_logger("market_data")

class MarketDataProcessor:
    """Processador de dados de mercado em tempo real"""
    
    def __init__(self):
        # Buffers para diferentes timeframes
        self.tick_buffers = {}
        self.ohlc_data = {}
        self.volume_profiles = {}
        
        # Configurações
        self.timeframes = ['1T', '5T', '15T', '1H', '4H', '1D']  # pandas freq strings
        self.max_buffer_size = 100000
        
    def process_tick(self, symbol: str, tick: Dict):
        """Processa um tick e atualiza estruturas de dados"""
        # Inicializar buffer se necessário
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.ohlc_data[symbol] = {}
            self.volume_profiles[symbol] = {}
        
        # Adicionar ao buffer
        self.tick_buffers[symbol].append(tick)
        
        # Atualizar OHLC
        self._update_ohlc(symbol, tick)
        
        # Atualizar volume profile
        self._update_volume_profile(symbol, tick)
    
    def _update_ohlc(self, symbol: str, tick: Dict):
        """Atualiza dados OHLC para diferentes timeframes"""
        timestamp = pd.Timestamp(tick['timestamp'])
        price = tick['mid']
        
        for tf in self.timeframes:
            if tf not in self.ohlc_data[symbol]:
                self.ohlc_data[symbol][tf] = {}
            
            # Determinar período atual
            period_start = timestamp.floor(tf)
            period_key = str(period_start)
            
            if period_key not in self.ohlc_data[symbol][tf]:
                # Novo período
                self.ohlc_data[symbol][tf][period_key] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0,
                    'tick_count': 0,
                    'timestamp': period_start
                }
            else:
                # Atualizar período existente
                bar = self.ohlc_data[symbol][tf][period_key]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
                bar['tick_count'] += 1
            
            # Estimar volume (simplificado)
            if 'bid_volume' in tick and 'ask_volume' in tick:
                volume = (tick['bid_volume'] + tick['ask_volume']) / 2
                self.ohlc_data[symbol][tf][period_key]['volume'] += volume
    
    def _update_volume_profile(self, symbol: str, tick: Dict):
        """Atualiza perfil de volume"""
        price = round(tick['mid'], 5)  # Arredondar para nível de preço
        
        if 'current' not in self.volume_profiles[symbol]:
            self.volume_profiles[symbol]['current'] = {}
        
        profile = self.volume_profiles[symbol]['current']
        
        if price not in profile:
            profile[price] = {
                'volume': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'count': 0
            }
        
        # Atualizar volume
        if 'bid_volume' in tick and 'ask_volume' in tick:
            volume = (tick['bid_volume'] + tick['ask_volume']) / 2
            profile[price]['volume'] += volume
            profile[price]['count'] += 1
            
            # Estimar direção (simplificado)
            if tick.get('last_price', tick['mid']) >= tick['ask']:
                profile[price]['buy_volume'] += volume
            else:
                profile[price]['sell_volume'] += volume
    
    def get_ohlc(self, symbol: str, timeframe: str, 
                 periods: int = 100) -> pd.DataFrame:
        """Retorna dados OHLC para timeframe específico"""
        if symbol not in self.ohlc_data or timeframe not in self.ohlc_data[symbol]:
            return pd.DataFrame()
        
        # Converter para DataFrame
        bars = list(self.ohlc_data[symbol][timeframe].values())
        
        # Limitar ao número de períodos solicitados
        if len(bars) > periods:
            bars = bars[-periods:]
        
        if bars:
            df = pd.DataFrame(bars)
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume', 'tick_count']]
            return df
        
        return pd.DataFrame()
    
    def get_volume_profile(self, symbol: str) -> Dict:
        """Retorna perfil de volume atual"""
        if symbol not in self.volume_profiles:
            return {}
        
        profile = self.volume_profiles[symbol].get('current', {})
        
        # Calcular estatísticas
        if profile:
            prices = list(profile.keys())
            volumes = [p['volume'] for p in profile.values()]
            
            # POC (Point of Control) - preço com maior volume
            poc_idx = np.argmax(volumes)
            poc_price = prices[poc_idx]
            
            # Value Area (70% do volume)
            total_volume = sum(volumes)
            sorted_prices = sorted(profile.items(), 
                                 key=lambda x: x[1]['volume'], 
                                 reverse=True)
            
            cumulative_volume = 0
            value_area_prices = []
            
            for price, data in sorted_prices:
                cumulative_volume += data['volume']
                value_area_prices.append(price)
                
                if cumulative_volume >= total_volume * 0.7:
                    break
            
            vah = max(value_area_prices)  # Value Area High
            val = min(value_area_prices)  # Value Area Low
            
            return {
                'profile': profile,
                'poc': poc_price,
                'vah': vah,
                'val': val,
                'total_volume': total_volume
            }
        
        return {}
    
    def calculate_vwap(self, symbol: str, periods: Optional[int] = None) -> float:
        """Calcula VWAP (Volume Weighted Average Price)"""
        if symbol not in self.tick_buffers:
            return 0
        
        ticks = list(self.tick_buffers[symbol])
        
        if periods:
            ticks = ticks[-periods:]
        
        if not ticks:
            return 0
        
        total_volume = 0
        volume_weighted_sum = 0
        
        for tick in ticks:
            price = tick['mid']
            volume = (tick.get('bid_volume', 0) + tick.get('ask_volume', 0)) / 2
            
            volume_weighted_sum += price * volume
            total_volume += volume
        
        if total_volume > 0:
            return volume_weighted_sum / total_volume
        
        return ticks[-1]['mid'] if ticks else 0
    
    def calculate_market_microstructure(self, symbol: str) -> Dict:
        """Calcula métricas de microestrutura de mercado"""
        if symbol not in self.tick_buffers or len(self.tick_buffers[symbol]) < 10:
            return {}
        
        ticks = list(self.tick_buffers[symbol])[-1000:]  # Últimos 1000 ticks
        
        # Calcular métricas
        spreads = [t['spread'] for t in ticks]
        prices = [t['mid'] for t in ticks]
        
        # Volatilidade realizada
        returns = np.diff(np.log(prices))
        realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60 * 60)  # Anualizada
        
        # Spread metrics
        avg_spread = np.mean(spreads)
        spread_vol = np.std(spreads)
        
        # Price impact (simplificado)
        price_changes = np.abs(np.diff(prices))
        avg_price_impact = np.mean(price_changes)
        
        # Eficiência de preço (autocorrelação)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0
        
        return {
            'realized_volatility': realized_vol,
            'average_spread': avg_spread,
            'spread_volatility': spread_vol,
            'average_price_impact': avg_price_impact,
            'price_efficiency': 1 - abs(autocorr),  # 1 = eficiente
            'tick_count': len(ticks)
        }
    
    def detect_liquidity_events(self, symbol: str) -> List[Dict]:
        """Detecta eventos de liquidez (sweeps, absorção, etc)"""
        events = []
        
        if symbol not in self.tick_buffers or len(self.tick_buffers[symbol]) < 100:
            return events
        
        ticks = list(self.tick_buffers[symbol])[-100:]
        
        # Detectar sweep de liquidez (movimento rápido de preço)
        for i in range(10, len(ticks)):
            window = ticks[i-10:i]
            price_change = abs(window[-1]['mid'] - window[0]['mid'])
            time_diff = (pd.Timestamp(window[-1]['timestamp']) - 
                        pd.Timestamp(window[0]['timestamp'])).total_seconds()
            
            if time_diff > 0:
                velocity = price_change / time_diff * 10000  # pips/segundo
                
                if velocity > 10:  # Threshold para sweep
                    events.append({
                        'type': 'liquidity_sweep',
                        'timestamp': window[-1]['timestamp'],
                        'velocity': velocity,
                        'price_change': price_change,
                        'direction': 'up' if window[-1]['mid'] > window[0]['mid'] else 'down'
                    })
        
        # Detectar absorção (grande volume sem movimento de preço)
        volume_profile = self.get_volume_profile(symbol)
        if volume_profile and 'profile' in volume_profile:
            for price, data in volume_profile['profile'].items():
                if data['volume'] > volume_profile['total_volume'] * 0.2:  # 20% do volume
                    # Verificar se preço não se moveu muito
                    price_ticks = [t for t in ticks if abs(t['mid'] - price) < 0.0002]
                    
                    if len(price_ticks) > 20:  # Muitos ticks no mesmo nível
                        events.append({
                            'type': 'absorption',
                            'timestamp': price_ticks[-1]['timestamp'],
                            'price': price,
                            'volume': data['volume'],
                            'tick_count': len(price_ticks)
                        })
        
        return events
    
    def calculate_order_flow_imbalance(self, symbol: str, window: int = 50) -> float:
        """Calcula desequilíbrio de fluxo de ordens"""
        if symbol not in self.tick_buffers:
            return 0
        
        ticks = list(self.tick_buffers[symbol])[-window:]
        
        if not ticks:
            return 0
        
        buy_volume = 0
        sell_volume = 0
        
        for i in range(1, len(ticks)):
            tick = ticks[i]
            prev_tick = ticks[i-1]
            
            volume = (tick.get('bid_volume', 0) + tick.get('ask_volume', 0)) / 2
            
            # Classificar como buy ou sell baseado no movimento
            if tick['mid'] > prev_tick['mid']:
                buy_volume += volume
            elif tick['mid'] < prev_tick['mid']:
                sell_volume += volume
            else:
                # Usar proximidade ao bid/ask
                if abs(tick['mid'] - tick['ask']) < abs(tick['mid'] - tick['bid']):
                    buy_volume += volume
                else:
                    sell_volume += volume
        
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            # Retorna valor entre -1 (sell pressure) e 1 (buy pressure)
            return (buy_volume - sell_volume) / total_volume
        
        return 0
    
    def get_market_summary(self, symbol: str) -> Dict:
        """Retorna resumo completo do mercado"""
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'tick_count': len(self.tick_buffers.get(symbol, [])),
        }
        
        # Último tick
        if symbol in self.tick_buffers and self.tick_buffers[symbol]:
            last_tick = self.tick_buffers[symbol][-1]
            summary['last_price'] = last_tick['mid']
            summary['bid'] = last_tick['bid']
            summary['ask'] = last_tick['ask']
            summary['spread'] = last_tick['spread']
        
        # VWAP
        summary['vwap'] = self.calculate_vwap(symbol)
        
        # Microestrutura
        micro = self.calculate_market_microstructure(symbol)
        summary.update(micro)
        
        # Order flow
        summary['order_flow_imbalance'] = self.calculate_order_flow_imbalance(symbol)
        
        # Volume profile
        vp = self.get_volume_profile(symbol)
        if vp:
            summary['poc'] = vp.get('poc', 0)
            summary['value_area_high'] = vp.get('vah', 0)
            summary['value_area_low'] = vp.get('val', 0)
        
        return summary
    
    def cleanup_old_data(self, hours_to_keep: int = 24):
        """Remove dados antigos para liberar memória"""
        cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
        
        for symbol in list(self.ohlc_data.keys()):
            for timeframe in list(self.ohlc_data[symbol].keys()):
                # Remover barras antigas
                bars = self.ohlc_data[symbol][timeframe]
                for period_key in list(bars.keys()):
                    if bars[period_key]['timestamp'] < cutoff_time:
                        del bars[period_key]
        
        logger.info("Limpeza de dados antigos concluída")