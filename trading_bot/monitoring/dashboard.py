# monitoring/dashboard.py
"""Configuração e templates para dashboards Grafana"""
import json
from typing import Dict, List
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger("dashboard")

class DashboardConfig:
    """Configurador de dashboards Grafana"""
    
    @staticmethod
    def generate_main_dashboard() -> Dict:
        """Gera configuração do dashboard principal"""
        return {
            "dashboard": {
                "id": None,
                "uid": "trading-bot-main",
                "title": "Trading Bot - Dashboard Principal",
                "tags": ["trading", "forex", "eurusd"],
                "timezone": "utc",
                "schemaVersion": 30,
                "version": 1,
                "refresh": "5s",
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "panels": [
                    # Row 1 - Overview
                    DashboardConfig._create_stat_panel(
                        id=1, x=0, y=0, w=4, h=4,
                        title="Saldo da Conta",
                        query='account_balance',
                        unit="currencyUSD",
                        thresholds=[10000, 15000, 20000],
                        colors=["red", "yellow", "green"]
                    ),
                    DashboardConfig._create_stat_panel(
                        id=2, x=4, y=0, w=4, h=4,
                        title="PnL Diário",
                        query='daily_pnl',
                        unit="currencyUSD",
                        thresholds=[-100, 0, 100],
                        colors=["red", "yellow", "green"]
                    ),
                    DashboardConfig._create_gauge_panel(
                        id=3, x=8, y=0, w=4, h=4,
                        title="Drawdown Atual",
                        query='drawdown_current_percent',
                        unit="percent",
                        min=0, max=20,
                        thresholds=[10, 15, 20],
                        colors=["green", "yellow", "red"]
                    ),
                    DashboardConfig._create_stat_panel(
                        id=4, x=12, y=0, w=4, h=4,
                        title="Win Rate",
                        query='avg(strategy_win_rate)',
                        unit="percentunit",
                        thresholds=[0.4, 0.5, 0.6],
                        colors=["red", "yellow", "green"]
                    ),
                    DashboardConfig._create_stat_panel(
                        id=5, x=16, y=0, w=4, h=4,
                        title="Trades Hoje",
                        query='increase(trades_total[24h])',
                        unit="short",
                        sparkline=True
                    ),
                    DashboardConfig._create_state_panel(
                        id=6, x=20, y=0, w=4, h=4,
                        title="Circuit Breaker",
                        query='circuit_breaker_status',
                        mappings={
                            "0": {"text": "CLOSED", "color": "green"},
                            "1": {"text": "OPEN", "color": "red"},
                            "2": {"text": "HALF-OPEN", "color": "yellow"}
                        }
                    ),
                    
                    # Row 2 - Performance Charts
                    DashboardConfig._create_timeseries_panel(
                        id=10, x=0, y=4, w=12, h=8,
                        title="Equity Curve",
                        queries=[
                            {'expr': 'account_equity', 'legendFormat': 'Equity'},
                            {'expr': 'account_balance', 'legendFormat': 'Balance'}
                        ],
                        unit="currencyUSD"
                    ),
                    DashboardConfig._create_timeseries_panel(
                        id=11, x=12, y=4, w=12, h=8,
                        title="PnL por Estratégia",
                        queries=[
                            {'expr': 'sum by(strategy)(rate(trades_total[5m]))', 
                             'legendFormat': '{{strategy}}'}
                        ],
                        unit="ops"
                    ),
                    
                    # Row 3 - Market Conditions
                    DashboardConfig._create_timeseries_panel(
                        id=20, x=0, y=12, w=8, h=6,
                        title="Volatilidade",
                        queries=[
                            {'expr': 'volatility_current', 'legendFormat': 'Volatilidade'}
                        ],
                        unit="percentunit"
                    ),
                    DashboardConfig._create_stat_panel(
                        id=21, x=8, y=12, w=4, h=6,
                        title="Spread Médio",
                        query='spread_average_pips',
                        unit="pips",
                        decimals=1
                    ),
                    DashboardConfig._create_bar_gauge_panel(
                        id=22, x=12, y=12, w=6, h=6,
                        title="Regime de Mercado",
                        query='market_regime',
                        unit="short"
                    ),
                    DashboardConfig._create_gauge_panel(
                        id=23, x=18, y=12, w=6, h=6,
                        title="Confiança do Regime",
                        query='regime_confidence * 100',
                        unit="percent",
                        min=0, max=100
                    ),
                    
                    # Row 4 - Execution Metrics
                    DashboardConfig._create_histogram_panel(
                        id=30, x=0, y=18, w=8, h=6,
                        title="Latência de Execução",
                        query='histogram_quantile(0.95, order_execution_time_ms_bucket)',
                        unit="ms"
                    ),
                    DashboardConfig._create_stat_panel(
                        id=31, x=8, y=18, w=4, h=6,
                        title="Taxa de Preenchimento",
                        query='sum(rate(orders_total{status="filled"}[5m])) / sum(rate(orders_total[5m]))',
                        unit="percentunit",
                        decimals=2
                    ),
                    DashboardConfig._create_table_panel(
                        id=32, x=12, y=18, w=12, h=6,
                        title="Estratégias Ativas",
                        query='strategy_active',
                        columns=['strategy', 'score', 'sharpe_ratio', 'win_rate']
                    ),
                    
                    # Row 5 - System Health
                    DashboardConfig._create_timeseries_panel(
                        id=40, x=0, y=24, w=6, h=6,
                        title="CPU & Memória",
                        queries=[
                            {'expr': 'cpu_usage_percent', 'legendFormat': 'CPU %'},
                            {'expr': 'memory_usage_percent', 'legendFormat': 'Memory %'}
                        ],
                        unit="percent"
                    ),
                    DashboardConfig._create_timeseries_panel(
                        id=41, x=6, y=24, w=6, h=6,
                        title="Latência WebSocket",
                        queries=[
                            {'expr': 'websocket_latency_ms', 'legendFormat': 'Latency'}
                        ],
                        unit="ms"
                    ),
                    DashboardConfig._create_stat_panel(
                        id=42, x=12, y=24, w=4, h=6,
                        title="Ticks/segundo",
                        query='rate(ticks_processed_total[1m])',
                        unit="ops",
                        sparkline=True
                    ),
                    DashboardConfig._create_heatmap_panel(
                        id=43, x=16, y=24, w=8, h=6,
                        title="Distribuição de Latência",
                        query='order_execution_time_ms_bucket'
                    )
                ]
            }
        }
    
    @staticmethod
    def _create_stat_panel(id: int, x: int, y: int, w: int, h: int,
                          title: str, query: str, unit: str = "short",
                          thresholds: List[float] = None, colors: List[str] = None,
                          decimals: int = 2, sparkline: bool = False) -> Dict:
        """Cria painel de estatística"""
        panel = {
            "id": id,
            "type": "stat",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "orientation": "auto",
                "textMode": "value_and_name",
                "graphMode": "area" if sparkline else "none",
                "justifyMode": "center"
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "decimals": decimals
                }
            }
        }
        
        if thresholds and colors:
            panel["fieldConfig"]["defaults"]["thresholds"] = {
                "mode": "absolute",
                "steps": [
                    {"color": colors[i], "value": val if i == 0 else thresholds[i-1]}
                    for i, val in enumerate([float('-inf')] + thresholds)
                ]
            }
        
        return panel
    
    @staticmethod
    def _create_gauge_panel(id: int, x: int, y: int, w: int, h: int,
                           title: str, query: str, unit: str,
                           min: float, max: float,
                           thresholds: List[float] = None,
                           colors: List[str] = None) -> Dict:
        """Cria painel gauge"""
        return {
            "id": id,
            "type": "gauge",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "orientation": "auto",
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "min": min,
                    "max": max,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": colors[i], "value": val if i == 0 else thresholds[i-1]}
                            for i, val in enumerate([min] + thresholds)
                        ] if thresholds and colors else []
                    }
                }
            }
        }
    
    @staticmethod
    def _create_timeseries_panel(id: int, x: int, y: int, w: int, h: int,
                                title: str, queries: List[Dict],
                                unit: str = "short") -> Dict:
        """Cria painel de série temporal"""
        return {
            "id": id,
            "type": "timeseries",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [
                {
                    "expr": q['expr'],
                    "refId": chr(65 + i),  # A, B, C...
                    "legendFormat": q.get('legendFormat', '')
                }
                for i, q in enumerate(queries)
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "gradientMode": "opacity",
                        "spanNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {
                            "mode": "none",
                            "group": "A"
                        },
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "scaleDistribution": {
                            "type": "linear"
                        }
                    }
                }
            },
            "options": {
                "tooltip": {
                    "mode": "multi",
                    "sort": "desc"
                },
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "calcs": ["mean", "last"]
                }
            }
        }
    
    @staticmethod
    def _create_state_panel(id: int, x: int, y: int, w: int, h: int,
                           title: str, query: str, mappings: Dict) -> Dict:
        """Cria painel de estado"""
        return {
            "id": id,
            "type": "state-timeline",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "lineWidth": 0,
                        "fillOpacity": 70
                    },
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                str(k): v
                            }
                        }
                        for k, v in mappings.items()
                    ]
                }
            }
        }
    
    @staticmethod
    def _create_bar_gauge_panel(id: int, x: int, y: int, w: int, h: int,
                               title: str, query: str, unit: str) -> Dict:
        """Cria painel bar gauge"""
        return {
            "id": id,
            "type": "bargauge",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
                "instant": True
            }],
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "orientation": "horizontal",
                "displayMode": "gradient",
                "showUnfilled": True
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "min": 0,
                    "max": 1
                }
            }
        }
    
    @staticmethod
    def _create_histogram_panel(id: int, x: int, y: int, w: int, h: int,
                               title: str, query: str, unit: str) -> Dict:
        """Cria painel histograma"""
        return {
            "id": id,
            "type": "histogram",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
                "format": "heatmap"
            }],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "lineWidth": 1,
                        "fillOpacity": 80
                    }
                }
            },
            "options": {
                "bucketDataBound": "auto"
            }
        }
    
    @staticmethod
    def _create_table_panel(id: int, x: int, y: int, w: int, h: int,
                           title: str, query: str, columns: List[str]) -> Dict:
        """Cria painel tabela"""
        return {
            "id": id,
            "type": "table",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
                "format": "table",
                "instant": True
            }],
            "options": {
                "showHeader": True,
                "sortBy": [{
                    "displayName": columns[1] if len(columns) > 1 else columns[0],
                    "desc": True
                }]
            }
        }
    
    @staticmethod
    def _create_heatmap_panel(id: int, x: int, y: int, w: int, h: int,
                             title: str, query: str) -> Dict:
        """Cria painel heatmap"""
        return {
            "id": id,
            "type": "heatmap",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
                "format": "heatmap"
            }],
            "options": {
                "calculate": True,
                "calculation": {
                    "xBuckets": {
                        "mode": "count",
                        "value": "20"
                    }
                },
                "color": {
                    "mode": "spectrum",
                    "scheme": "Spectral",
                    "steps": 128
                },
                "exemplars": {
                    "color": "rgba(255,0,255,0.7)"
                },
                "filterValues": {
                    "le": 1e-9
                },
                "rowsFrame": {
                    "layout": "auto"
                },
                "tooltip": {
                    "show": True,
                    "yHistogram": False
                },
                "yAxis": {
                    "axisPlacement": "left",
                    "reverse": False
                }
            }
        }
    
    @staticmethod
    def save_dashboard(dashboard_config: Dict, filepath: str):
        """Salva configuração do dashboard em arquivo"""
        with open(filepath, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        logger.info(f"Dashboard salvo em: {filepath}")
    
    @staticmethod
    def generate_alert_rules() -> List[Dict]:
        """Gera regras de alerta para Grafana"""
        return [
            {
                "uid": "high_drawdown",
                "title": "High Drawdown Alert",
                "condition": "drawdown_current_percent > 15",
                "data": [{
                    "refId": "A",
                    "queryType": "",
                    "model": {
                        "expr": "drawdown_current_percent",
                        "refId": "A"
                    }
                }],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": "1m",
                "annotations": {
                    "summary": "Drawdown exceeded 15%"
                }
            },
            {
                "uid": "circuit_breaker_open",
                "title": "Circuit Breaker Open",
                "condition": "circuit_breaker_status == 1",
                "data": [{
                    "refId": "A",
                    "queryType": "",
                    "model": {
                        "expr": "circuit_breaker_status",
                        "refId": "A"
                    }
                }],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": "10s",
                "annotations": {
                    "summary": "Circuit breaker has been triggered"
                }
            },
            {
                "uid": "low_balance",
                "title": "Low Account Balance",
                "condition": "account_balance < 5000",
                "data": [{
                    "refId": "A",
                    "queryType": "",
                    "model": {
                        "expr": "account_balance",
                        "refId": "A"
                    }
                }],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": "5m",
                "annotations": {
                    "summary": "Account balance below $5000"
                }
            }
        ]