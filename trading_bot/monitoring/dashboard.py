# monitoring/dashboard.py
"""Configuração e templates para dashboards Grafana"""
import json
from typing import Dict, List, Optional, Any # Adicionado Any e Optional
from datetime import datetime # Não usado diretamente, mas comum em dashboards

from utils.logger import setup_logger

logger = setup_logger("dashboard_config") # Renomeado logger para evitar conflito com o logger do módulo dashboard

class DashboardConfig:
    """Configurador de dashboards Grafana"""

    @staticmethod
    def generate_main_dashboard() -> Dict[str, Any]: # Usar Any
        """Gera configuração do dashboard principal"""
        # Define um UID único para o dashboard se não for para ser None
        dashboard_uid = "trading-bot-main-v2" # Exemplo de UID, pode ser gerado ou fixo

        return {
            # "id": None, # 'id' é geralmente gerenciado pelo Grafana, 'uid' é para identificação única
            "uid": dashboard_uid,
            "title": "Trading Bot - Dashboard Principal",
            "tags": ["trading", "forex", "eurusd", CONFIG.SYMBOL], # Adicionado CONFIG.SYMBOL
            "timezone": "browser", # Usar 'browser' para timezone local do usuário ou 'utc'
            "schemaVersion": 36, # Versões mais recentes do Grafana usam schemas maiores
            "version": 2, # Incrementar versão se fizer mudanças significativas
            "refresh": "10s", # Ajustar refresh rate conforme necessidade (5s pode ser muito frequente)
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "panels": [
                # === Row 1: Visão Geral da Conta e Risco ===
                DashboardConfig._create_stat_panel(
                    panel_id=1, grid_pos={"x": 0, "y": 0, "w": 4, "h": 4}, # Usar nomes mais descritivos
                    title="Saldo da Conta",
                    prom_query='account_balance{job="trading_bot"}', # Adicionar job label para filtrar
                    unit="currencyUSD",
                    thresholds_config={'mode': 'absolute', 'steps': [
                        {"color": "red", "value": 0}, # Abaixo de 0 é crítico (improvável, mas para exemplo)
                        {"color": "orange", "value": float(CONFIG.INITIAL_BALANCE) * 0.8 if hasattr(CONFIG, 'INITIAL_BALANCE') else 8000},
                        {"color": "green", "value": float(CONFIG.INITIAL_BALANCE) if hasattr(CONFIG, 'INITIAL_BALANCE') else 10000}
                    ]},
                    color_mode="background" # Colorir o fundo
                ),
                DashboardConfig._create_stat_panel(
                    panel_id=2, grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
                    title="PnL Diário ($)",
                    prom_query='daily_pnl{job="trading_bot"}',
                    unit="currencyUSD",
                    thresholds_config={'mode': 'absolute', 'steps': [
                        {"color": "red", "value": None}, # Default para o primeiro (geralmente o mais baixo)
                        {"color": "orange", "value": -float(CONFIG.DAILY_LOSS_LIMIT) * float(CONFIG.INITIAL_BALANCE) * 0.5 if hasattr(CONFIG, 'DAILY_LOSS_LIMIT') else -150},
                        {"color": "green", "value": 0}
                    ]},
                    color_mode="value"
                ),
                DashboardConfig._create_gauge_panel(
                    panel_id=3, grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
                    title="Drawdown Atual (%)",
                    prom_query='drawdown_current_percent{job="trading_bot"}',
                    unit="percent",
                    min_val=0, max_val=float(CONFIG.MAX_DRAWDOWN) * 100 if hasattr(CONFIG, 'MAX_DRAWDOWN') else 20, # Max do gauge é o Max DD configurado
                    thresholds_config={'mode': 'absolute', 'steps': [
                        {"color": "green", "value": 0},
                        {"color": "orange", "value": float(CONFIG.WARNING_DRAWDOWN) * 100 if hasattr(CONFIG, 'WARNING_DRAWDOWN') else 10},
                        {"color": "red", "value": float(CONFIG.MAX_DRAWDOWN) * 0.9 * 100 if hasattr(CONFIG, 'MAX_DRAWDOWN') else 18} # Vermelho perto do limite
                    ]}
                ),
                DashboardConfig._create_stat_panel(
                    panel_id=4, grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
                    title="Win Rate (Global)",
                    # Assumindo que strategy_win_rate é por estratégia, precisamos de um global ou média
                    prom_query='avg_over_time(bot_global_win_rate[1h]) * 100 or irate(trades_total{job="trading_bot", result="win"}[5m]) / irate(trades_total{job="trading_bot"}[5m]) * 100',
                    unit="percent", # unit é 'percent' (0-100) ou 'percentunit' (0.0-1.0)
                    decimals=1,
                     thresholds_config={'mode': 'absolute', 'steps': [
                        {"color": "red", "value": 0},
                        {"color": "orange", "value": 40},
                        {"color": "green", "value": 55}
                    ]}
                ),
                DashboardConfig._create_stat_panel(
                    panel_id=5, grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
                    title="Trades (Última Hora)",
                    prom_query='sum(increase(trades_total{job="trading_bot"}[1h]))',
                    unit="short", # Nenhum decimal para contagem
                    decimals=0,
                    sparkline_on=True # Renomeado
                ),
                DashboardConfig._create_stat_panel( # Usar stat panel para melhor visualização de texto
                    panel_id=6, grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
                    title="Circuit Breaker",
                    prom_query='circuit_breaker_status{job="trading_bot"}',
                    unit="string", # String para status
                    value_mappings=[ # Usar value_mappings para stat panel
                        {"value": "0", "text": "FECHADO (OK)", "color": "green"},
                        {"value": "1", "text": "ABERTO (PARADO)", "color": "red"},
                        {"value": "2", "text": "TESTE (Limitado)", "color": "orange"}
                    ]
                ),

                # === Row 2: Curva de Equity e Performance de Estratégias ===
                DashboardConfig._create_timeseries_panel(
                    panel_id=10, grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
                    title="Curva de Equity e Balanço",
                    queries=[
                        {'expr': 'account_equity{job="trading_bot"}', 'legendFormat': 'Equity'},
                        {'expr': 'account_balance{job="trading_bot"}', 'legendFormat': 'Balanço'}
                    ],
                    unit="currencyUSD",
                    fill_opacity=10,
                    line_width=2
                ),
                DashboardConfig._create_bar_chart_panel( # Melhor como bar chart para PnL por estratégia
                    panel_id=11, grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
                    title="PnL Total por Estratégia (Últimas 24h)",
                    queries=[
                        # Assumindo que trades_total tem um label 'pnl_value' ou similar,
                        # ou que há uma métrica 'strategy_pnl_total'.
                        # Exemplo com uma métrica gauge 'strategy_pnl_total_usd'
                        {'expr': 'sum by(strategy) (increase(strategy_pnl_sum_usd{job="trading_bot"}[24h]))',
                         'legendFormat': '{{strategy}}'}
                    ],
                    unit="currencyUSD",
                    orientation="horizontal"
                ),

                # === Row 3: Condições de Mercado ===
                DashboardConfig._create_timeseries_panel(
                    panel_id=20, grid_pos={"x": 0, "y": 12, "w": 8, "h": 6},
                    title="Volatilidade Atual (ex: ATR %)", # Ser mais específico
                    queries=[
                        {'expr': 'volatility_current{job="trading_bot"} * 100', 'legendFormat': 'Volatilidade (%)'}
                        # Se 'volatility_current' for ATR, precisa ser normalizado ou ter unidade clara
                    ],
                    unit="percent" # Ajustar unidade
                ),
                DashboardConfig._create_stat_panel(
                    panel_id=21, grid_pos={"x": 8, "y": 12, "w": 4, "h": 6},
                    title="Spread Médio (EURUSD)",
                    prom_query='spread_average_pips{job="trading_bot", symbol="EURUSD"}', # Adicionar filtro de símbolo
                    unit="pips", # Unidade customizada 'pips'
                    decimals=2 # Spread geralmente tem 1-2 decimais em pips
                ),
                 DashboardConfig._create_table_panel( # Tabela para mostrar regimes
                    panel_id=22, grid_pos={"x": 12, "y": 12, "w": 6, "h": 6},
                    title="Regime de Mercado",
                    queries=[{'expr': 'market_regime{job="trading_bot"}', 'legendFormat': '{{regime}}'}],
                    # A query acima retorna múltiplas séries, uma para cada label 'regime'.
                    # Precisamos de transformações para mostrar o regime ativo.
                    # Exemplo de transformação (pode precisar ajustar query ou usar transformações no Grafana):
                    # 'topk(1, market_regime{job="trading_bot"} == 1)'
                    # Ou, se market_regime for uma string com o regime atual:
                    # 'market_regime_label{job="trading_bot"}' (métrica com label do regime)
                    # Esta parte é complexa e depende de como 'market_regime' é exposto.
                    # Para simplificar, se 'market_regime' tiver valor 1 para o ativo:
                    transformations=[{
                        "id": "filterByValue",
                        "options": {
                            "filterType": "include", "type": "match",
                            "match": "value", "value": 1
                        }
                    }, {
                        "id": "organize",
                        "options": {"excludeByName": {}, "indexByName": {}, "renameByName": {"Value": "Regime Ativo"}}
                    }],
                    value_field_name="Regime Ativo" # Para qual campo da query aplicar mappings
                ),
                DashboardConfig._create_gauge_panel(
                    panel_id=23, grid_pos={"x": 18, "y": 12, "w": 6, "h": 6},
                    title="Confiança do Regime (%)",
                    prom_query='regime_confidence{job="trading_bot"} * 100',
                    unit="percent",
                    min_val=0, max_val=100,
                    thresholds_config={'mode': 'absolute', 'steps': [
                        {"color": "red", "value": 0},
                        {"color": "orange", "value": 50},
                        {"color": "green", "value": 75}
                    ]}
                ),

                # === Row 4: Métricas de Execução ===
                DashboardConfig._create_timeseries_panel( # Melhor que histograma para tendência de latência
                    panel_id=30, grid_pos={"x": 0, "y": 18, "w": 8, "h": 6},
                    title="Latência de Execução de Ordem (p95)",
                    queries=[
                        {'expr': 'histogram_quantile(0.95, sum(rate(order_execution_time_ms_bucket{job="trading_bot"}[5m])) by (le))',
                         'legendFormat': 'Latência p95'}
                    ],
                    unit="ms"
                ),
                DashboardConfig._create_stat_panel(
                    panel_id=31, grid_pos={"x": 8, "y": 18, "w": 4, "h": 6},
                    title="Taxa de Preenchimento (Fill Rate)",
                    # Query ajustada para taxa de preenchimento
                    prom_query='(sum(rate(orders_total{job="trading_bot", status="filled"}[5m])) / sum(rate(orders_total{job="trading_bot", status=~"filled|rejected|cancelled"}[5m])) * 100) or vector(0)',
                    unit="percent",
                    decimals=2,
                     thresholds_config={'mode': 'absolute', 'steps': [
                        {"color": "red", "value": 0},
                        {"color": "orange", "value": 90},
                        {"color": "green", "value": 98}
                    ]}
                ),
                DashboardConfig._create_table_panel(
                    panel_id=32, grid_pos={"x": 12, "y": 18, "w": 12, "h": 6},
                    title="Performance por Estratégia (Ativas)",
                    # Query para buscar score, sharpe, win_rate das estratégias ativas
                    queries=[{
                        'expr': 'strategy_score{job="trading_bot", active="1"}', 'legendFormat': 'Score - {{strategy}}', 'refId': 'A'
                    },{
                        'expr': 'strategy_sharpe_ratio{job="trading_bot", active="1"}', 'legendFormat': 'Sharpe - {{strategy}}', 'refId': 'B'
                    },{
                        'expr': 'strategy_win_rate{job="trading_bot", active="1"} * 100', 'legendFormat': 'Win Rate (%) - {{strategy}}', 'refId': 'C'
                    }],
                    # Usar transformações para juntar as métricas por estratégia
                    transformations=[{
                        "id": "merge", # Merge multiple series
                        "options": {}
                    },{
                        "id": "organize", # Renomear e reordenar
                        "options": {
                            "indexByName": {},
                            "renameByName": {
                                "Value #A": "Score", "Value #B": "Sharpe Ratio", "Value #C": "Win Rate (%)",
                                "Time": "Strategy" # O label 'strategy' se tornará o 'Time' após o merge
                            },
                             "excludeByName": {"Time": True} # Excluir a coluna Time original se não for strategy
                        }
                    }],
                    column_styles=[ # Exemplo de formatação de colunas
                        {"unit": "short", "decimals": 2, "pattern": "Score"},
                        {"unit": "short", "decimals": 2, "pattern": "Sharpe Ratio"},
                        {"unit": "percent", "decimals": 1, "pattern": "Win Rate (%)"}
                    ]
                ),

                # === Row 5: Saúde do Sistema ===
                DashboardConfig._create_timeseries_panel(
                    panel_id=40, grid_pos={"x": 0, "y": 24, "w": 8, "h": 6}, # Aumentado w
                    title="CPU & Memória (%)",
                    queries=[
                        {'expr': 'cpu_usage_percent{job="trading_bot"}', 'legendFormat': 'CPU Uso (%)'},
                        {'expr': 'memory_usage_percent{job="trading_bot"}', 'legendFormat': 'Memória Uso (%)'}
                    ],
                    unit="percent",
                    max_data_points=200
                ),
                DashboardConfig._create_timeseries_panel(
                    panel_id=41, grid_pos={"x": 8, "y": 24, "w": 8, "h": 6}, # Aumentado w
                    title="Latência WebSocket (Feed)",
                    queries=[
                        # Assumindo que websocket_latency_ms é um histograma ou summary
                        {'expr': 'histogram_quantile(0.90, sum(rate(websocket_latency_ms_bucket{job="trading_bot"}[5m])) by (le))', 'legendFormat': 'Latência p90 (ms)'},
                        {'expr': 'histogram_quantile(0.99, sum(rate(websocket_latency_ms_bucket{job="trading_bot"}[5m])) by (le))', 'legendFormat': 'Latência p99 (ms)'}
                    ],
                    unit="ms",
                    max_data_points=200
                ),
                DashboardConfig._create_stat_panel(
                    panel_id=42, grid_pos={"x": 16, "y": 24, "w": 8, "h": 6}, # Aumentado w
                    title="Ticks Processados/segundo (EURUSD)",
                    prom_query='sum(rate(ticks_processed_total{job="trading_bot", symbol="EURUSD"}[1m]))', # Média por minuto
                    unit="ops", # Operações por segundo
                    decimals=1,
                    sparkline_on=True
                )
                # Removido heatmap para simplificar, pode ser adicionado depois se necessário.
            ]
        }


    @staticmethod
    def _create_base_panel(panel_id: int, panel_type: str, title: str, grid_pos: Dict[str, int],
                           queries: Optional[List[Dict[str, str]]] = None,
                           targets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Cria a estrutura base de um painel."""
        panel_struct = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "gridPos": grid_pos,
        }
        if targets: # Se targets já estiverem formatados (mais controle)
            panel_struct["targets"] = targets
        elif queries: # Se apenas queries simples forem fornecidas
             panel_struct["targets"] = [
                {"expr": q_item['expr'], "refId": chr(65 + i), "legendFormat": q_item.get('legendFormat', '')}
                for i, q_item in enumerate(queries)
            ]
        return panel_struct

    @staticmethod
    def _apply_field_config(panel: Dict[str, Any], unit: str = "short", decimals: Optional[int] = None,
                            min_val: Optional[float] = None, max_val: Optional[float] = None,
                            thresholds_config: Optional[Dict[str, Any]] = None,
                            value_mappings: Optional[List[Dict[str, Any]]] = None,
                            column_styles: Optional[List[Dict[str, Any]]] = None): # Para tabelas
        """Aplica configurações de campo a um painel."""
        if "fieldConfig" not in panel:
            panel["fieldConfig"] = {"defaults": {}}

        defaults = panel["fieldConfig"]["defaults"]
        defaults["unit"] = unit
        if decimals is not None:
            defaults["decimals"] = decimals
        if min_val is not None:
            defaults["min"] = min_val
        if max_val is not None:
            defaults["max"] = max_val

        if thresholds_config:
            defaults["thresholds"] = thresholds_config
        else: # Garantir que thresholds default não seja None se não especificado
            defaults["thresholds"] = {"mode": "absolute", "steps": [{"color": "green", "value": None}]}


        if value_mappings: # Para stat panel com mapeamento de texto/cor
            defaults["mappings"] = [{
                "type": "value",
                "options": {mapping['value']: {"text": mapping['text'], "color": mapping['color']} for mapping in value_mappings}
            }]

        if column_styles: # Para painéis de tabela
            if "overrides" not in panel["fieldConfig"]:
                panel["fieldConfig"]["overrides"] = []
            for style in column_styles:
                panel["fieldConfig"]["overrides"].append({
                    "matcher": {"id": "byName", "options": style["pattern"]},
                    "properties": [
                        {"id": "unit", "value": style.get("unit", unit)},
                        {"id": "decimals", "value": style.get("decimals", decimals)}
                    ]
                })


    @staticmethod
    def _create_stat_panel(panel_id: int, grid_pos: Dict[str, int], title: str, prom_query: str,
                          unit: str = "short", thresholds_config: Optional[Dict[str, Any]] = None,
                          color_mode: str = "value", # "value" ou "background"
                          decimals: Optional[int] = 2, sparkline_on: bool = False, # Renomeado
                          value_mappings: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Cria painel de estatística (stat)"""
        panel = DashboardConfig._create_base_panel(panel_id, "stat", title, grid_pos, queries=[{"expr": prom_query}])
        DashboardConfig._apply_field_config(panel, unit, decimals, thresholds_config=thresholds_config, value_mappings=value_mappings)

        panel["options"] = {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
            "orientation": "auto",
            "textMode": "auto", # Deixar auto para Grafana decidir
            "colorMode": color_mode,
            "graphMode": "area" if sparkline_on else "none",
            "justifyMode": "auto"
        }
        return panel

    @staticmethod
    def _create_gauge_panel(panel_id: int, grid_pos: Dict[str, int], title: str, prom_query: str, unit: str,
                           min_val: float, max_val: float, # Renomeado min/max
                           thresholds_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Usar thresholds_config
        """Cria painel gauge"""
        panel = DashboardConfig._create_base_panel(panel_id, "gauge", title, grid_pos, queries=[{"expr": prom_query}])
        DashboardConfig._apply_field_config(panel, unit, min_val=min_val, max_val=max_val, thresholds_config=thresholds_config)

        panel["options"] = {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
            "orientation": "auto",
            "showThresholdLabels": False,
            "showThresholdMarkers": True
        }
        return panel

    @staticmethod
    def _create_timeseries_panel(panel_id: int, grid_pos: Dict[str, int], title: str, queries: List[Dict[str,str]],
                                unit: str = "short", fill_opacity: int = 0, line_width: int = 1, # Defaults ajustados
                                legend_calcs: Optional[List[str]] = None, max_data_points: Optional[int] = None) -> Dict[str, Any]:
        """Cria painel de série temporal (timeseries)"""
        panel = DashboardConfig._create_base_panel(panel_id, "timeseries", title, grid_pos, queries=queries)
        DashboardConfig._apply_field_config(panel, unit)

        panel["fieldConfig"]["defaults"]["custom"] = { # fieldConfig.defaults.custom
            "drawStyle": "line",
            "lineInterpolation": "linear", # Linear é geralmente mais performático
            "lineWidth": line_width,
            "fillOpacity": fill_opacity,
            "gradientMode": "none", # None é mais simples
            "spanNulls": False, # ou 'connected'
            "showPoints": "auto",
            "pointSize": 5,
            "stacking": {"mode": "none", "group": "A"},
            "axisPlacement": "auto",
            "axisLabel": "",
            "scaleDistribution": {"type": "linear"}
        }
        panel["options"] = {
            "tooltip": {"mode": "multi", "sort": "none"}, # sort: none, asc, desc
            "legend": {
                "displayMode": "list", # table, list, hidden
                "placement": "bottom",
                "calcs": legend_calcs if legend_calcs else ["lastNotNull", "mean", "max"]
            }
        }
        if max_data_points:
             panel["maxDataPoints"] = max_data_points
        return panel

    @staticmethod
    def _create_bar_chart_panel(panel_id: int, grid_pos: Dict[str, int], title: str,
                               queries: List[Dict[str, str]], unit: str = "short",
                               orientation: str = "auto", # auto, horizontal, vertical
                               stacking: str = "none") -> Dict[str, Any]: # none, normal, percent
        """Cria painel de gráfico de barras (barchart)"""
        panel = DashboardConfig._create_base_panel(panel_id, "barchart", title, grid_pos, queries=queries)
        DashboardConfig._apply_field_config(panel, unit)
        panel["options"] = {
            "orientation": orientation,
            "stacking": stacking,
            "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["sum"]},
            "tooltip": {"mode": "multi", "sort": "none"}
        }
        return panel


    @staticmethod
    def _create_table_panel(panel_id: int, grid_pos: Dict[str, int], title: str,
                           queries: List[Dict[str, str]], # Pode ter múltiplas queries e usar transformações
                           transformations: Optional[List[Dict[str, Any]]] = None,
                           column_styles: Optional[List[Dict[str, Any]]] = None,
                           value_field_name: Optional[str] = "Value") -> Dict[str, Any]: # Para aplicar value mappings
        """Cria painel tabela"""
        panel = DashboardConfig._create_base_panel(panel_id, "table", title, grid_pos, targets=[ # Usar targets para mais controle
            {**q, "datasource": {"type": "prometheus", "uid": "prometheus"}, "format": "table", "instant": True} # Adicionar datasource e format
             for q in queries
        ])
        DashboardConfig._apply_field_config(panel, column_styles=column_styles) # Aplicar estilos de coluna

        panel["options"] = {
            "showHeader": True,
            "sortBy": [{"displayName": value_field_name, "desc": True}] if value_field_name else [],
            "footer": {"show": False, "reducer": ["sum"], "fields": []}
        }
        if transformations:
            panel["transformations"] = transformations
        return panel

    # Removidos _create_state_panel, _create_bar_gauge_panel, _create_histogram_panel, _create_heatmap_panel
    # por brevidade e porque os métodos acima (stat, gauge, timeseries, barchart, table) cobrem os casos de uso
    # principais do dashboard original. Se forem realmente necessários, podem ser reimplementados seguindo o padrão.

    @staticmethod
    def save_dashboard_json(dashboard_config: Dict[str, Any], filepath: str): # Renomeado de save_dashboard
        """Salva configuração do dashboard em arquivo JSON."""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True) # Garantir que o diretório exista
            with open(output_path, 'w') as f:
                json.dump(dashboard_config, f, indent=2, sort_keys=True) # Adicionado sort_keys
            logger.info(f"Dashboard salvo em: {output_path}")
        except Exception as e:
            logger.exception(f"Erro ao salvar dashboard em {filepath}:")


    @staticmethod
    def generate_alerting_rules_for_prometheus(rules_filepath: str): # Renomeado e com filepath
        """Gera regras de alerta em formato YAML para Prometheus Alertmanager."""
        # Esta função geraria um arquivo .rules.yml para o Prometheus
        # A estrutura exata depende da sua configuração do Alertmanager.
        # Exemplo simplificado:
        alert_rules = {
            "groups": [{
                "name": "TradingBotAlerts",
                "rules": [
                    {
                        "alert": "HighDrawdown",
                        "expr": 'drawdown_current_percent{job="trading_bot"} > 15', # 15%
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Drawdown Alto no Trading Bot ({{ $labels.job }})",
                            "description": "Drawdown atual é {{ $value | printf \"%.2f\" }}%, excedendo o limite de 15%."
                        }
                    },
                    {
                        "alert": "CircuitBreakerOpen",
                        "expr": 'circuit_breaker_status{job="trading_bot"} == 1', # Status 1 = Open
                        "for": "30s",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Circuit Breaker do Trading Bot Acionado ({{ $labels.job }})",
                            "description": "O circuit breaker foi acionado, trading está PARADO."
                        }
                    },
                    {
                        "alert": "LowAccountBalance",
                        "expr": f'account_balance{{job="trading_bot"}} < {float(CONFIG.INITIAL_BALANCE) * 0.5 if hasattr(CONFIG, "INITIAL_BALANCE") else 5000}', # Ex: < 50% do inicial
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Saldo Baixo na Conta do Trading Bot ({{ $labels.job }})",
                            "description": "Saldo da conta é {{ $value | printf \"%.2f\" }} USD."
                        }
                    },
                    {
                        "alert": "FeedConnectionDown",
                        "expr": 'connection_status{job="trading_bot", service="feed"} == 0',
                        "for": "2m",
                        "labels": {"severity": "error"},
                        "annotations": {
                            "summary": "Conexão do Feed de Dados CAIU ({{ $labels.job }})",
                            "description": "O bot não está recebendo dados de mercado do feed."
                        }
                    }
                ]
            }]
        }
        try:
            import yaml # Adicionar PyYAML aos requirements se usar
            output_path = Path(rules_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(alert_rules, f, sort_keys=False)
            logger.info(f"Regras de alerta para Prometheus salvas em: {output_path}")
        except ImportError:
            logger.error("Biblioteca PyYAML não instalada. Não foi possível gerar arquivo de regras de alerta.")
        except Exception as e:
            logger.exception(f"Erro ao gerar regras de alerta para Prometheus em {rules_filepath}:")

        return alert_rules # Retornar o dict para outros usos se necessário