import pytest

@pytest.mark.asyncio
async def test_risk_manager_initialization(monkeypatch):
    monkeypatch.setenv("TT_TT_ACCOUNT_ID", "dummy")
    monkeypatch.setenv("TT_TT_PASSWORD", "dummy")

    import sys, types
    dummy = types.ModuleType("dummy")
    monkeypatch.setitem(sys.modules, "pandas", dummy)
    monkeypatch.setitem(sys.modules, "talib", dummy)
    dummy_colorlog = types.ModuleType("colorlog")
    class _CF:
        def __init__(self, *args, **kwargs):
            pass
    dummy_colorlog.ColoredFormatter = _CF
    monkeypatch.setitem(sys.modules, "colorlog", dummy_colorlog)
    dummy_ntplib = types.ModuleType("ntplib")
    class _Client:
        def __init__(self, *args, **kwargs):
            pass
    class _Stats:
        pass
    class _Exc(Exception):
        pass
    dummy_ntplib.NTPClient = _Client
    dummy_ntplib.NTPStats = _Stats
    dummy_ntplib.NTPException = _Exc
    monkeypatch.setitem(sys.modules, "ntplib", dummy_ntplib)
    monkeypatch.setitem(sys.modules, "pytz", dummy)

    from risk import RiskManager

    rm = RiskManager()
    assert rm.daily_start_balance_rm == 0.0
    assert rm.high_water_mark_session == 0.0
    assert rm.circuit_breaker.state.value == "closed"

    await rm.initialize(10000.0, account_currency="USD")

    assert rm.daily_start_balance_rm == 10000.0
    assert rm.high_water_mark_session == 10000.0
    assert rm.circuit_breaker.state.value == "closed"
