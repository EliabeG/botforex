import pytest


def _patch_deps(monkeypatch):
    import sys, types
    dummy = types.ModuleType("dummy")
    monkeypatch.setenv("TT_TT_ACCOUNT_ID", "dummy")
    monkeypatch.setenv("TT_TT_PASSWORD", "dummy")
    monkeypatch.setitem(sys.modules, "pandas", dummy)
    monkeypatch.setitem(sys.modules, "talib", dummy)
    dummy_colorlog = types.ModuleType("colorlog")
    class _CF:
        def __init__(self, *a, **k):
            pass
    dummy_colorlog.ColoredFormatter = _CF
    monkeypatch.setitem(sys.modules, "colorlog", dummy_colorlog)
    dummy_ntplib = types.ModuleType("ntplib")
    class _Client:
        pass
    dummy_ntplib.NTPClient = _Client
    dummy_ntplib.NTPStats = type("S", (), {})
    dummy_ntplib.NTPException = Exception
    monkeypatch.setitem(sys.modules, "ntplib", dummy_ntplib)
    monkeypatch.setitem(sys.modules, "pytz", dummy)


@pytest.mark.usefixtures("monkeypatch")
def test_circuit_breaker_initial_state(monkeypatch):
    _patch_deps(monkeypatch)
    from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
    cb = CircuitBreaker()
    assert cb.state == CircuitBreakerState.CLOSED
