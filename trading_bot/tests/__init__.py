# tests/__init__.py
"""Módulo de testes"""
import pytest

# Configurações de teste
# Esta linha é específica do pytest e geralmente é usada para
# habilitar reescrita de asserções em módulos de fixtures para melhor output.
# Requer que 'tests.fixtures' seja um módulo Python válido (ou seja, tests/fixtures.py ou tests/fixtures/__init__.py).
pytest.register_assert_rewrite('tests.fixtures') #

# ===================================