from types import SimpleNamespace
import pytest
from workflow import registry


def test_current_algorithms_are_flat_and_discovered():
    assert set(registry.list_algorithms())=={'ppo','ga2e','spo','rvl','dae','vmpo'}
    for name in registry.list_algorithms():
        mod=registry.import_module('onpolicy.'+name)
        assert mod.__file__.endswith('/'+name+'.py')
        assert registry.get_algo(name).config_cls is mod.CONFIG_CLASS


def test_external_algorithm_without_registry_edit(monkeypatch):
    entry=registry.AlgoEntry(lambda *a:None,dict)
    ep=SimpleNamespace(name='research_algo',load=lambda:lambda:entry)
    monkeypatch.setattr(registry.metadata,'entry_points',lambda **kw:[ep])
    assert 'research_algo' in registry.list_algorithms()
    assert registry.get_algo('research_algo') is entry


def test_duplicate_plugin_fails(monkeypatch):
    monkeypatch.setattr(registry.metadata,'entry_points',lambda **kw:[SimpleNamespace(name='ppo')])
    with pytest.raises(ValueError,match='duplicate'):registry.list_algorithms()
