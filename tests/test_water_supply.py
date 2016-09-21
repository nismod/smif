import pytest
from tests.fixtures.water_supply import WaterSupplyPython, WaterSupplyExecutable

def test_simulate_rain_python():
    ws = WaterSupplyPython()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful == True
    results = ws.results
    assert results["water"] == 1


def test_simulate_rain_cost_python():
    ws = WaterSupplyPython()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful == True
    results = ws.results
    assert results["cost"] == 1

def test_simulate_rain_executable():
    ws = WaterSupplyExecutable()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful == True
    results = ws.results
    assert results['water'] == 1


def test_simulate_rain_cost_executable():
    ws = WaterSupplyExecutable()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful == True
    results = ws.results
    assert results['cost'] == 1
