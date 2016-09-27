"""Tests to ensure that the example simulation model fixtures are behaving


"""
from pytest import raises
from tests.fixtures.water_supply import (WaterSupplyExecutable,
                                         WaterSupplyPython, raininess_oracle)


def test_raininess_oracle():
    time = [2010, 2020, 2030, 2042, 2050]
    expected = [1, 2, 3, 4, 5]

    for result in zip(time, expected):
        actual = raininess_oracle(result[0])
        assert actual == result[1]


def test_raininess_oracle_out_of_range():
    msg = "timestep 2051 is outside of the range [2010, 2050]"
    with raises(AssertionError, message=msg):
        raininess_oracle(2051)


def test_simulate_rain_python():
    ws = WaterSupplyPython()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results["water"] == 1


def test_simulate_rain_cost_python():
    ws = WaterSupplyPython()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results["cost"] == 1


def test_simulate_rain_executable():
    ws = WaterSupplyExecutable()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results['water'] == 1


def test_simulate_rain_cost_executable():
    ws = WaterSupplyExecutable()
    ws.initialise({
        "raininess": 1
    })
    ws.simulate()
    assert ws.run_successful
    results = ws.results
    assert results['cost'] == 1
