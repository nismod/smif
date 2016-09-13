import pytest
import smif

def test_simulate_rain():
    ws = smif.WaterSupply()
    ws.initialise({
        "raininess": 1
    })
    output = ws.simulate()
    assert output["water"] == 1


def test_simulate_rain_cost():
    ws = smif.WaterSupply()
    ws.initialise({
        "raininess": 1
    })
    output = ws.simulate()
    assert output["cost"] == 1

