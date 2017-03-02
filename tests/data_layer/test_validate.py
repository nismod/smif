"""Test config validation
"""
from pytest import raises
from smif.data_layer.validate import (ValidationError, validate_interventions,
                                      validate_sos_model_config)


def test_modelrun_config_validate():
    data = {
        "sector_models": [
            {
                "name": "water_supply",
                "path": "../models/water_supply.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                "initial_conditions": [
                    "../data/water_supply/initial_conditions/initial_2015_oxford.yaml"
                ],
                "interventions": [
                    "../data/water_supply/interventions/types.yaml"
                ]
            }
        ],
        "timesteps": "timesteps.yaml",
        "scenario_data": [
            {
                "parameter": "population",
                "file": "../data/scenario/population.yaml"
            },
            {
                "parameter": "raininess",
                "file": "../data/scenario/raininess.yaml"
            }
        ],
        "planning": {
            "pre_specified": {
                "use": True,
                "files": [
                    "../planning/expected_to_2020.yaml",
                    "../planning/national_infrastructure_pipeline.yaml"
                ]
            },
            "rule_based": {
                "use": False
            },
            "optimisation": {
                "use": False
            }
        }
    }
    validate_sos_model_config(data)


def test_missing_timestep():
    data = {
        "sector_models": [
            {
                "name": "water_supply",
                "path": "../models/water_supply.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                "initial_conditions": [
                    "../data/water_supply/initial_conditions/initial_2015_oxford.yaml"
                ],
                "interventions": [
                    "../data/water_supply/interventions/types.yaml"
                ]
            }
        ],
        "scenario_data": [
            {
                "parameter": "population",
                "file": "../data/scenario/population.yaml"
            },
            {
                "parameter": "raininess",
                "file": "../data/scenario/raininess.yaml"
            }
        ],
        "planning": {
            "pre_specified": {
                "use": True,
                "files": [
                    "../planning/expected_to_2020.yaml",
                    "../planning/national_infrastructure_pipeline.yaml"
                ]
            },
            "rule_based": {
                "use": False
            },
            "optimisation": {
                "use": False
            }
        }
    }

    msg = "No 'timesteps' file specified in main config"
    with raises(ValidationError) as ex:
        validate_sos_model_config(data)
    assert msg in str(ex.value)


def test_used_planning_needs_files():
    data = {
        "sector_models": [
            {
                "name": "water_supply",
                "path": "../models/water_supply.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                "initial_conditions": [
                    "../data/water_supply/initial_conditions/initial_2015_oxford.yaml"
                ],
                "interventions": [
                    "../data/water_supply/interventions/types.yaml"
                ]
            }
        ],
        "timesteps": "timesteps.yaml",
        "scenario_data": [
            {
                "parameter": "population",
                "file": "../data/scenario/population.yaml"
            },
            {
                "parameter": "raininess",
                "file": "../data/scenario/raininess.yaml"
            }
        ],
        "planning": {
            "pre_specified": {
                "use": True
            },
            "rule_based": {
                "use": False
            },
            "optimisation": {
                "use": False
            }
        }
    }

    msg = "No 'files' provided for the 'pre_specified' planning type in main config"
    with raises(ValidationError) as ex:
        validate_sos_model_config(data)
    assert msg in str(ex.value)


def test_interventions_checks_for_units():
    data = [
        {
            'name': 'asset',
            'capacity': 3,
            'location': 'galloway',
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £/km"
            }
        }
    ]

    msg = "Loading interventions from /path/to/data.yaml, asset.capacity " + \
          "was 3 but should have specified units, e.g. " + \
          "{'value': 3, 'units': 'm'}"

    with raises(ValidationError) as ex:
        validate_interventions(data, "/path/to/data.yaml")

    assert msg in str(ex.value)
