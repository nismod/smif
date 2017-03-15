"""Test config validation
"""
from smif.data_layer.validate import (VALIDATION_ERRORS,
                                      validate_interventions,
                                      validate_planning_config,
                                      validate_sector_model_initial_config,
                                      validate_sos_model_config,
                                      validate_timesteps)


def get_sos_model_config():
    """Return sample sos_model config
    """
    return {
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


def get_sector_model_initial_config():
    """Return sample sector_model initial config
    """
    return {
        'name': 'test_model',
        'config_dir': '/path/to/config',
        'path': '/path/to/model/run.py',
        'classname': 'SectorModelImplementation'
    }


def get_intervention():
    """Return sample intervention
    """
    return {
        'name': 'asset',
        'capacity': {
            'value': 3,
            'units': 'MW'
        },
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


def test_modelrun_config_validate():
    """Expect no errors from a complete, valid config
    """
    data = get_sos_model_config()
    validate_sos_model_config(data)


def test_modelrun_config_invalid():
    """Expect an error if not a dict
    """
    invalid_possibilities = [
        0,
        [],
        "just a string",
        3.1415
    ]

    for invalid_data in invalid_possibilities:
        validate_sos_model_config(invalid_data)
        ex = VALIDATION_ERRORS.pop()
        msg = "Main config file should contain setup data"
        assert msg in str(ex)


def test_missing_timestep():
    """Expect an error if missing timesteps
    """
    data = get_sos_model_config()
    del data['timesteps']

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "No 'timesteps' file specified in main config"
    assert msg in str(ex)


def test_invalid_timesteps_file():
    """Expect an error if timesteps is not a path to a file
    """
    data = get_sos_model_config()
    data['timesteps'] = 3

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "Expected 'timesteps' in main config to specify a timesteps file, instead got 3"
    assert msg in str(ex)


def test_invalid_timesteps():
    """Expect a list of timesteps, else error
    """
    data = 2010
    validate_timesteps(data, "timestep.yaml")
    ex = VALIDATION_ERRORS.pop()
    msg = "expected a list of timesteps"
    assert msg in str(ex)


def test_invalid_single_timestep():
    """Expect an error for non-integer timesteps
    """
    data = [2010, "January 2015"]
    validate_timesteps(data, "timestep.yaml")
    ex = VALIDATION_ERRORS.pop()
    msg = "timesteps should be integer years"
    assert msg in str(ex)


def test_missing_sector_models():
    """Expect an error if missing sector_models
    """
    data = get_sos_model_config()
    del data['sector_models']

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "No 'sector_models' specified in main config"
    assert msg in str(ex)


def test_sector_models_not_list():
    """Expect an error if sector_models is not a list
    """
    data = get_sos_model_config()
    data['sector_models'] = 42

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "Expected 'sector_models' in main config to specify a list of sector " + \
          "models to run, instead got 42."
    assert msg in str(ex)


def test_sector_models_empty_list():
    """Expect an error if sector_models is an empty list
    """
    data = get_sos_model_config()
    data['sector_models'] = []

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "No 'sector_models' specified in main config"
    assert msg in str(ex)


def test_sector_model_missing_required():
    """Expect an error if a sector_model is missing a required key
    """
    required_keys = ['name', 'config_dir', 'path', 'classname']
    for key in required_keys:
        data = get_sector_model_initial_config()
        del data[key]

        validate_sector_model_initial_config(data)
        ex = VALIDATION_ERRORS.pop()
        msg = "Expected a value for '{}'".format(key)
        assert msg in str(ex)


def test_missing_planning():
    """Expect an error if missing planning
    """
    data = get_sos_model_config()
    del data['planning']

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "No 'planning' mode specified in main config"
    assert msg in str(ex)


def test_used_planning_needs_files():
    """Expect an error if a planning mode is to be used, but has no files
    """
    data = get_sos_model_config()
    del data["planning"]["pre_specified"]["files"]

    validate_sos_model_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "No 'files' provided for the 'pre_specified' planning type in main config"
    assert msg in str(ex)


def test_planning_missing_required():
    """Expect an error if missing a planning mode
    """
    required_keys = ["pre_specified", "rule_based", "optimisation"]
    for key in required_keys:
        data = get_sos_model_config()["planning"]
        del data[key]

        validate_planning_config(data)
        ex = VALIDATION_ERRORS.pop()
        msg = "No '{}' settings specified under 'planning'".format(key)
        assert msg in str(ex)


def test_planning_missing_use():
    """Expect an error if a planning mode is missing "use""
    """
    data = get_sos_model_config()["planning"]
    del data["rule_based"]["use"]

    validate_planning_config(data)
    ex = VALIDATION_ERRORS.pop()
    msg = "No 'use' settings specified for 'rule_based' 'planning'"
    assert msg in str(ex)


def test_interventions_missing_required():
    """Expect an error if an intervention is missing required key
    """
    required_keys = ["name", "location", "capital_cost", "operational_lifetime",
                     "economic_lifetime"]

    for key in required_keys:
        intervention = get_intervention()
        del intervention[key]

        data = [intervention]

        msg = "required a value for '{}' in each intervention".format(key)

        validate_interventions(data, "/path/to/data.yaml")
        ex = VALIDATION_ERRORS.pop()
        assert msg in str(ex)


def test_interventions_checks_for_units():
    """Expect an error if an intervention's "capacity" has no units
    """
    intervention = get_intervention()
    intervention["capacity"] = 3

    data = [intervention]

    msg = "Loading interventions from /path/to/data.yaml, asset.capacity " + \
          "was 3 but should have specified units, e.g. " + \
          "{'value': 3, 'units': 'm'}"

    validate_interventions(data, "/path/to/data.yaml")
    ex = VALIDATION_ERRORS.pop()
    assert msg in str(ex)
