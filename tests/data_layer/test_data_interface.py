"""Test data interface
"""
from datetime import datetime

from pytest import fixture
from smif.data_layer import DatafileInterface


@fixture(scope='function')
def get_sos_model_run():
    """Return sample sos_model_run
    """
    return {
        'name': 'unique_model_run_name',
        'description': 'a description of what the model run contains',
        'stamp': datetime(2017, 9, 20, 12, 53, 23),
        'timesteps': [
            2015,
            2020,
            2025
        ],
        'sos_model': 'energy',
        'decision_module': 'energy_moea.py',
        'scenarios': [
            {
                'population': 'High Population (ONS)'
            }
        ],
        'narratives': [
            {
                'technology': [
                    'Energy Demand - High Tech'
                ]
            },
            {
                'governance': 'Central Planning'
            }
        ]
    }


@fixture(scope='function')
def get_sos_model():
    """Return sample sos_model
    """
    return {
        'name': 'energy',
        'description': "A system of systems model which encapsulates "
                       "the future supply and demand of energy for the UK",
        'scenario_sets': [
            'population'
            ],
        'sector_models': [
            'energy_demand',
            'energy_supply'
            ],
        'dependencies': [
            {
                'source_model': 'population',
                'source_model_output': 'count',
                'sink_model': 'energy_demand',
                'sink_model_input': 'population'
            },
            {
                'source_model': 'energy_demand',
                'source_model_output': 'gas_demand',
                'sink_model': 'energy_supply',
                'sink_model_input': 'natural_gas_demand'
            }
        ]
    }


@fixture(scope='function')
def get_sector_model():
    """Return sample sector_model
    """
    return {
        'name': 'energy_demand',
        'description': "Computes the energy demand of the"
                       "UK population for each timestep",
        'classname': 'EnergyDemandWrapper',
        'path': '../../models/energy_demand/run.py',
        'inputs': [
            {
                'name': 'population',
                'spatial_resolution': 'lad',
                'temporal_resolution': 'annual',
                'units': 'people'
            }
        ],
        'outputs': [
            {
                'name': 'gas_demand',
                'spatial_resolution': 'lad',
                'temporal_resolution': 'hourly',
                'units': 'GWh'
            }
        ],
        'parameters': [
            {
                'absolute_range': '(0.5, 2)',
                'default_value': 1,
                'description': "Difference in floor area per person"
                               "in end year compared to base year",
                'name': 'assump_diff_floorarea_pp',
                'suggested_range': '(0.5, 2)',
                'units': 'percentage'
            }
        ],
        'interventions': ['./energy_demand.yml'],
        'initial_conditions': ['./energy_demand_init.yml']
    }


def test_yaml_sos_model_run(get_sos_model_run, setup_folder_structure):
    """ Test to write two sos_model_run configurations to Yaml files, then
    read the Yaml files and compare that the result is equal.
    """
    basefolder = setup_folder_structure
    config_handler = DatafileInterface(str(basefolder.join('config')))

    sos_model_run1 = get_sos_model_run
    sos_model_run1['name'] = 'sos_model_run1'
    config_handler.write_sos_model_run(sos_model_run1)

    sos_model_run2 = get_sos_model_run
    sos_model_run2['name'] = 'sos_model_run2'
    config_handler.write_sos_model_run(sos_model_run2)

    sos_model_runs = config_handler.read_sos_model_runs()

    assert sos_model_runs.count(sos_model_run1) == 1
    assert sos_model_runs.count(sos_model_run2) == 1


def test_yaml_sos_model(get_sos_model, setup_folder_structure):
    """ Test to write two soS_model configurations to Yaml files, then
    read the Yaml files and compare that the result is equal.
    """
    basefolder = setup_folder_structure
    config_handler = DatafileInterface(str(basefolder.join('config')))

    sos_model1 = get_sos_model
    sos_model1['name'] = 'sos_model_1'
    config_handler.write_sos_model(sos_model1)

    sos_model2 = get_sos_model
    sos_model2['name'] = 'sos_model_2'
    config_handler.write_sos_model(sos_model2)

    sos_models = config_handler.read_sos_models()

    assert sos_models.count(sos_model1) == 1
    assert sos_models.count(sos_model2) == 1


def test_yaml_sector_model(get_sector_model, setup_folder_structure):
    """ Test to write a sector_model configuration to a Yaml file
    read the Yaml file and compare that the result is equal.
    Finally check if the name shows up the the readlist.
    """
    sector_model = get_sector_model
    basefolder = setup_folder_structure

    config_handler = DatafileInterface(str(basefolder.join('config')))
    config_handler.write_sector_model(sector_model)
    assert sector_model == config_handler.read_sector_model(sector_model['name'])
    assert sector_model['name'] in config_handler.read_sector_models()
