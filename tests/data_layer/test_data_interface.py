"""Test data interface
"""
from pytest import fixture
from smif.data_layer import YamlInterface
from datetime import datetime

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
        'description': 'A system of systems model which encapsulates the future supply and demand of energy for the UK',
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
        'description': 'Computes the energy demand of the UK population for each timestep',
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
                'description': 'Difference in floor area per person in end year compared to base year',
                'name': 'assump_diff_floorarea_pp',
                'suggested_range': '(0.5, 2)',
                'units': 'percentage'
            }
        ],
        'interventions': ['./energy_demand.yml'],
        'initial_conditions': ['./energy_demand_init.yml']
    }

def test_yaml_write_sos_model_run(get_sos_model_run):
    """ Test to write a sos_model_run configuration to a Yaml file
    """
    sos_model_run = get_sos_model_run
    
    assert sos_model_run['name'] != ''
    assert sos_model_run['description'] != ''
    assert sos_model_run['stamp'] != ''
    assert len(sos_model_run['timesteps']) > 0
    assert sos_model_run['sos_model'] != ''
    assert sos_model_run['decision_module'] != ''
    assert len(sos_model_run['scenarios']) > 0
    assert len(sos_model_run['narratives']) > 0
    
    config_file_handle = YamlInterface('../config')
    config_file_handle.write_sos_model_run(sos_model_run)

def test_yaml_write_sos_model(get_sos_model):
    """ Test to write a sos_model configuration to a Yaml file
    """
    sos_model = get_sos_model
        
    assert sos_model['name'] != ''
    assert sos_model['description'] != ''
    assert len(sos_model['scenario_sets']) > 0
    assert len(sos_model['sector_models']) > 0
    assert len(sos_model['dependencies']) > 0
    
    config_file_handle = YamlInterface('../config')
    config_file_handle.write_sos_model(sos_model)

def test_yaml_write_sector_model(get_sector_model):
    """ Test to write a sector_model configuration to a Yaml file
    """
    sector_model = get_sector_model
        
    assert sector_model['name'] != ''
    assert sector_model['description'] != ''
    assert sector_model['classname'] != ''
    assert sector_model['path'] != ''
    assert len(sector_model['inputs']) > 0
    assert len(sector_model['outputs']) > 0
    assert len(sector_model['parameters']) > 0
    assert len(sector_model['interventions']) > 0
    assert len(sector_model['initial_conditions']) > 0
    
    config_file_handle = YamlInterface('../config')
    config_file_handle.write_sector_model(sector_model)