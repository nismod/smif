"""Test data interface
"""
import json
import os
from datetime import datetime

from pytest import fixture
from smif.data_layer import DatafileInterface
from smif.data_layer.load import dump


@fixture(scope='function')
def get_project_config():
    """Return sample project configuration
    """
    return {
        'project_name': 'NISMOD v2.0',
        'scenario_sets': [
            {
                'description': 'The annual change in UK population',
                'name': 'population'
            }
        ],
        'narrative_sets': [
            {
                'description': 'Defines the rate and nature of technological change',
                'name': 'technology'
            },
            {
                'description': 'Defines the nature of governance and influence upon decisions',
                'name': 'governance'
            }
        ],
        'region_sets': [
            {
                'description': 'Local authority districts for the UK',
                'filename': 'lad.csv',
                'name': 'lad'
            }
        ],
        'interval_sets': [
            {
                'description': 'The 8760 hours in the year named by hour',
                'filename': 'hourly.csv', 'name': 'hourly'
            },
            {
                'description': 'One annual timestep, used for aggregate yearly data',
                'filename': 'annual.csv', 'name': 'annual'
            }
        ],
        'units': 'user_units.txt',
        'scenario_data':
        [
            {
                'description': 'The High ONS Forecast for UK population out to 2050',
                'filename': 'population_high.csv',
                'name': 'High Population (ONS)',
                'parameters': [
                    {
                        'name': 'population_count',
                        'spatial_resolution': 'lad',
                        'temporal_resolution': 'annual',
                        'units': 'people',
                    }
                ],
                'scenario_set': 'population',
            },
            {
                'description': 'The Low ONS Forecast for UK population out to 2050',
                'filename': 'population_low.csv',
                'name': 'Low Population (ONS)',
                'parameters': [
                    {
                        'name': 'population_count',
                        'spatial_resolution': 'lad',
                        'temporal_resolution': 'annual',
                        'units': 'people',
                    }
                ],
                'scenario_set': 'population',
            }
        ],
        'narrative_data': [
            {
                'description': 'High penetration of SMART technology on the demand side',
                'filename': 'energy_demand_high_tech.yml',
                'name': 'Energy Demand - High Tech',
                'narrative_set': 'technology',
            },
            {
                'description': 'Stronger role for central government in planning and ' +
                               'regulation, less emphasis on market-based solutions',
                'filename': 'central_planning.yml',
                'name': 'Central Planning',
                'narrative_set': 'governance',
            }
        ]
    }


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


def test_datafileinterface_sos_model_run(get_sos_model_run, setup_folder_structure):
    """ Test to write two sos_model_run configurations to Yaml files, then
    read the Yaml files and compare that the result is equal.
    """
    basefolder = setup_folder_structure
    project_config_path = os.path.join(str(basefolder), 'config', 'project.yml')
    dump(get_project_config, project_config_path)
    config_handler = DatafileInterface(str(basefolder))

    sos_model_run1 = get_sos_model_run
    sos_model_run1['name'] = 'sos_model_run1'
    config_handler.write_sos_model_run(sos_model_run1)

    sos_model_run2 = get_sos_model_run
    sos_model_run2['name'] = 'sos_model_run2'
    config_handler.write_sos_model_run(sos_model_run2)

    sos_model_runs = config_handler.read_sos_model_runs()

    assert sos_model_runs.count(sos_model_run1) == 1
    assert sos_model_runs.count(sos_model_run2) == 1


def test_datafileinterface_sos_model(get_sos_model, setup_folder_structure):
    """ Test to write two soS_model configurations to Yaml files, then
    read the Yaml files and compare that the result is equal.
    """
    basefolder = setup_folder_structure
    project_config_path = os.path.join(str(basefolder), 'config', 'project.yml')
    dump(get_project_config, project_config_path)
    config_handler = DatafileInterface(str(basefolder))

    sos_model1 = get_sos_model
    sos_model1['name'] = 'sos_model_1'
    config_handler.write_sos_model(sos_model1)

    sos_model2 = get_sos_model
    sos_model2['name'] = 'sos_model_2'
    config_handler.write_sos_model(sos_model2)

    sos_models = config_handler.read_sos_models()

    assert sos_models.count(sos_model1) == 1
    assert sos_models.count(sos_model2) == 1


def test_datafileinterface_sector_model(setup_folder_structure, get_project_config,
                                        get_sector_model):
    """ Test to write a sector_model configuration to a Yaml file
    read the Yaml file and compare that the result is equal.
    Finally check if the name shows up the the readlist.
    """
    basefolder = setup_folder_structure
    project_config_path = os.path.join(str(basefolder), 'config', 'project.yml')
    dump(get_project_config, project_config_path)
    config_handler = DatafileInterface(str(basefolder))

    sector_model = get_sector_model
    config_handler.write_sector_model(sector_model)
    assert sector_model == config_handler.read_sector_model(sector_model['name'])
    assert sector_model['name'] in config_handler.read_sector_models()


def test_datafileinterface_region_set(setup_folder_structure, get_project_config,
                                      setup_region_data):
    """ Test to dump a GeoJSON file and then read the file
    using the datafile interface. Finally check if the name
    shows up the the returned dictionary.
    """
    basefolder = setup_folder_structure
    project_config_path = os.path.join(str(basefolder), 'config', 'project.yml')
    dump(get_project_config, project_config_path)
    region_data = setup_region_data

    with open(os.path.join(str(basefolder), 'data', 'regions',
                           'test_region.json'), 'w+') as region_file:
        json.dump(region_data, region_file)

    config_handler = DatafileInterface(str(basefolder))
    test_region = config_handler.read_region_set_data('test_region.json')

    assert test_region[0]['properties']['name'] == 'oxford'


def test_datafileinterface_project(setup_folder_structure, get_project_config):
    """ Test to read and write the project configuration
    """
    basefolder = setup_folder_structure
    project_config_path = os.path.join(str(basefolder), 'config', 'project.yml')
    dump(get_project_config, project_config_path)

    config_handler = DatafileInterface(str(basefolder))

    # Region sets / read existing (from fixture)
    region_sets = config_handler.read_region_sets()
    assert region_sets[0]['name'] == 'lad'
    assert len(region_sets) == 1

    # Region sets / add
    region_set = {
        'name': 'lad_NL',
        'description': 'Local authority districts for the Netherlands',
        'filename': 'lad_NL.csv'
    }
    config_handler.write_region_set(region_set)
    region_sets = config_handler.read_region_sets()
    assert len(region_sets) == 2

    # Region sets / modify
    region_set = {
        'name': 'lad_NL',
        'description': 'Local authority districts for the Netherlands',
        'filename': 'lad_NL_V2.csv'
    }
    config_handler.write_region_set(region_set)
    region_sets = config_handler.read_region_sets()
    assert len(region_sets) == 2
    for region_set in region_sets:
        if region_set['name'] == 'lad_NL':
            assert region_set['filename'] == 'lad_NL_V2.csv'

    # Interval sets / read existing (from fixture)
    interval_sets = config_handler.read_interval_sets()
    assert interval_sets[0]['name'] == 'hourly'
    assert len(interval_sets) == 2

    # Interval sets / add
    interval_set = {
        'name': 'monthly',
        'description': 'The 12 months of the year',
        'filename': 'monthly.csv'
    }
    config_handler.write_interval_set(interval_set)
    interval_sets = config_handler.read_interval_sets()
    assert len(interval_sets) == 3

    # Interval sets / modify
    interval_set = {
        'name': 'monthly',
        'description': 'The 12 months of the year',
        'filename': 'monthly_V2.csv'
    }
    config_handler.write_interval_set(interval_set)
    interval_sets = config_handler.read_interval_sets()
    assert len(interval_sets) == 3
    for interval_set in interval_sets:
        if interval_set['name'] == 'monthly':
            assert interval_set['filename'] == 'monthly_V2.csv'

    # Scenario data
    scenario_sets = config_handler.read_scenario_sets()
    assert scenario_sets[0]['name'] == 'population'
    assert len(scenario_sets) == 1

    scenario_set = {
        'name': 'temperature',
        'description': 'The annual change in temperature',
    }

    config_handler.write_interval_set(scenario_set)
    scenario_sets = config_handler.read_scenario_sets()
    assert len(scenario_sets) == 2
