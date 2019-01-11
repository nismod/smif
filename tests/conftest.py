#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Holds fixtures for the smif package tests
"""
from __future__ import absolute_import, division, print_function

import json
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from pytest import fixture
from smif.data_layer import Store
from smif.data_layer.data_array import DataArray
from smif.data_layer.file.file_config_store import _write_yaml_file as dump
from smif.data_layer.memory_interface import (MemoryConfigStore,
                                              MemoryDataStore,
                                              MemoryMetadataStore)
from smif.metadata import Spec

logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                    filemode='w')


@fixture
def empty_store():
    """Store fixture
    """
    # implement each part using the memory classes, simpler than mocking
    # each other implementation of a part is tested fully by e.g. test_config_store.py
    return Store(
        config_store=MemoryConfigStore(),
        metadata_store=MemoryMetadataStore(),
        data_store=MemoryDataStore()
    )


@fixture
def setup_empty_folder_structure(tmpdir_factory):

    folder_list = ['models', 'results', 'config', 'data']

    config_folders = [
        'dimensions',
        'model_runs',
        'scenarios',
        'sector_models',
        'sos_models',
    ]
    for folder in config_folders:
        folder_list.append(os.path.join('config', folder))

    data_folders = [
        'coefficients',
        'dimensions',
        'initial_conditions',
        'interventions',
        'narratives',
        'scenarios',
        'strategies',
        'parameters'
    ]
    for folder in data_folders:
        folder_list.append(os.path.join('data', folder))

    test_folder = tmpdir_factory.mktemp("smif")

    for folder in folder_list:
        test_folder.mkdir(folder)

    return test_folder


@fixture
def setup_folder_structure(setup_empty_folder_structure, oxford_region, remap_months,
                           initial_system, planned_interventions):
    """

    Returns
    -------
    :class:`LocalPath`
        Path to the temporary folder
    """
    test_folder = setup_empty_folder_structure

    region_file = test_folder.join('data', 'dimensions', 'test_region.geojson')
    region_file.write(json.dumps(oxford_region))

    intervals_file = test_folder.join('data', 'dimensions', 'annual.yml')
    intervals_file.write("""\
- name: '1'
  interval: [[P0Y, P1Y]]
""")

    intervals_file = test_folder.join('data', 'dimensions', 'hourly.yml')
    intervals_file.write("""\
- name: '1'
  interval: [[PT0H, PT1H]]
""")

    initial_conditions_dir = str(test_folder.join('data', 'initial_conditions'))
    dump(initial_conditions_dir, 'init_system', initial_system)

    interventions_dir = str(test_folder.join('data', 'interventions'))
    dump(interventions_dir, 'planned_interventions', planned_interventions)

    dimensions_dir = str(test_folder.join('data', 'dimensions'))
    dump(dimensions_dir, 'remap', remap_months)

    units_file = test_folder.join('data', 'user_units.txt')
    with units_file.open(mode='w') as units_fh:
        units_fh.write("blobbiness = m^3 * 10^6\n")
        units_fh.write("people = [people]\n")
        units_fh.write("mcm = 10^6 * m^3\n")
        units_fh.write("GBP=[currency]\n")

    return test_folder


@fixture
def initial_system():
    """Initial system (interventions with build_date)
    """
    return [
        {'name': 'water_asset_a', 'build_year': 2017},
        {'name': 'water_asset_b', 'build_year': 2017},
        {'name': 'water_asset_c', 'build_year': 2017},
    ]


@fixture
def parameters():
    return [
        {
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'unit': '%'
        }
    ]


@fixture
def planned_interventions():
    """Return pre-specified planning intervention data
    """
    return [
        {
            'name': 'water_asset_a',
            'capacity': {'value': 6, 'unit': 'Ml'},
            'description': 'Existing water treatment plants',
            'location': {'lat': 51.74556, 'lon': -1.240528}
        },
        {
            'name': 'water_asset_b',
            'capacity':  {'value': 6, 'unit': 'Ml'},
            'description': 'Existing water treatment plants',
            'location': {'lat': 51.74556, 'lon': -1.240528}
        },
        {
            'name': 'water_asset_c',
            'capacity': {'value': 6, 'unit': 'Ml'},
            'description': 'Existing water treatment plants',
            'location': {'lat': 51.74556, 'lon': -1.240528}
        },
    ]


@fixture
def oxford_region():
    data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::27700"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "oxford"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [448180, 209366],
                            [449500, 211092],
                            [450537, 211029],
                            [450873, 210673],
                            [451250, 210793],
                            [451642, 210023],
                            [453855, 208466],
                            [454585, 208468],
                            [456077, 207967],
                            [456146, 207738],
                            [456668, 207779],
                            [456708, 207444],
                            [456278, 207122],
                            [456149, 206615],
                            [455707, 206798],
                            [455749, 204521],
                            [456773, 204488],
                            [457014, 204184],
                            [456031, 203475],
                            [456444, 202854],
                            [456087, 202044],
                            [455369, 201799],
                            [454396, 202203],
                            [453843, 201634],
                            [452499, 203209],
                            [452052, 203566],
                            [451653, 203513],
                            [450645, 205137],
                            [449497, 205548],
                            [449051, 206042],
                            [448141, 208446],
                            [448180, 209366]
                        ]
                    ]
                }
            },
        ]
    }
    return data


@fixture
def initial_conditions():
    return [{'name': 'solar_installation', 'build_year': 2017}]


@fixture
def interventions():
    return {
        'solar_installation': {
            'name': 'solar_installation',
            'capacity': 5,
            'capacity_units': 'MW'
        },
        'wind_installation': {
            'name': 'wind_installation',
            'capacity': 4,
            'capacity_units': 'MW'
        }
    }


@fixture
def water_interventions_abc():
    return [
        {
            "name": "water_asset_a",
            "location": "oxford",
            "capital_cost": {
                "units": "£",
                "value": 1000
            },
            "economic_lifetime": {
                "units": "years",
                "value": 25
            },
            "operational_lifetime": {
                "units": "years",
                "value": 25
            }
        },
        {
            "name": "water_asset_b",
            "location": "oxford",
            "capital_cost": {
                "units": "£",
                "value": 1500
            },
            "economic_lifetime": {
                "units": "years",
                "value": 25
            },
            "operational_lifetime": {
                "units": "years",
                "value": 25
            }
        },
        {
            "name": "water_asset_c",
            "location": "oxford",
            "capital_cost": {
                "units": "£",
                "value": 3000
            },
            "economic_lifetime": {
                "units": "years",
                "value": 25
            },
            "operational_lifetime": {
                "units": "years",
                "value": 25
            }
        }
    ]


@fixture
def model_run():
    """Return sample model_run
    """
    return {
        'name': 'unique_model_run_name',
        'description': 'a description of what the model run contains',
        'stamp': '2017-09-20T12:53:23+00:00',
        'timesteps': [
            2015,
            2020,
            2025
        ],
        'sos_model': 'energy',
        'scenarios': {
            'population': 'High Population (ONS)'
        },
        'strategies': [
            {
                'type': 'pre-specified-planning',
                'name': 'energy_supply',
                'description': 'description of the strategy',
                'model_name': 'energy_supply',
            }
        ],
        'narratives': {
            'technology': [
                'Energy Demand - High Tech'
            ],
            'governance': [
                'Central Planning'
            ]
        }
    }


@fixture
def get_sos_model(sample_narratives):
    """Return sample sos_model
    """
    return {
        'name': 'energy',
        'description': "A system of systems model which encapsulates "
                       "the future supply and demand of energy for the UK",
        'scenarios': [
            'population'
        ],
        'narratives': sample_narratives,
        'sector_models': [
            'energy_demand',
            'energy_supply'
        ],
        'scenario_dependencies': [
            {
                'source': 'population',
                'source_output': 'population_count',
                'sink': 'energy_demand',
                'sink_input': 'population'
            }
        ],
        'model_dependencies': [
            {
                'source': 'energy_demand',
                'source_output': 'gas_demand',
                'sink': 'energy_supply',
                'sink_input': 'natural_gas_demand'
            }
        ]
    }


@fixture
def get_sector_model(annual, hourly):
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
                'dtype': 'int',
                'dims': ['lad', 'annual'],
                'coords': {
                    'lad': ['a', 'b'],
                    'annual': annual
                },
                'absolute_range': [0, int(1e12)],
                'expected_range': [0, 100000],
                'unit': 'people'
            }
        ],
        'outputs': [
            {
                'name': 'gas_demand',
                'dtype': 'float',
                'dims': ['lad', 'hourly'],
                'coords': {
                    'lad': ['a', 'b'],
                    'hourly': hourly
                },
                'absolute_range': [0, float('inf')],
                'expected_range': [0.01, 10],
                'unit': 'GWh'
            }
        ],
        'parameters': [
            {
                'name': 'smart_meter_savings',
                'description': "Difference in floor area per person"
                               "in end year compared to base year",
                'absolute_range': [0, float('inf')],
                'expected_range': [0.5, 2],
                'unit': '%',
                'dtype': 'float'
            },
            {
                'name': 'homogeneity_coefficient',
                'description': "How homegenous the centralisation"
                               "process is",
                'absolute_range': [0, 1],
                'expected_range': [0, 1],
                'unit': 'percentage',
                'dtype': 'float'
            }
        ],
        'interventions': [],
        'initial_conditions': []
    }


@fixture
def energy_supply_sector_model(hourly):
    """Return sample sector_model
    """
    return {
        'name': 'energy_supply',
        'description': "Supply system model",
        'classname': 'EnergySupplyWrapper',
        'path': '../../models/energy_supply/run.py',
        'inputs': [
            {
                'name': 'natural_gas_demand',
                'dims': ['lad', 'hourly'],
                'coords': {
                    'lad': ['a', 'b'],
                    'hourly': hourly
                },
                'absolute_range': [0, float('inf')],
                'expected_range': [0, 100],
                'dtype': 'float',
                'unit': 'GWh'
            }
        ],
        'outputs': [],
        'parameters': [],
        'interventions': [],
        'initial_conditions': []
    }


@fixture
def water_supply_sector_model(hourly):
    """Return sample sector_model
    """
    return {
        'name': 'water_supply',
        'description': "Supply system model",
        'classname': 'WaterSupplyWrapper',
        'path': '../../models/water_supply/run.py',
        'inputs': [],
        'outputs': [],
        'parameters': [
            {
                'name': 'clever_water_meter_savings',
                'description': "",
                'absolute_range': [0, 1],
                'expected_range': [0, 0.2],
                'unit': 'percentage',
                'dtype': 'float'
            },
            {
                'name': 'per_capita_water_demand',
                'description': "",
                'absolute_range': [0, float('inf')],
                'expected_range': [0, 0.05],
                'unit': 'Ml/day',
                'dtype': 'float'
            }
        ],
        'interventions': [],
        'initial_conditions': []
    }


@fixture
def get_sector_model_parameter_defaults(get_sector_model):
    """DataArray for each parameter default
    """
    data = {
        'smart_meter_savings': np.array(0.5),
        'homogeneity_coefficient': np.array(0.1)
    }
    for param in get_sector_model['parameters']:
        nda = data[param['name']]
        spec = Spec.from_dict(param)
        data[param['name']] = DataArray(spec, nda)
    return data


@fixture
def get_multidimensional_param():
    spec = Spec.from_dict({
        'name': 'ss_t_base_heating',
        'description': 'Industrial base temperature',
        'default': '../energy_demand/parameters/ss_t_base_heating.csv',
        'unit': '',
        'dims': ['interpolation_params', 'end_yr'],
        'coords': {
            'interpolation_params': ['diffusion_choice', 'value_ey'],
            'end_yr': [2030, 2050]
        },
        'dtype': 'float'
    })
    dataframe = pd.DataFrame([
        {
            'interpolation_params': 'diffusion_choice',
            'end_yr': 2030,
            'ss_t_base_heating': 0
        },
        {
            'interpolation_params': 'diffusion_choice',
            'end_yr': 2050,
            'ss_t_base_heating': 0
        },
        {
            'interpolation_params': 'value_ey',
            'end_yr': 2030,
            'ss_t_base_heating': 15.5
        },
        {
            'interpolation_params': 'value_ey',
            'end_yr': 2050,
            'ss_t_base_heating': 15.5
        },
    ]).set_index(['interpolation_params', 'end_yr'])
    return DataArray.from_df(spec, dataframe)


@fixture
def get_sector_model_no_coords(get_sector_model):
    model = deepcopy(get_sector_model)
    for spec_group in ('inputs', 'outputs', 'parameters'):
        for spec in model[spec_group]:
            try:
                del spec['coords']
            except KeyError:
                pass
    return model


@fixture
def sample_scenarios():
    """Return sample scenario
    """
    return [
        {
            'name': 'population',
            'description': 'The annual change in UK population',
            'provides': [
                {
                    'name': "population_count",
                    'description': "The count of population",
                    'unit': 'people',
                    'dtype': 'int',
                    'dims': ['lad', 'annual']
                },
            ],
            'variants': [
                {
                    'name': 'High Population (ONS)',
                    'description': 'The High ONS Forecast for UK population out to 2050',
                    'data': {
                        'population_count': 'population_high.csv'
                    }
                },
                {
                    'name': 'Low Population (ONS)',
                    'description': 'The Low ONS Forecast for UK population out to 2050',
                    'data': {
                        'population_count': 'population_low.csv'
                    }
                },
            ],
        },
    ]


@fixture
def sample_scenario_data(scenario, get_sector_model, energy_supply_sector_model,
                         water_supply_sector_model):
    scenario_data = {}

    for scenario in [scenario]:
        for variant in scenario['variants']:
            for data_key, data_value in variant['data'].items():
                spec = Spec.from_dict(
                    [provides for provides in scenario['provides']
                     if provides['name'] == data_key][0])
                nda = np.random.random(spec.shape)
                da = DataArray(spec, nda)
                key = (scenario['name'], variant['name'], data_key)
                scenario_data[key] = da

    return scenario_data


@fixture
def get_scenario():
    """Return sample scenario
    """
    return {
        "name": "Economy",
        "description": "Economic projections for the UK",
        "provides": [
            {
                'name': "gva",
                'description': "GVA",
                'dtype': "float",
                'unit': "million GBP"
            }
        ],
        "variants": [
            {
                "name": "Central Economy (High)",
                "data": {
                    "gva": 3,
                }
            }
        ]
    }


@fixture(scope='function')
def get_narrative():
    """Return sample narrative
    """
    return {
        'name': 'technology',
        'description': 'Describes the evolution of technology',
        'provides': {
            'energy_demand': ['smart_meter_savings']
        },
        'variants': [
            {
                'name': 'high_tech_dsm',
                'description': 'High takeup of smart technology on the demand side',
                'data': {
                    'smart_meter_savings': 'high_tech_dsm.csv'
                }
            }
        ]
    }


@fixture
def sample_narratives(get_narrative):
    """Return sample narratives
    """
    return [
        get_narrative,
        {
            'name': 'governance',
            'description': 'Defines the nature of governance and influence upon decisions',
            'provides': {
                'energy_demand': ['homogeneity_coefficient']
            },
            'variants': [
                {
                    'name': 'Central Planning',
                    'description': 'Stronger role for central government in planning and ' +
                                   'regulation, less emphasis on market-based solutions',
                    'data': {
                        'homogeneity_coefficient': 'homogeneity_coefficient.csv'
                    }
                },
            ],
        },
    ]


@fixture
def sample_narrative_data(sample_narratives, get_sector_model, energy_supply_sector_model,
                          water_supply_sector_model):
    narrative_data = {}
    sos_model_name = 'energy'
    sector_models = {}
    sector_models[get_sector_model['name']] = get_sector_model
    sector_models[energy_supply_sector_model['name']] = energy_supply_sector_model
    sector_models[water_supply_sector_model['name']] = water_supply_sector_model

    for narrative in sample_narratives:
        for sector_model_name, param_names in narrative['provides'].items():
            sector_model = sector_models[sector_model_name]
            for param_name in param_names:
                param = _pick_from_list(sector_model['parameters'], param_name)
                for variant in narrative['variants']:
                    spec = Spec.from_dict(param)
                    nda = np.random.random(spec.shape)
                    da = DataArray(spec, nda)
                    key = (sos_model_name, narrative['name'], variant['name'], param_name)
                    narrative_data[key] = da
    return narrative_data


@fixture
def sample_results():
    spec = Spec(name='energy_use', dtype='float')
    data = np.array(1, dtype=float)
    return DataArray(spec, data)


def _pick_from_list(list_, name):
    for item in list_:
        if item['name'] == name:
            return item
    assert False, '{} not found in {}'.format(name, list_)


@fixture
def sample_dimensions(remap_months, hourly, annual):
    """Return sample dimensions
    """
    return [
        {
            'name': 'lad',
            'description': 'Local authority districts for the UK',
            'elements': ['a', 'b']
        },
        {
            'name': 'hourly',
            'description': 'The 8760 hours in the year named by hour',
            'elements': hourly
        },
        {
            'name': 'annual',
            'description': 'One annual timestep, used for aggregate yearly data',
            'elements': annual,
        },
        {
            'name': 'remap_months',
            'description': 'Remapped months to four representative months',
            'elements': remap_months,
        },
        {
            'name': 'technology_type',
            'description': 'Technology dimension for narrative fixture',
            'elements': [
                {'name': 'water_meter'},
                {'name': 'electricity_meter'},
            ]
        },
        {
            'name': 'county',
            'elements': [
                {'name': 'oxford'}
            ]
        },
        {
            'name': 'season',
            'elements': [
                {'name': 'cold_month'},
                {'name': 'spring_month'},
                {'name': 'hot_month'},
                {'name': 'fall_month'}
            ]
        }
    ]


@fixture
def get_dimension():
    return {
        "name": "annual",
        "description": "Single annual interval of 8760 hours",
        "elements":
            [
                {
                    "end": "PT8760H",
                    "id": 1,
                    "start": "PT0H"
                }
            ]
    }


@fixture
def hourly():
    return [
        {
            'name': str(n),
            'interval': [['PT{}H'.format(n), 'PT{}H'.format(n+1)]]
        }
        for n in range(8)  # should be 8760
    ]


@fixture
def annual():
    return [
        {
            'name': 1,
            'interval': [['PT0H', 'PT8760H']]
        }
    ]


@fixture
def remap_months():
    """Remapping four representative months to months across the year

    In this case we have a model which represents the seasons through
    the year using one month for each season. We then map the four
    model seasons 1, 2, 3 & 4 onto the months throughout the year that
    they represent.

    The data will be presented to the model using the four time intervals,
    1, 2, 3 & 4. When converting to hours, the data will be replicated over
    the year.  When converting from hours to the model time intervals,
    data will be averaged and aggregated.

    """
    data = [
        {'name': 'cold_month', 'interval': [['P0M', 'P1M'], ['P1M', 'P2M'], ['P11M', 'P12M']]},
        {'name': 'spring_month', 'interval': [['P2M', 'P3M'], ['P3M', 'P4M'], ['P4M', 'P5M']]},
        {'name': 'hot_month', 'interval': [['P5M', 'P6M'], ['P6M', 'P7M'], ['P7M', 'P8M']]},
        {'name': 'fall_month', 'interval': [['P8M', 'P9M'], ['P9M', 'P10M'], ['P10M', 'P11M']]}
    ]
    return data


@fixture
def minimal_model_run():
    return {
        'name': 'test_modelrun',
        'timesteps': [2010, 2015, 2010],
        'sos_model': 'energy'
    }


@fixture
def strategies():
    return [
        {
            'type': 'pre-specified-planning',
            'description': 'a description',
            'model_name': 'test_model',
            'interventions': [
                {'name': 'a', 'build_year': 2020},
                {'name': 'b', 'build_year': 2025},
            ]
        },
        {
            'type': 'rule-based',
            'description': 'reduce emissions',
            'path': 'planning/energyagent.py',
            'classname': 'EnergyAgent'
        }
    ]


@fixture
def unit_definitions():
    return ['kg = kilograms']


@fixture
def dimension():
    return {'name': 'category', 'elements': [1, 2, 3]}


@fixture
def conversion_source_spec():
    return Spec(name='a', dtype='float', unit='ml')


@fixture
def conversion_sink_spec():
    return Spec(name='b', dtype='float', unit='ml')


@fixture
def conversion_coefficients():
    return np.array([[1]])


@fixture
def scenario(sample_dimensions):

    return deepcopy({
        'name': 'mortality',
        'description': 'The annual mortality rate in UK population',
        'provides': [
            {
                'name': 'mortality',
                'dims': ['lad'],
                'coords': {'lad': sample_dimensions[0]['elements']},
                'dtype': 'float',
            }
        ],
        'variants': [
            {
                'name': 'low',
                'description': 'Mortality (Low)',
                'data': {
                    'mortality': 'mortality_low.csv',
                },
            }
        ]
    })


@fixture
def scenario_no_coords(scenario):
    scenario = deepcopy(scenario)
    for spec in scenario['provides']:
        try:
            del spec['coords']
        except KeyError:
            pass
    return scenario


@fixture
def narrative_no_coords(get_narrative):
    get_narrative = deepcopy(get_narrative)
    for spec in get_narrative['provides']:
        try:
            del spec['coords']
        except KeyError:
            pass
    return get_narrative


@fixture
def state():
    return [
        {
            'name': 'test_intervention',
            'build_year': 1900
        }
    ]
