#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conftest.py for smif.

Read more about conftest.py under:
https://pytest.org/latest/plugins.html
"""
from __future__ import absolute_import, division, print_function

import copy
import json
import logging
import os

from pytest import fixture
from smif.data_layer.load import dump

from .convert.conftest import regions_half_squares, remap_months

logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                    filemode='w')


@fixture
def setup_empty_folder_structure(tmpdir_factory):
    folder_list = [
        'config',
        os.path.join('config', 'model_runs'),
        os.path.join('config', 'sos_models'),
        os.path.join('config', 'sector_models'),
        'data',
        os.path.join('data', 'initial_conditions'),
        os.path.join('data', 'interventions'),
        os.path.join('data', 'narratives'),
        os.path.join('data', 'dimensions'),
        os.path.join('data', 'scenarios'),
        os.path.join('data', 'coefficients'),
        os.path.join('data', 'strategies'),
        'models',
        'results'
    ]

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

    initial_conditions_file = test_folder.join('data', 'initial_conditions', 'init_system.yml')
    dump(initial_system, str(initial_conditions_file))

    planned_interventions_file = test_folder.join(
        'data', 'interventions', 'planned_interventions.yml')
    dump(planned_interventions, str(planned_interventions_file))

    remap_months_file = test_folder.join('data', 'dimensions', 'remap.yml')
    data = remap_months
    dump(data, str(remap_months_file))

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
        'narrative_sets': [
            'technology'
        ],
        'sector_models': [
            'energy_demand',
            'energy_supply'
        ],
        'dependencies': [
            {
                'source': 'population',
                'source_output': 'count',
                'sink': 'energy_demand',
                'sink_input': 'population'
            },
            {
                'source': 'energy_demand',
                'source_output': 'gas_demand',
                'sink': 'energy_supply',
                'sink_input': 'natural_gas_demand'
            }
        ]
    }


@fixture
def get_sector_model(annual, hourly, regions_half_squares):
    """Return sample sector_model
    """
    return {
        'name': 'energy_demand_sample',
        'description': "Computes the energy demand of the"
                       "UK population for each timestep",
        'classname': 'EnergyDemandWrapper',
        'path': '../../models/energy_demand/run.py',
        'inputs': [
            {
                'name': 'population',
                'dims': ['lad', 'annual'],
                'coords': {
                    'lad': regions_half_squares,
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
                'dims': ['lad', 'hourly'],
                'coords': {
                    'lad': regions_half_squares,
                    'hourly': hourly
                },
                'absolute_range': [0, float('inf')],
                'expected_range': [0.01, 10],
                'unit': 'GWh'
            }
        ],
        'parameters': [
            {
                'name': 'assump_diff_floorarea_pp',
                'description': "Difference in floor area per person"
                               "in end year compared to base year",
                'absolute_range': [0, float('inf')],
                'expected_range': [0.5, 2],
                'default': 1,
                'unit': 'percentage'
            }
        ],
        'interventions': [],
        'initial_conditions': []
    }


@fixture
def get_sector_model_no_coords(get_sector_model):
    model = copy.deepcopy(get_sector_model)
    for spec_group in ('inputs', 'outputs', 'parameters'):
        for spec in model[spec_group]:
            try:
                del spec['coords']
            except KeyError:
                pass
    return model


@fixture
def sample_scenarios():
    """Return sample scenario_set
    """
    return [
        {
            'name': 'population',
            'description': 'The annual change in UK population',
            'provides': [
                {
                    'name': "population_count",
                    'description': "The count of population",
                    'dtype': 'int'
                },
            ],
            'variants': [
                {
                    'name': 'High Population (ONS)',
                    'description': 'The High ONS Forecast for UK population out to 2050',
                    'data': {
                        'population_count': 'population_high.csv',
                    },
                },
                {
                    'name': 'Low Population (ONS)',
                    'description': 'The Low ONS Forecast for UK population out to 2050',
                    'data': {
                        'population_count': 'population_low.csv',
                    },
                },
            ],
        },
    ]


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
                    "gva": "economy_high.csv",
                }
            }
        ]
    }


@fixture
def sample_narratives():
    """Return sample narratives
    """
    return [
        {
            'name': 'technology',
            'description': 'Defines the rate and nature of technological change',
            'provides': [],
            'variants': [
                {
                    'name': 'Energy Demand - High Tech',
                    'description': 'High penetration of SMART technology on the demand side',
                    'data': {
                        '': 'energy_demand_high_tech.yml',
                    }
                },
            ],
        },
        {
            'name': 'governance',
            'description': 'Defines the nature of governance and influence upon decisions',
            'provides': [],
            'variants': [
                {
                    'name': 'Central Planning',
                    'description': 'Stronger role for central government in planning and ' +
                                   'regulation, less emphasis on market-based solutions',
                    'data': {
                        '': 'central_planning.yml',
                    },
                },
            ],
        },
    ]


@fixture(scope='function')
def get_narrative():
    """Return sample narrative
    """
    return {
        "name": "technology",
        "provides": [],
        "variants": [
            {
                "name": "High Tech Demand Side Management",
                "description": "High penetration of SMART technology on the demand side",
                "data": {
                    '': "high_tech_dsm.yml",
                }
            }
        ]
    }


@fixture
def sample_dimensions(regions_half_squares, remap_months, hourly, annual):
    """Return sample dimensions
    """
    return [
        {
            'name': 'lad',
            'description': 'Local authority districts for the UK',
            'elements': regions_half_squares,
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
        }
    ]


@fixture
def get_dimension():
    return {
        "name":"annual",
        "description": "Single annual interval of 8760 hours",
        "elements":
            [
                {
                    "end": "PT8760H",
                    "id": "1",
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
            'name': '1',
            'interval': [['PT0H', 'PT8760H']]
        }
    ]
