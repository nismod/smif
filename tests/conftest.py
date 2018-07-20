#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    conftest.py for smif.

    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
from copy import copy

from pytest import fixture
from smif.convert.area import RegionSet
from smif.convert.area import get_register as get_region_register
from smif.convert.interval import IntervalSet
from smif.convert.interval import get_register as get_interval_register
from smif.convert.unit import get_register as get_unit_register
from smif.data_layer import DatafileInterface
from smif.data_layer.load import dump
from smif.parameters import Narrative

from .convert.conftest import (months, one_day, remap_months, remap_months_csv,
                               seasons, twenty_four_hours)
from .convert.test_area import (regions_half_squares, regions_half_triangles,
                                regions_rect, regions_single_half_square)

logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                    filemode='w')


@fixture(scope='function')
def setup_folder_structure(tmpdir_factory, oxford_region,
                           annual_intervals, initial_system,
                           planned_interventions):
    """

    Returns
    -------
    :class:`LocalPath`
        Path to the temporary folder
    """
    folder_list = [
        'config',
        os.path.join('config', 'sos_model_runs'),
        os.path.join('config', 'sos_models'),
        os.path.join('config', 'sector_models'),
        'data',
        os.path.join('data', 'initial_conditions'),
        os.path.join('data', 'interval_definitions'),
        os.path.join('data', 'interventions'),
        os.path.join('data', 'narratives'),
        os.path.join('data', 'region_definitions'),
        os.path.join('data', 'scenarios'),
        os.path.join('data', 'coefficients'),
        os.path.join('data', 'strategies'),
        'models',
        'results'
    ]

    test_folder = tmpdir_factory.mktemp("smif")

    for folder in folder_list:
        test_folder.mkdir(folder)

    region_file = test_folder.join('data', 'region_definitions', 'test_region.json')
    region_file.write(json.dumps(oxford_region))

    intervals_file = test_folder.join('data', 'interval_definitions', 'annual.csv')
    intervals_file.write("id,start,end\n1,P0Y,P1Y\n")

    intervals_file = test_folder.join('data', 'interval_definitions', 'hourly.csv')
    intervals_file.write("id,start,end\n1,PT0H,PT1H\n")

    initial_conditions_file = test_folder.join('data', 'initial_conditions', 'init_system.yml')
    dump(initial_system, str(initial_conditions_file))

    planned_interventions_file = test_folder.join(
        'data', 'interventions', 'planned_interventions.yml')
    dump(planned_interventions, str(planned_interventions_file))

    data = remap_months_csv()
    intervals_file = test_folder.join(
        'data', 'interval_definitions', 'remap.csv')
    keys = data[0].keys()
    with open(str(intervals_file), 'w+') as open_csv_file:
        dict_writer = csv.DictWriter(open_csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

    units_file = test_folder.join('data', 'user_units.txt')
    units_file.write("blobbiness = m^3 * 10^6\n")

    return test_folder


@fixture(scope='function')
def setup_runpy_file(setup_folder_structure):
    """The python script should contain an instance of SectorModel which wraps
    the sector model and allows it to be run.
    """
    base_folder = setup_folder_structure
    # Write a file for the water_supply model
    filename = base_folder.join('models', 'water_supply', '__init__.py')
    contents = """
from smif.model.sector_model import SectorModel

class WaterSupplySectorModel(SectorModel):
    def simulate(self, timestep, data=None):
        return {self.name: data}

    def extract_obj(self, results):
        return 0

"""
    filename.write(contents, ensure=True)


@fixture(scope='function')
def initial_system():
    """Initial system (interventions with build_date)
    """
    return [
        {'name': 'water_asset_a', 'build_year': 2017},
        {'name': 'water_asset_b', 'build_year': 2017},
        {'name': 'water_asset_c', 'build_year': 2017},
    ]


@fixture(scope='function')
def initial_system_bis():
    """An extra intervention for the initial system
    """
    return [{'name': 'water_asset_d', 'build_year': 2017}]


@fixture
def parameters():
    return [
        {
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        }
    ]


@fixture(scope='function')
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


@fixture(scope='function')
def water_inputs():
    return [
        {
            'name': 'reservoir pumpiness',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual',
            'units': 'magnitude'
        }
    ]


@fixture(scope='function')
def water_outputs():
    return [
        {
            'name': 'storage_state',
            'spatial_resolution': 'national',
            'temporal_resolution': 'annual',
            'units': 'Ml'
        },
        {
            'name': 'storage_blobby',
            'spatial_resolution': 'national',
            'temporal_resolution': 'annual',
            'units': 'Ml'
        },
        {
            'name': 'total_water_demand',
            'spatial_resolution': 'national',
            'temporal_resolution': 'annual',
            'units': 'Ml'
        }
    ]


@fixture(scope='session')
def annual_intervals_csv():
    return [
        {
            "start": "P0Y",
            "end": "P1Y",
            "id": '1'
        }
    ]


@fixture(scope='session')
def annual_intervals():
    return [
        (
         '1', [("P0Y", "P1Y")]
        )
    ]


@fixture(scope='session')
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


@fixture(scope='function')
def gb_ni_regions():
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
                    "name": "GB"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0, 1],
                            [0, 1.1],
                            [2, 3]
                        ]
                    ]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "NI"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [2, 3],
                            [2, 3.2],
                            [4, 5]
                        ]
                    ]
                }
            },
        ]
    }
    return data


@fixture(scope='function')
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


@fixture(scope="session", autouse=True)
def setup_registers(oxford_region, annual_intervals, tmpdir_factory):
    """One-time setup: load all the fixture region and interval
    sets into the module-level registers.
    """
    regions = get_region_register()
    lsoa = RegionSet('LSOA', oxford_region['features'])
    regions.register(lsoa)
    regions.register(regions_half_squares())
    regions.register(regions_single_half_square())
    regions.register(regions_half_triangles())
    regions.register(regions_rect())

    # register alt rect (same area)
    regions_rect_alt = copy(regions_rect())
    regions_rect_alt.name = 'rect_alt'
    regions.register(regions_rect_alt)

    intervals = get_interval_register()
    intervals.register(IntervalSet('annual', annual_intervals))
    intervals.register(IntervalSet('months', months()))
    intervals.register(IntervalSet('seasons', seasons()))
    intervals.register(IntervalSet('hourly_day', twenty_four_hours()))
    intervals.register(IntervalSet('one_day', one_day()))
    intervals.register(IntervalSet('remap_months', remap_months()))

    test_folder = tmpdir_factory.mktemp("smif")

    units_file = test_folder.join('user_units.txt')
    units_file.write("mcm = 10^6 * m^3\nGBP=[currency]\npeople=[people]\n")

    units = get_unit_register()
    units.register(str(units_file))


@fixture(scope='function')
def project_config():
    """Return sample project configuration
    """
    return {
        'project_name': 'NISMOD v2.0',
        'scenario_sets': [
            {
                'description': 'The annual change in UK population',
                'name': 'population',
                'facets': {'name': "population_count",
                           'description': "The count of population"}
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
        'region_definitions': [
            {
                'description': 'Local authority districts for the UK',
                'filename': 'test_region.json',
                'name': 'lad'
            }
        ],
        'interval_definitions': [
            {
                'description': 'The 8760 hours in the year named by hour',
                'filename': 'hourly.csv', 'name': 'hourly'
            },
            {
                'description': 'One annual timestep, used for aggregate yearly data',
                'filename': 'annual.csv', 'name': 'annual'
            },
            {
                'description': 'Remapped months to four representative months',
                'filename': 'remap.csv', 'name': 'remap_months'
            }
        ],
        'units': 'user_units.txt',
        'scenarios':
        [
            {
                'description': 'The High ONS Forecast for UK population out to 2050',
                'name': 'High Population (ONS)',
                'facets': [
                    {
                        'name': 'population_count',
                        'filename': 'population_high.csv',
                        'spatial_resolution': 'lad',
                        'temporal_resolution': 'annual',
                        'units': 'people',
                    }
                ],
                'scenario_set': 'population',
            },
            {
                'description': 'The Low ONS Forecast for UK population out to 2050',
                'name': 'Low Population (ONS)',
                'facets': [
                    {
                        'name': 'population_count',
                        'filename': 'population_low.csv',
                        'spatial_resolution': 'lad',
                        'temporal_resolution': 'annual',
                        'units': 'people',
                    }
                ],
                'scenario_set': 'population',
            }
        ],
        'narratives': [
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
        'stamp': '2017-09-20T12:53:23+00:00',
        'timesteps': [
            2015,
            2020,
            2025
        ],
        'sos_model': 'energy',
        'decision_module': 'energy_moea.py',
        'scenarios': {
            'population': 'High Population (ONS)'
        },
        'strategies': [{'strategy': 'pre-specified-planning',
                        'description': 'description of the strategy',
                        'model_name': 'energy_supply',
                        'filename': 'energy_supply.csv'}],
        'narratives': {
            'technology': [
                'Energy Demand - High Tech'
            ],
            'governance': [
                'Central Planning'
            ]
        }
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
        'narrative_sets': [
            'technology'
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
        'name': 'energy_demand_sample',
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
        'interventions': ['planned_interventions.yml'],
        'initial_conditions': ['init_system.yml']
    }


@fixture(scope='function')
def get_scenario_set():
    """Return sample scenario_set
    """
    return {
        "description": "Growth in UK economy",
        "name": "economy",
        "facets": {"name": "economy_low",
                   "description": "a description"}
    }


@fixture(scope='function')
def get_scenario():
    """Return sample scenario
    """
    return {
        "description": "Central Economy for the UK (High)",
        "name": "Central Economy (High)",
        "facets": [
            {
                "filename": "economy_low.csv",
                "name": "economy_low",
                "spatial_resolution": "national",
                "temporal_resolution": "annual",
                "units": "million people"
            }
        ],
        "scenario_set": "economy"
    }


@fixture(scope='function')
def get_narrative():
    """Return sample narrative
    """
    return {
        "description": "High penetration of SMART technology on the demand side",
        "filename": "high_tech_dsm.yml",
        "name": "High Tech Demand Side Management",
        "narrative_set": "technology"
    }


@fixture(scope='function')
def get_scenario_data():
    """Return sample scenario_data
    """
    return [
        {
            'value': 100,
            'units': 'people',
            'region': 'oxford',
            'interval': 1,
            'year': 2015
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'oxford',
            'interval': 1,
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 1,
            'year': 2017
        }
    ]


@fixture(scope='function')
def narrative_data():
    """Return sample narrative_data
    """
    return {
        'energy_demand': {
            'smart_meter_savings': 8
        },
        'water_supply': {
            'clever_water_meter_savings': 8
        }
    }


@fixture(scope='function')
def get_handler(setup_folder_structure, project_config):
    basefolder = setup_folder_structure
    project_config_path = os.path.join(
        str(basefolder), 'config', 'project.yml')
    dump(project_config, project_config_path)

    return DatafileInterface(str(basefolder), 'local_binary')


@fixture
def get_narrative_obj():
    narrative = Narrative('Energy Demand - High Tech',
                          'A description',
                          'technology')
    return narrative


@fixture
def get_narrative_set():
    return {
        'name': 'technology',
        'description': 'Describes the evolution of technology'
    }
