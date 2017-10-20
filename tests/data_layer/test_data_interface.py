"""Test data interface
"""
import csv
import json
import os
from copy import copy
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
            }
        ],
        'units': 'user_units.txt',
        'scenarios':
        [
            {
                'description': 'The High ONS Forecast for UK population out to 2050',
                'name': 'High Population (ONS)',
                'parameters': [
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
                'parameters': [
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
        'interventions': ['energy_demand.yml'],
        'initial_conditions': ['energy_demand_init.yml']
    }


@fixture(scope='function')
def get_scenario_data():
    """Return sample scenario_data
    """
    return [
        {
            'value': 100,
            'units': 'people',
            'region': 'GB',
            'year': 2015
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'GB',
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'GB',
            'year': 2017
        }
    ]


@fixture(scope='function')
def get_narrative_data():
    """Return sample narrative_data
    """
    return [
        {
            'energy_demand': [
                {
                    'name': 'smart_meter_savings',
                    'value': 8
                }
            ],
            'water_supply': [
                {
                    'name': 'clever_water_meter_savings',
                    'value': 8
                }
            ]
        }
    ]


class TestDatafileInterface():

    def test_sos_model_run(self, get_sos_model_run, setup_folder_structure):
        """ Test to write two sos_model_run configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)
        config_handler = DatafileInterface(str(basefolder))

        sos_model_run1 = get_sos_model_run
        sos_model_run1['name'] = 'sos_model_run1'
        config_handler.write_sos_model_run(sos_model_run1)

        sos_model_run2 = get_sos_model_run
        sos_model_run2['name'] = 'sos_model_run2'
        config_handler.write_sos_model_run(sos_model_run2)

        sos_model_runs = config_handler.read_sos_model_runs()
        assert sos_model_runs[0]['name'] == 'sos_model_run1'
        assert sos_model_runs[1]['name'] == 'sos_model_run2'

        sos_model_run3 = get_sos_model_run
        sos_model_run3['name'] = 'sos_model_run3'
        config_handler.update_sos_model_run('sos_model_run2', sos_model_run3)

        sos_model_runs = config_handler.read_sos_model_runs()
        print(sos_model_runs)

        assert sos_model_runs[0]['name'] == 'sos_model_run1'
        assert sos_model_runs[1]['name'] == 'sos_model_run3'

    def test_sos_model(self, get_sos_model, setup_folder_structure):
        """ Test to write two soS_model configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)
        config_handler = DatafileInterface(str(basefolder))

        sos_model1 = get_sos_model
        sos_model1['name'] = 'sos_model_1'
        config_handler.write_sos_model(sos_model1)

        sos_model2 = copy(get_sos_model)
        sos_model2['name'] = 'sos_model_2'
        config_handler.write_sos_model(sos_model2)

        sos_models = config_handler.read_sos_models()
        assert sos_model1 in sos_models
        assert sos_model2 in sos_models

        sos_model3 = copy(get_sos_model)
        sos_model3['name'] = 'sos_model_3'
        config_handler.update_sos_model('sos_model_2', sos_model3)

        sos_models = config_handler.read_sos_models()
        assert sos_model1 in sos_models
        assert sos_model2 not in sos_models
        assert sos_model3 in sos_models

    def test_sector_model(self, setup_folder_structure, get_project_config,
                          get_sector_model):
        """ Test to write a sector_model configuration to a Yaml file
        read the Yaml file and compare that the result is equal.
        Finally check if the name shows up the the readlist.
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)
        config_handler = DatafileInterface(str(basefolder))

        sector_model1 = copy(get_sector_model)
        sector_model1['name'] = 'sector_model_1'
        config_handler.write_sector_model(sector_model1)

        sector_model2 = copy(get_sector_model)
        sector_model2['name'] = 'sector_model_2'
        config_handler.write_sector_model(sector_model2)

        sector_models = config_handler.read_sector_models()

        assert sector_models.count(sector_model1['name']) == 1
        assert sector_models.count(sector_model2['name']) == 1

        sector_model1_read = config_handler.read_sector_model(
            sector_model1['name'])
        assert sector_model1_read == sector_model1

        sector_model3 = get_sector_model
        sector_model3['name'] = 'sector_model_3'
        config_handler.update_sector_model('sector_model_2', sector_model3)

        sector_models = config_handler.read_sector_models()
        assert sector_models.count(sector_model2['name']) == 0
        assert sector_models.count(sector_model3['name']) == 1

    def test_region_definition_data(self, setup_folder_structure, get_project_config,
                                    setup_region_data):
        """ Test to dump a region_definition_set (GeoJSON) data-file and then read the data
        using the datafile interface. Finally check if the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)
        region_definition_data = setup_region_data

        with open(os.path.join(str(basefolder), 'data', 'region_definitions',
                               'test_region.json'), 'w+') as region_definition_file:
            json.dump(region_definition_data, region_definition_file)

        config_handler = DatafileInterface(str(basefolder))
        test_region_definition = config_handler.read_region_definition_data(
            'lad')

        assert test_region_definition[0]['properties']['name'] == 'oxford'

    def test_scenario_data(self, setup_folder_structure, get_project_config,
                           get_scenario_data):
        """ Test to dump a scenario (CSV) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)
        scenario_data = get_scenario_data

        keys = scenario_data[0].keys()
        with open(os.path.join(str(basefolder), 'data', 'scenarios',
                               'population_high.csv'), 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        config_handler = DatafileInterface(str(basefolder))
        test_scenario = config_handler.read_scenario_data(
            'High Population (ONS)')

        assert len(test_scenario) == 1
        assert 'population_count' in test_scenario
        assert test_scenario['population_count'][0]['region'] == 'GB'

    def test_narrative_data(self, setup_folder_structure, get_project_config,
                            get_narrative_data):
        """ Test to dump a narrative (yml) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        basefolder = setup_folder_structure
        narrative_data_path = os.path.join(str(basefolder), 'data', 'narratives',
                                           'central_planning.yml')
        dump(get_narrative_data, narrative_data_path)

        config_handler = DatafileInterface(str(basefolder))
        test_narrative = config_handler.read_narrative_data('Central Planning')

        assert test_narrative[0]['energy_demand'][0]['name'] == 'smart_meter_savings'

    def test_project_region_definitions(self, setup_folder_structure,
                                        get_project_config):
        """ Test to read and write the project configuration
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        config_handler = DatafileInterface(str(basefolder))

        # region_definition sets / read existing (from fixture)
        region_definitions = config_handler.read_region_definitions()
        assert region_definitions[0]['name'] == 'lad'
        assert len(region_definitions) == 1

        # region_definition sets / add
        region_definition = {
            'name': 'lad_NL',
            'description': 'Local authority districts for the Netherlands',
            'filename': 'lad_NL.csv'
        }
        config_handler.write_region_definition(region_definition)
        region_definitions = config_handler.read_region_definitions()
        assert len(region_definitions) == 2
        for region_definition in region_definitions:
            if region_definition['name'] == 'lad_NL':
                assert region_definition['filename'] == 'lad_NL.csv'

        # region_definition sets / modify
        region_definition = {
            'name': 'lad_NL',
            'description': 'Local authority districts for the Netherlands',
            'filename': 'lad_NL_V2.csv'
        }
        config_handler.update_region_definition('lad_NL', region_definition)
        region_definitions = config_handler.read_region_definitions()
        assert len(region_definitions) == 2
        for region_definition in region_definitions:
            if region_definition['name'] == 'lad_NL':
                assert region_definition['filename'] == 'lad_NL_V2.csv'

        # region_definition sets / modify unique identifier (name)
        region_definition['name'] = 'name_change'
        config_handler.update_region_definition('lad_NL', region_definition)
        region_definitions = config_handler.read_region_definitions()
        assert len(region_definitions) == 2
        for region_definition in region_definitions:
            if region_definition['name'] == 'name_change':
                assert region_definition['filename'] == 'lad_NL_V2.csv'

    def test_project_interval_definitions(self, setup_folder_structure,
                                          get_project_config):
        """ Test to read and write the project configuration
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        config_handler = DatafileInterface(str(basefolder))

        # interval_definitions / read existing (from fixture)
        interval_definitions = config_handler.read_interval_definitions()
        assert interval_definitions[0]['name'] == 'hourly'
        assert len(interval_definitions) == 2

        # interval_definition sets / add
        interval_definition = {
            'name': 'monthly',
            'description': 'The 12 months of the year',
            'filename': 'monthly.csv'
        }
        config_handler.write_interval_definition(interval_definition)
        interval_definitions = config_handler.read_interval_definitions()
        assert len(interval_definitions) == 3
        for interval_definition in interval_definitions:
            if interval_definition['name'] == 'monthly':
                assert interval_definition['filename'] == 'monthly.csv'

        # interval_definition sets / modify
        interval_definition = {
            'name': 'monthly',
            'description': 'The 12 months of the year',
            'filename': 'monthly_V2.csv'
        }
        config_handler.update_interval_definition(
            interval_definition['name'], interval_definition)
        interval_definitions = config_handler.read_interval_definitions()
        assert len(interval_definitions) == 3
        for interval_definition in interval_definitions:
            if interval_definition['name'] == 'monthly':
                assert interval_definition['filename'] == 'monthly_V2.csv'

        # region_definition sets / modify unique identifier (name)
        interval_definition['name'] = 'name_change'
        config_handler.update_interval_definition(
            'monthly', interval_definition)
        interval_definitions = config_handler.read_interval_definitions()
        assert len(interval_definitions) == 3
        for interval_definition in interval_definitions:
            if interval_definition['name'] == 'name_change':
                assert interval_definition['filename'] == 'monthly_V2.csv'

    def test_project_scenario_sets(self, setup_folder_structure, get_project_config):
        """ Test to read and write the project configuration
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        config_handler = DatafileInterface(str(basefolder))

        # Scenario sets / read existing (from fixture)
        scenario_sets = config_handler.read_scenario_sets()
        assert scenario_sets[0]['name'] == 'population'
        assert len(scenario_sets) == 1

        # Scenario sets / add
        scenario_set = {
            'description': 'The annual mortality rate in UK population',
            'name': 'mortality'
        }
        config_handler.write_scenario_set(scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'mortality':
                expected = 'The annual mortality rate in UK population'
                assert scenario_set['description'] == expected

        # Scenario sets / modify
        scenario_set = {
            'description': 'The annual mortality rate in NL population',
            'name': 'mortality'
        }
        config_handler.update_scenario_set(scenario_set['name'], scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'mortality':
                expected = 'The annual mortality rate in NL population'
                assert scenario_set['description'] == expected

        # Scenario sets / modify unique identifier (name)
        scenario_set['name'] = 'name_change'
        config_handler.update_scenario_set('mortality', scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'name_change':
                expected = 'The annual mortality rate in NL population'
                assert scenario_set['description'] == expected

    def test_project_scenarios(self, setup_folder_structure, get_project_config):
        """ Test to read and write the project configuration
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        config_handler = DatafileInterface(str(basefolder))

        # Scenarios / read existing (from fixture)
        scenarios = config_handler.read_scenario_set('population')
        assert scenarios[0]['name'] == 'High Population (ONS)'
        assert len(scenarios) == 2

        # Scenarios / add
        scenario = {
            'description': 'The Medium ONS Forecast for UK population out to 2050',
            'filename': 'population_medium.csv',
            'name': 'Medium Population (ONS)',
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
        config_handler.write_scenario(scenario)
        scenarios = config_handler.read_scenario_set('population')
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'Medium Population (ONS)':
                assert scenario['filename'] == 'population_medium.csv'

        # Scenarios / modify
        scenario['filename'] = 'population_med.csv'
        config_handler.update_scenario(scenario['name'], scenario)
        scenarios = config_handler.read_scenario_set('population')
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'Medium Population (ONS)':
                assert scenario['filename'] == 'population_med.csv'

        # Scenarios / modify unique identifier (name)
        scenario['name'] = 'name_change'
        config_handler.update_scenario('Medium Population (ONS)', scenario)
        scenarios = config_handler.read_scenario_set('population')
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'name_change':
                assert scenario['filename'] == 'population_med.csv'

    def test_project_narrative_sets(self, setup_folder_structure, get_project_config):
        """ Test to read and write the project configuration
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        config_handler = DatafileInterface(str(basefolder))

        # Narrative sets / read existing (from fixture)
        narrative_sets = config_handler.read_narrative_sets()
        assert narrative_sets[0]['name'] == 'technology'
        assert len(narrative_sets) == 2

        # Narrative sets / add
        narrative_set = {
            'description': 'New narrative set',
            'name': 'new_narrative_set'
        }
        config_handler.write_narrative_set(narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'new_narrative_set':
                assert narrative_set['description'] == 'New narrative set'

        # Narrative sets / modify
        narrative_set = {
            'description': 'New narrative set description',
            'name': 'new_narrative_set'
        }
        config_handler.update_narrative_set(
            narrative_set['name'], narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'new_narrative_set':
                assert narrative_set['description'] == 'New narrative set description'

        # Narrative sets / modify unique identifier (name)
        narrative_set['name'] = 'name_change'
        config_handler.update_narrative_set('new_narrative_set', narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'name_change':
                assert narrative_set['description'] == 'New narrative set description'

    def test_project_narratives(self, setup_folder_structure, get_project_config):
        """ Test to read and write the project configuration
        """
        basefolder = setup_folder_structure
        project_config_path = os.path.join(
            str(basefolder), 'config', 'project.yml')
        dump(get_project_config, project_config_path)

        config_handler = DatafileInterface(str(basefolder))

        # Narratives / read existing (from fixture)
        narratives = config_handler.read_narrative_set('technology')
        assert narratives[0]['name'] == 'Energy Demand - High Tech'
        assert len(narratives) == 1

        # narratives / add
        narrative = {
            'description': 'Low penetration of SMART technology on the demand side',
            'filename': 'energy_demand_low_tech.yml',
            'name': 'Energy Demand - Low Tech',
            'narrative_set': 'technology',
        }
        config_handler.write_narrative(narrative)
        narratives = config_handler.read_narrative_set('technology')
        assert len(narratives) == 2
        for narrative in narratives:
            if narrative['name'] == 'Energy Demand - Low Tech':
                assert narrative['filename'] == 'energy_demand_low_tech.yml'

        # narratives / modify
        narrative['filename'] = 'energy_demand_low_tech_v2.yml'
        config_handler.update_narrative(narrative['name'], narrative)
        narratives = config_handler.read_narrative_set('technology')
        assert len(narratives) == 2
        for narrative in narratives:
            if narrative['name'] == 'Energy Demand - Low Tech':
                assert narrative['filename'] == 'energy_demand_low_tech_v2.yml'

        # narratives / modify unique identifier (name)
        narrative['name'] = 'name_change'
        config_handler.update_narrative('Energy Demand - Low Tech', narrative)
        narratives = config_handler.read_narrative_set('technology')
        assert len(narratives) == 2
        for narrative in narratives:
            if narrative['name'] == 'name_change':
                assert narrative['filename'] == 'energy_demand_low_tech_v2.yml'
