import numpy as np
from pytest import fixture, raises
from smif.convert.area import RegionSet
from smif.convert.interval import IntervalSet
from smif.model.scenario_model import ScenarioModel, ScenarioModelBuilder


@fixture(scope='function')
def get_scenario_model_object():

    data = np.array([[[3.]], [[5.]], [[1.]]], dtype=float)
    scenario_model = ScenarioModel('test_scenario_model')
    scenario_model.add_output('raininess',
                              scenario_model.regions.get_entry('LSOA'),
                              scenario_model.intervals.get_entry('annual'),
                              'ml')
    scenario_model.add_data('raininess', data, [2010, 2011, 2012])
    return scenario_model


class TestScenarioObject:

    def test_serialise_scenario(self):
        scenario_model = ScenarioModel('High Population (ONS)')
        scenario_model.add_output('population_count',
                                  scenario_model.regions.get_entry('LSOA'),
                                  scenario_model.intervals.get_entry('annual'),
                                  'people')
        scenario_model.description = 'The High ONS Forecast for UK population out to 2050'
        scenario_model.scenario_set = 'population'
        actual = scenario_model.as_dict()
        expected = {'name': 'High Population (ONS)',
                    'description': 'The High ONS Forecast for UK population out to 2050',
                    'scenario_set': 'population',
                    'parameters': [{
                        'name': 'population_count',
                        'spatial_resolution': 'LSOA',
                        'temporal_resolution': 'annual',
                        'units': 'people'}]
                    }
        assert actual == expected

    def test_serialise_scenario_two_outputs(self):
        scenario_model = ScenarioModel('High Population (ONS)')
        scenario_model.add_output('population_count',
                                  scenario_model.regions.get_entry('LSOA'),
                                  scenario_model.intervals.get_entry('annual'),
                                  'people')
        scenario_model.add_output('population_density',
                                  scenario_model.regions.get_entry('LSOA'),
                                  scenario_model.intervals.get_entry('annual'),
                                  'people/km^2')
        scenario_model.description = 'The High ONS Forecast for UK population out to 2050'
        scenario_model.scenario_set = 'population'
        actual = scenario_model.as_dict()
        expected = {'name': 'High Population (ONS)',
                    'description': 'The High ONS Forecast for UK population out to 2050',
                    'scenario_set': 'population',
                    'parameters': [{
                        'name': 'population_count',
                        'spatial_resolution': 'LSOA',
                        'temporal_resolution': 'annual',
                        'units': 'people'},
                        {'name': 'population_density',
                         'spatial_resolution': 'LSOA',
                         'temporal_resolution': 'annual',
                         'units': 'people/km^2'}
                    ]
                    }
        assert actual == expected


class TestScenarioModelData:

    def test_nest_scenario_data(self,
                                setup_country_data,
                                get_scenario_model_object):
        data = {'length': [
            {
                'year': 2015,
                'region': 'GB',
                'interval': 'wet_season',
                'value': 3
            },
            {
                'year': 2015,
                'region': 'GB',
                'interval': 'dry_season',
                'value': 5
            },
            {
                'year': 2015,
                'region': 'NI',
                'interval': 'wet_season',
                'value': 1
            },
            {
                'year': 2015,
                'region': 'NI',
                'interval': 'dry_season',
                'value': 2
            },
            {
                'year': 2016,
                'region': 'GB',
                'interval': 'wet_season',
                'value': 4
            },
            {
                'year': 2016,
                'region': 'GB',
                'interval': 'dry_season',
                'value': 6
            },
            {
                'year': 2016,
                'region': 'NI',
                'interval': 'wet_season',
                'value': 1
            },
            {
                'year': 2016,
                'region': 'NI',
                'interval': 'dry_season',
                'value': 2.5
            }
        ]}

        expected = np.array([
            # 2015
            [
                # GB
                [3, 5],
                # NI
                [1, 2]
            ],
            # 2016
            [
                # GB
                [4, 6],
                # NI
                [1, 2.5]
            ]
        ], dtype=float)

        builder = ScenarioModelBuilder('test_scenario_model')

        interval_data = [
            {'id': 'wet_season', 'start': 'P0M', 'end': 'P5M'},
            {'id': 'dry_season', 'start': 'P5M', 'end': 'P10M'},
            {'id': 'wet_season', 'start': 'P10M', 'end': 'P1Y'},
        ]
        hack = ScenarioModel('hacky')

        hack.intervals.register(
            IntervalSet('seasonal', interval_data))
        hack.regions.register(
            RegionSet('country', setup_country_data['features']))

        config = {'name': 'mass',
                  'scenario_set': '',
                  'parameters': [{
                      'spatial_resolution': 'country',
                      'temporal_resolution': 'seasonal',
                      'units': 'kg',
                      'name': 'length'
                      }]}
        builder.construct(config, data, [2015, 2016])
        scenario = builder.finish()

        actual = scenario.get_data('length')
        assert np.allclose(actual, expected)

    def test_scenario_data_defaults(self, setup_region_data):
        data = {'length': [
            {
                'year': 2015,
                'interval': 1,
                'value': 3.14,
                'region': 'oxford'
            }
        ]}

        expected = np.array([[[3.14]]])

        builder = ScenarioModelBuilder('length')
        builder.construct({
            'name': 'length',
            'scenario_set': '',
            'parameters': [{
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'units': 'm',
                'name': 'length'
                }]
        }, data, [2015])
        scenario = builder.finish()
        assert scenario.get_data('length') == expected

    def test_scenario_data_missing_year(self, setup_region_data,
                                        ):
        data = {'length': [
            {
                'value': 3.14
            }
        ]}

        builder = ScenarioModelBuilder('length')

        msg = "Scenario data item missing year"
        with raises(ValueError) as ex:
            builder.construct({
                'name': 'length',
                'scenario_set': '',
                'parameters': [{
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'units': 'm',
                    'name': 'length'
                    }]
            }, data, [2015])
        assert msg in str(ex.value)

    def test_scenario_data_missing_param_region(self, setup_region_data,
                                                ):
        data = {'length': [
            {
                'value': 3.14,
                'region': 'missing',
                'interval': 1,
                'year': 2015
            }
        ]}

        builder = ScenarioModelBuilder('length')

        msg = "Region 'missing' not defined in set 'LSOA' for parameter 'length'"
        with raises(ValueError) as ex:
            builder.construct({
                'name': 'length',
                'scenario_set': '',
                'parameters': [{
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'units': 'm',
                    'name': 'length'
                    }]
            }, data, [2015])
        assert msg in str(ex)

    def test_scenario_data_missing_param_interval(self, setup_region_data,
                                                  ):
        data = {'length': [
            {
                'value': 3.14,
                'region': 'oxford',
                'interval': 1,
                'year': 2015
            },
            {
                'value': 3.14,
                'region': 'oxford',
                'interval': 'extra',
                'year': 2015
            }
        ]}

        builder = ScenarioModelBuilder('length')
        msg = "Interval 'extra' not defined in set 'annual' for parameter 'length'"
        with raises(ValueError) as ex:
            builder.construct({
                'name': 'length',
                'scenario_set': '',
                'parameters': [{
                    'name': 'length',
                    'units': 'm',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual'}]},
                data, [2015])
        assert msg in str(ex)

    def test_data_list_to_array(self):

        data = [
            {
                'year': 2010,
                'value': 3,
                'region': 'oxford',
                'interval': 1
            },
            {
                'year': 2011,
                'value': 5,
                'region': 'oxford',
                'interval': 1
            },
            {
                'year': 2012,
                'value': 1,
                'region': 'oxford',
                'interval': 1
            }
        ]

        builder = ScenarioModelBuilder("test_scenario_model")

        spatial = builder.region_register.get_entry('LSOA')
        temporal = builder.interval_register.get_entry('annual')

        actual = builder._data_list_to_array('raininess', data,
                                             [2010, 2011, 2012],
                                             spatial, temporal)
        expected = np.array([[[3.]], [[5.]], [[1.]]], dtype=float)
        np.testing.assert_equal(actual, expected)
