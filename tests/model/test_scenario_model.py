from unittest.mock import Mock

from pytest import fixture
from smif.model.scenario_model import ScenarioModel, ScenarioModelBuilder


@fixture(scope='function')
def get_scenario_model_object():
    scenario_model = ScenarioModel('test_scenario_model')
    scenario_model.add_output('raininess',
                              scenario_model.regions.get_entry('LSOA'),
                              scenario_model.intervals.get_entry('annual'),
                              'ml')
    # data = np.array([[[3.]], [[5.]], [[1.]]], dtype=float)
    # scenario_model.add_data('raininess', data, [2010, 2011, 2012])
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
        expected = {
            'name': 'High Population (ONS)',
            'description': 'The High ONS Forecast for UK population out to 2050',
            'scenario_set': 'population',
            'facets': [
                {
                    'name': 'population_count',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'units': 'people'
                }
            ]
        }
        assert actual == expected

    def test_serialise_scenario_two_outputs(self, setup_folder_structure):
        scenario_model = ScenarioModel('High Population (ONS)')
        scenario_model.add_output('population_count',
                                  scenario_model.regions.get_entry('LSOA'),
                                  scenario_model.intervals.get_entry('annual'),
                                  'people')
        scenario_model.add_output('population_density',
                                  scenario_model.regions.get_entry('LSOA'),
                                  scenario_model.intervals.get_entry('annual'),
                                  'people / kilometer ** 2')
        scenario_model.description = 'The High ONS Forecast for UK population out to 2050'
        scenario_model.scenario_set = 'population'
        actual = scenario_model.as_dict()
        # sort to match expected output
        actual['facets'].sort(key=lambda p: p['name'])

        expected = {
            'name': 'High Population (ONS)',
            'description': 'The High ONS Forecast for UK population out to 2050',
            'scenario_set': 'population',
            'facets': [
                {
                    'name': 'population_count',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'units': 'people'
                },
                {
                    'name': 'population_density',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'units': 'people / kilometer ** 2'
                }
            ]
        }
        assert actual == expected


class TestScenarioModelData:

    def test_scenario_data(self):
        """Scenario model simulate method should be a no-op
        """
        builder = ScenarioModelBuilder('test_scenario_model')
        config = {
            'name': 'mass',
            'scenario_set': '',
            'facets': [
                {
                    'name': 'length',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'units': 'kg'
                }
            ]
        }
        builder.construct(config)
        scenario = builder.finish()

        data_handle = Mock()
        actual = scenario.simulate(data_handle)
        assert actual is data_handle
        data_handle.assert_not_called()
