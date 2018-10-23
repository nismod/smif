"""Holds fixtures for the data layer module tests
"""
from pytest import fixture
from smif.data_layer import DatafileInterface, MemoryInterface
from smif.metadata import Spec


@fixture(scope='function')
def get_memory_handler():
    return MemoryInterface()


@fixture(scope='function')
def config_handler(get_handler_binary):
    return get_handler_binary


@fixture(scope='function')
def get_handler_csv(setup_folder_structure, sample_scenarios, sample_narratives,
                    sample_dimensions):
    handler = DatafileInterface(str(setup_folder_structure), 'local_csv', validation=False)
    for scenario in sample_scenarios:
        handler.write_scenario(scenario)
    for dimension in sample_dimensions:
        handler.write_dimension(dimension)
    return handler


@fixture(scope='function')
def get_handler_binary(setup_folder_structure, sample_scenarios, sample_narratives,
                       sample_dimensions, get_sector_model):
    handler = DatafileInterface(str(setup_folder_structure), 'local_binary', validation=False)
    for scenario in sample_scenarios:
        handler.write_scenario(scenario)
    for dimension in sample_dimensions:
        handler.write_dimension(dimension)
    handler.write_sector_model(get_sector_model)
    return handler


@fixture
def get_scenario_data():
    """Return sample scenario_data
    """
    return [
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'timestep': 2017
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'timestep': 2017
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'timestep': 2017
        },
        {
            'population_count': 210,
            'county': 'oxford',
            'season': 'fall_month',
            'timestep': 2017
        },
    ]


@fixture
def get_faulty_scenario_data():
    """Return faulty sample scenario_data
    """
    return [
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'year': 2017
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'year': 2017
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'year': 2017
        },
        {
            'population_count': 210,
            'county': 'oxford',
            'season': 'fall_month',
            'year': 2017
        },
    ]


@fixture(scope='function')
def get_remapped_scenario_data():
    """Return sample scenario_data
    """
    data = [
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'timestep': 2015
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'timestep': 2015
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'timestep': 2015
        },
        {
            'population_count': 210,
            'county': 'oxford',
            'season': 'fall_month',
            'timestep': 2015
        },
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'timestep': 2016
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'timestep': 2016
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'timestep': 2016
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'fall_month',
            'timestep': 2016
        }
    ]
    spec = Spec(
        name='people',
        unit='people',
        dtype='int',
        dims=['county', 'season'],
        coords={
            'county': ['oxford'],
            'season': ['cold_month', 'spring_month', 'hot_month', 'fall_month']
        }
    )
    return data, spec


@fixture
def parameter_spec():
    spec = Spec.from_dict({
            'name': 'smart_meter_savings',
            'description': "Difference in floor area per person"
                           "in end year compared to base year",
            'absolute_range': [0, float('inf')],
            'expected_range': [0.5, 2],
            'default': 'data_file.csv',
            'unit': 'percentage',
            'dtype': 'float'})
    return spec
