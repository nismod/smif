from pytest import fixture
from smif.data_layer import DatafileInterface, MemoryInterface
from smif.metadata import Spec


@fixture(scope='function')
def get_memory_handler():
    return MemoryInterface()


@fixture
def get_handler(get_handler_binary):
    return get_handler_binary


@fixture(scope='function')
def get_handler_csv(setup_folder_structure, sample_scenarios, sample_narratives,
                    sample_dimensions):
    handler = DatafileInterface(str(setup_folder_structure), 'local_csv')
    for scenario in sample_scenarios:
        handler.write_scenario(scenario)
    for narrative in sample_narratives:
        handler.write_narrative(narrative)
    for dimension in sample_dimensions:
        handler.write_dimension(dimension)
    return handler


@fixture(scope='function')
def get_handler_binary(setup_folder_structure, sample_scenarios, sample_narratives,
                       sample_dimensions):
    handler = DatafileInterface(str(setup_folder_structure), 'local_binary')
    for scenario in sample_scenarios:
        handler.write_scenario(scenario)
    for narrative in sample_narratives:
        handler.write_narrative(narrative)
    for dimension in sample_dimensions:
        handler.write_dimension(dimension)
    return handler


@fixture
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
def get_remapped_scenario_data():
    """Return sample scenario_data
    """
    data = [
        {
            'value': 100,
            'units': 'people',
            'region': 'oxford',
            'interval': 'cold_month',
            'year': 2015
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'oxford',
            'interval': 'spring_month',
            'year': 2015
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 'hot_month',
            'year': 2015
        },
        {
            'value': 210,
            'units': 'people',
            'region': 'oxford',
            'interval': 'fall_month',
            'year': 2015
        },
        {
            'value': 100,
            'units': 'people',
            'region': 'oxford',
            'interval': 'cold_month',
            'year': 2016
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'oxford',
            'interval': 'spring_month',
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 'hot_month',
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 'fall_month',
            'year': 2016
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
