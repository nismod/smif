from pytest import fixture
from smif.model.sos_model import SosModel
from smif.modelrun import ModelRunBuilder


@fixture(scope='function')
def get_model_runconfig_data(setup_project_folder, setup_region_data):
    path = setup_project_folder
    water_supply_wrapper_path = str(
        path.join(
            'models', 'water_supply', '__init__.py'
        )
    )
    return {
        "timesteps": [2010, 2011, 2012],
        "dependencies": [],
        "sector_model_data": [
            {
                "name": "water_supply",
                "path": water_supply_wrapper_path,
                "classname": "WaterSupplySectorModel",
                "inputs": [],
                "outputs": [],
                "initial_conditions": [],
                "interventions": []
            }
        ],
        "planning": [],
        "scenario_data": {
            'raininess': [
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
        },
        "region_sets": {'BSOA': setup_region_data['features']},
        "interval_sets": {
            'yearly': [
                {
                    'id': 1,
                    'start': 'P0Y',
                    'end': 'P1Y'
                }
            ]
        },
        "scenario_metadata": [
            {
                'name': 'raininess',
                'temporal_resolution': 'yearly',
                'spatial_resolution': 'BSOA',
                'units': 'ml'
            }
        ]
    }


@fixture(scope='function')
def get_model_run(setup_project_folder, setup_region_data):
    path = setup_project_folder
    water_supply_wrapper_path = str(
        path.join(
            'models', 'water_supply', '__init__.py'
        )
    )

    config_data = {
        'timesteps': [2010, 2011, 2012],
        'dependencies': [],
        'region_sets': {},
        'interval_sets': {},
        'planning': [],
        'scenario_metadata':
            [{
                'name': 'raininess',
                'temporal_resolution': 'yearly',
                'spatial_resolution': 'BSOA',
                'units': 'ml'
            }],
        'scenario_data': {
            "raininess": [
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
        },
        "sector_model_data": [
            {
                "name": "water_supply",
                "path": water_supply_wrapper_path,
                "classname": "WaterSupplySectorModel",
                "inputs": [],
                "outputs": [],
                "initial_conditions": [],
                "interventions": []
            }
        ]
        }

    builder = ModelRunBuilder()
    builder.construct(config_data)
    return builder.finish()


class TestModelRunBuilder:

    def test_builder(self, get_model_runconfig_data):

        config_data = get_model_runconfig_data

        builder = ModelRunBuilder()
        builder.construct(config_data)

        modelrun = builder.finish()

        assert isinstance(modelrun.sos_model, SosModel)

        assert modelrun.name == ''
        assert modelrun.model_horizon == [2010, 2011, 2012]
        assert modelrun.status == 'Built'
        assert modelrun.strategies is None
        assert modelrun.narratives is None


class TestModelRun:

    def test_run_static(self, get_model_run):
        model_run = get_model_run
        model_run.run()

    def test_run_with_global_parameters(self, get_model_run):
        model_run = get_model_run

        sos_model = model_run.sos_model
        sos_model.name = 'test_sos_model'

        sos_model_param = {
            'name': 'sos_model_param',
            'description': 'A global parameter passed to all contained models',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'}

        sos_model.add_parameter(sos_model_param)

        sos_model.models['water_supply'].simulate = lambda x, y: y

        model_run.run()

        assert model_run.results == {}
        assert sos_model.results == {}

        assert 'sos_model_param' in sos_model.parameters['test_sos_model']

    def test_run_with_sector_parameters(self, get_model_run):

        model_run = get_model_run

        sos_model = model_run.sos_model
        sos_model.name = 'test_sos_model'

        sector_model = sos_model.models['water_supply']

        sector_model_param = {
            'name': 'sector_model_param',
            'description': 'A model parameter passed to a specific model',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'}

        sector_model.add_parameter(sector_model_param)

        assert 'sector_model_param' in sos_model.parameters['water_supply']

        model_run.run()

        assert model_run.results == {}
        assert sos_model.results == {}
