# -*- coding: utf-8 -*-

from pytest import fixture, raises
from smif import SpaceTimeValue
from smif.sector_model import SectorModel
from smif.sos_model import SosModel, SosModelBuilder

from .fixtures.water_supply import WaterSupplySectorModel


@fixture(scope='function')
def get_sos_model_only_scenario_dependencies():
    builder = SosModelBuilder()
    builder.add_timesteps([2010])
    builder.add_scenario_data({
        "raininess": [
            {
                'year': 2010,
                'value': 3,
                'units': 'ml'
            },
            {
                'year': 2011,
                'value': 5,
                'units': 'ml'
            },
            {
                'year': 2012,
                'value': 1,
                'units': 'ml'
            }
        ]
    })

    ws = WaterSupplySectorModel()
    ws.name = 'water_supply'
    ws.inputs = {
        'dependencies': [
            {
                'name': 'raininess',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'from_model': 'scenario'
            }
        ]
    }
    builder.add_model(ws)

    ws2 = WaterSupplySectorModel()
    ws2.name = 'water_supply_2'
    ws2.inputs = {
        'dependencies': [
            {
                'name': 'raininess',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual',
                'from_model': 'scenario'
            }
        ]
    }
    builder.add_model(ws2)

    return builder.finish()


class TestSosModel():

    def test_run_static(self, get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        sos_model.run()

    def test_run_no_timesteps(self, get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        sos_model.timesteps = []

        with raises(ValueError) as ex:
            sos_model.run()
        assert "No timesteps" in str(ex.value)

    def test_run_sequential(self, get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        sos_model.timesteps = [2010, 2011, 2012]
        sos_model.run()

    def test_run_single_sector(self, get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        sos_model.run_sector_model('water_supply')

    def test_run_missing_sector(self, get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies

        with raises(AssertionError) as ex:
            sos_model.run_sector_model('impossible')
        assert "Model 'impossible' does not exist" in str(ex.value)


class TestSosModelBuilder():

    def test_builder(self, setup_project_folder):

        builder = SosModelBuilder()

        builder.add_timesteps([2010, 2011, 2012])

        model = WaterSupplySectorModel()
        model.name = 'water_supply'
        model.interventions = [
            {"name": "water_asset_a", "location": "oxford"},
            {"name": "water_asset_b", "location": "oxford"},
            {"name": "water_asset_c", "location": "oxford"}
        ]

        builder.add_model(model)
        assert isinstance(builder.sos_model.model_list['water_supply'],
                          SectorModel)

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']
        assert sos_model.intervention_names == [
            "water_asset_a",
            "water_asset_b",
            "water_asset_c"
        ]

    def get_config_data(self, path):
        water_supply_wrapper_path = str(
            path.join(
                'models', 'water_supply', '__init__.py'
            )
        )
        return {
            "timesteps": [2010, 2011, 2012],
            "sector_model_data": [{
                "name": "water_supply",
                "path": water_supply_wrapper_path,
                "classname": "WaterSupplySectorModel",
                "inputs": {},
                "outputs": {},
                "initial_conditions": [],
                "interventions": []
            }],
            "planning": [],
            "scenario_data": {}
        }

    def test_construct(self, setup_project_folder):
        """Test constructing from single dict config
        """
        config = self.get_config_data(setup_project_folder)
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()

        assert isinstance(sos_model, SosModel)
        assert sos_model.sector_models == ['water_supply']
        assert isinstance(sos_model.model_list['water_supply'], SectorModel)
        assert sos_model.timesteps == [2010, 2011, 2012]

    def test_missing_planning_asset(self, setup_project_folder):
        config = self.get_config_data(setup_project_folder)
        config["planning"] = [
            {
                "name": "test_intervention",
                "build_date": 2012
            }
        ]
        builder = SosModelBuilder()
        builder.construct(config)

        with raises(AssertionError) as ex:
            builder.finish()
        assert "Intervention 'test_intervention' in planning file not found" in str(ex.value)

    def test_missing_planning_timeperiod(self, setup_project_folder):
        config = self.get_config_data(setup_project_folder)
        config["planning"] = [
            {
                "name": "test_intervention",
                "location": "UK",
                "build_date": 2025
            }
        ]
        config["sector_model_data"][0]["interventions"] = [
            {
                "name": "test_intervention",
                "location": "UK"
            }
        ]
        builder = SosModelBuilder()
        builder.construct(config)

        with raises(AssertionError) as ex:
            builder.finish()
        assert "Timeperiod '2025' in planning file not found" in str(ex.value)

    def test_scenario_dependency(self, setup_project_folder):
        """Expect successful build with dependency on scenario data
        """
        config = self.get_config_data(setup_project_folder)
        config["sector_model_data"][0]["inputs"] = {
            "dependencies": [
                {
                    'name': 'population',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'from_model': 'scenario'
                }
            ]
        }
        builder = SosModelBuilder()
        builder.construct(config)
        builder.finish()

    def test_build_valid_dependencies(self, one_dependency):
        builder = SosModelBuilder()
        builder.add_timesteps([2010])
        builder.add_planning([])

        ws = WaterSupplySectorModel()
        ws.name = "water_supply"
        ws.inputs = one_dependency
        builder.add_model(ws)

        with raises(AssertionError) as error:
            builder.finish()

        msg = "Missing dependency: water_supply depends on macguffins produced " + \
              "from macguffins_model, which is not supplied."
        assert str(error.value) == msg

    def test_cyclic_dependencies(self):
        a_inputs = {
            'decision variables': [],
            'parameters': [],
            'dependencies': [
                {
                    'name': 'b value',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'from_model': 'b_model'
                }
            ]
        }

        b_inputs = {
            'decision variables': [],
            'parameters': [],
            'dependencies': [
                {
                    'name': 'a value',
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual',
                    'from_model': 'a_model'
                }
            ]
        }

        builder = SosModelBuilder()
        builder.add_timesteps([2010])
        builder.add_planning([])

        a_model = WaterSupplySectorModel()
        a_model.name = "a_model"
        a_model.inputs = a_inputs
        builder.add_model(a_model)

        b_model = WaterSupplySectorModel()
        b_model.name = "b_model"
        b_model.inputs = b_inputs
        builder.add_model(b_model)

        with raises(NotImplementedError):
            builder.finish()

    def test_nest_scenario_data(self):
        data = {
            "mass": [
                {
                    'year': 2015,
                    'region': 'GB',
                    'interval': 'spring',
                    'value': 3,
                    'units': 'kg'
                },
                {
                    'year': 2015,
                    'region': 'GB',
                    'interval': 'summer',
                    'value': 5,
                    'units': 'kg'
                },
                {
                    'year': 2015,
                    'region': 'NI',
                    'interval': 'spring',
                    'value': 1,
                    'units': 'kg'
                },
                {
                    'year': 2016,
                    'region': 'GB',
                    'interval': 'spring',
                    'value': 4,
                    'units': 'kg'
                }
            ]
        }

        expected = {
            2015: {
                "mass": [
                    SpaceTimeValue('GB', 'spring', 3, 'kg'),
                    SpaceTimeValue('GB', 'summer', 5, 'kg'),
                    SpaceTimeValue('NI', 'spring', 1, 'kg')
                    ]
                    },
            2016: {
                "mass": [
                    SpaceTimeValue('GB', 'spring', 4, 'kg')
                         ]

            }
        }

        builder = SosModelBuilder()
        builder.add_scenario_data(data)

        actual = builder.sos_model._scenario_data

        assert actual[2015]["mass"] == \
            [SpaceTimeValue("GB", "spring", 3, 'kg'),
             SpaceTimeValue('GB', 'summer', 5, 'kg'),
             SpaceTimeValue('NI', 'spring', 1, 'kg')]
        assert actual == expected

        expected_value = [3, 5, 1]
        expected_region = ['GB', 'GB', 'NI']

        for exp_val, exp_reg, entry in zip(expected_value,
                                           expected_region,
                                           actual[2015]['mass']):
            assert entry.value == exp_val
            assert entry.region == exp_reg
            assert entry.units == 'kg'

    def test_scenario_data_defaults(self):
        data = {
            "length": [
                {
                    'year': 2015,
                    'interval': 'annual',
                    'value': 3.14,
                    'units': 'm',
                    'region': 'national'
                }
            ]
        }

        expected = {2015: {"length": [SpaceTimeValue("national", 'annual', 3.14, 'm')]}}

        builder = SosModelBuilder()
        builder.add_scenario_data(data)
        assert builder.sos_model._scenario_data == expected

    def test_scenario_data_missing_year(self):
        data = {
            "length": [
                {
                    'value': 3.14,
                    'units': 'm'
                }
            ]
        }

        builder = SosModelBuilder()

        msg = "Scenario data item missing year"
        with raises(ValueError) as ex:
            builder.add_scenario_data(data)
        assert msg in str(ex.value)
