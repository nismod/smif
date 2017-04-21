# -*- coding: utf-8 -*-

from pytest import fixture, raises
from smif import SpaceTimeValue
from smif.decision import Planning
from smif.sector_model import SectorModel
from smif.sos_model import ModelSet, SosModel, SosModelBuilder

from .fixtures.water_supply import WaterSupplySectorModel


@fixture(scope='function')
def get_sos_model_only_scenario_dependencies(setup_region_data):
    builder = SosModelBuilder()
    builder.add_timesteps([2010, 2011, 2012])
    builder.load_region_sets({'LSOA': setup_region_data['features']})
    interval_data = [
        {
            'id': 1,
            'start': 'P0Y',
            'end': 'P1Y'
        }
    ]
    builder.load_interval_sets({'annual': interval_data})
    builder.add_resolution_mapping({
        'scenario': {
            'raininess': {
                'temporal_resolution': 'annual',
                'spatial_resolution': 'LSOA'
            }
        }
    })
    builder.add_scenario_data({
        "raininess": [
            {
                'year': 2010,
                'value': 3,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            },
            {
                'year': 2011,
                'value': 5,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            },
            {
                'year': 2012,
                'value': 1,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            }
        ]
    })

    ws = WaterSupplySectorModel()
    ws.name = 'water_supply'
    ws.inputs = [
        {
            'name': 'raininess',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    ws.outputs = [
        {
            'name': 'cost',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        },
        {
            'name': 'water',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    ws.interventions = [
        {"name": "water_asset_a", "location": "oxford"},
        {"name": "water_asset_b", "location": "oxford"},
        {"name": "water_asset_c", "location": "oxford"},
    ]
    ws_data = {
        'name': ws.name,
        'initial_conditions': [],
        'inputs': ws.inputs,
        'outputs': ws.outputs,
        'interventions': ws.interventions,
    }

    builder.add_model(ws)
    builder.add_model_data(ws, ws_data)

    ws2 = WaterSupplySectorModel()
    ws2.name = 'water_supply_2'
    ws2.inputs = [
        {
            'name': 'raininess',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]

    builder.add_model(ws2)

    return builder.finish()


@fixture(scope='function')
def get_sos_model_with_model_dependency(setup_region_data):
    builder = SosModelBuilder()
    builder.add_timesteps([2010])
    interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
    builder.load_interval_sets({'annual': interval_data})
    builder.load_region_sets({'LSOA': setup_region_data['features']})
    builder.add_resolution_mapping({
        'scenario': {
            'raininess': {
                'temporal_resolution': 'annual',
                'spatial_resolution': 'LSOA'
            }
        }
    })
    builder.add_scenario_data({
        "raininess": [
            {
                'year': 2010,
                'value': 3,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            },
            {
                'year': 2011,
                'value': 5,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            },
            {
                'year': 2012,
                'value': 1,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            }
        ]
    })

    ws = WaterSupplySectorModel()
    ws.name = 'water_supply'
    ws.inputs = [
        {
            'name': 'raininess',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]

    ws.outputs = [
        {
            'name': 'water',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        },
        {
            'name': 'cost',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    ws.interventions = [
        {"name": "water_asset_a", "location": "oxford"},
        {"name": "water_asset_b", "location": "oxford"},
        {"name": "water_asset_c", "location": "oxford"}
    ]
    builder.add_model(ws)

    ws2 = WaterSupplySectorModel()
    ws2.name = 'water_supply_2'
    ws2.inputs = [
        {
            'name': 'water',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]

    builder.add_model(ws2)

    return builder.finish()


@fixture(scope='function')
def get_sos_model_with_summed_dependency(setup_region_data):
    builder = SosModelBuilder()
    builder.add_timesteps([2010])
    builder.load_region_sets({'LSOA': setup_region_data['features']})
    interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
    builder.load_interval_sets({'annual': interval_data})
    builder.add_resolution_mapping({
        'scenario': {
            'raininess': {
                'temporal_resolution': 'annual',
                'spatial_resolution': 'LSOA'
            }
        }
    })
    builder.add_scenario_data({
        "raininess": [
            {
                'year': 2010,
                'value': 3,
                'units': 'ml',
                'region': 'oxford',
                'interval': 1
            }
        ]
    })

    ws = WaterSupplySectorModel()
    ws.name = 'water_supply'
    ws.inputs = [
        {
            'name': 'raininess',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    ws.outputs = [
        {
            'name': 'water',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    builder.add_model(ws)

    ws2 = WaterSupplySectorModel()
    ws2.name = 'water_supply_2'
    ws2.inputs = [
        {
            'name': 'raininess',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    ws2.outputs = [
        {
            'name': 'water',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    builder.add_model(ws2)

    ws3 = WaterSupplySectorModel()
    ws3.inputs = [
        {
            'name': 'water',
            'spatial_resolution': 'LSOA',
            'temporal_resolution': 'annual'
        }
    ]
    ws3.name = 'water_supply_3'
    builder.add_model(ws3)

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

    def test_set_timesteps(self):
        sos_model = SosModel()
        sos_model.timesteps = [2010, 2011, 2012]
        assert sos_model.timesteps == [2010, 2011, 2012]

        # remove duplicates
        sos_model.timesteps = [2010, 2011, 2012, 2012, 2012]
        assert sos_model.timesteps == [2010, 2011, 2012]

        # sort order
        sos_model.timesteps = [2011, 2012, 2010]
        assert sos_model.timesteps == [2010, 2011, 2012]

    def test_timestep_before(self):
        sos_model = SosModel()
        sos_model.timesteps = [2010, 2011, 2012]
        assert sos_model.timestep_before(2010) is None
        assert sos_model.timestep_before(2011) == 2010
        assert sos_model.timestep_before(2012) == 2011
        assert sos_model.timestep_before(2013) is None

    def test_timestep_after(self):
        sos_model = SosModel()
        sos_model.timesteps = [2010, 2011, 2012]
        assert sos_model.timestep_after(2010) == 2011
        assert sos_model.timestep_after(2011) == 2012
        assert sos_model.timestep_after(2012) is None
        assert sos_model.timestep_after(2013) is None

    def test_guess_outputs_zero(self, get_sos_model_only_scenario_dependencies):
        """If no previous timestep has results, guess outputs as zero
        """
        sos_model = get_sos_model_only_scenario_dependencies
        ws_model = sos_model.models['water_supply']
        model_set = ModelSet(
            {ws_model},
            sos_model
        )

        results = model_set.guess_results(ws_model, 2010)
        expected = {
            "cost": [
                SpaceTimeValue("oxford", 1, 0, "unknown")
            ],
            "water": [
                SpaceTimeValue("oxford", 1, 0, "unknown")
            ]
        }
        assert results == expected

    def test_guess_outputs_last_year(self, get_sos_model_only_scenario_dependencies):
        """If a previous timestep has results, guess outputs as identical
        """
        sos_model = get_sos_model_only_scenario_dependencies
        ws_model = sos_model.models['water_supply']
        model_set = ModelSet(
            {ws_model},
            sos_model
        )

        expected = {
            "cost": [
                SpaceTimeValue("oxford", 1, 3.14, "unknown")
            ],
            "water": [
                SpaceTimeValue("oxford", 1, 2.71, "unknown")
            ]
        }

        # set up data as though from previous timestep simulation
        year_before = sos_model.timestep_before(2011)
        assert year_before == 2010
        sos_model._results[year_before]['water_supply'] = expected

        results = model_set.guess_results(ws_model, 2011)
        assert results == expected

    def test_converged_first_iteration(self, get_sos_model_only_scenario_dependencies):
        """Should not report convergence after a single iteration
        """
        sos_model = get_sos_model_only_scenario_dependencies
        ws_model = sos_model.models['water_supply']
        model_set = ModelSet(
            {ws_model},
            sos_model
        )

        results = model_set.guess_results(ws_model, 2010)
        model_set.iterated_results[ws_model.name] = [results]

        assert not model_set.converged(2010)

    def test_converged_two_identical(self, get_sos_model_only_scenario_dependencies):
        """Should report converged if the last two output sets are identical
        """
        sos_model = get_sos_model_only_scenario_dependencies
        ws_model = sos_model.models['water_supply']
        model_set = ModelSet(
            {ws_model},
            sos_model
        )

        results = model_set.guess_results(ws_model, 2010)
        model_set.iterated_results = {
            "water_supply": [results, results]
        }

        assert model_set.converged(2010)

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

    def test_run_with_planning(self, get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        planning_data = [
            {
                'name': 'water_asset_a',
                'build_date': 2010,
            },
            {
                'name': 'energy_asset_a',
                'build_date': 2011,
            }
        ]
        planning = Planning(planning_data)
        sos_model.planning = planning

        model = sos_model.models['water_supply']
        actual = sos_model.get_decisions(model, 2010)
        assert actual[0].name == 'water_asset_a'
        assert actual[0].location == 'oxford'

        _, results = sos_model.run_sector_model_timestep(model, 2010)
        actual = results
        expected = {'cost': 2.528, 'water': 2}
        for key, value in expected.items():
            assert actual[key][0].value == value

        _, results = sos_model.run_sector_model_timestep(model, 2011)
        actual = results
        expected = {'cost': 2.528, 'water': 2}
        for key, value in expected.items():
            assert actual[key][0].value == value

    def test_add_data_series(self):
        """Expect addition to work on lists of SpaceTimeValues
        """
        list_a = [
            SpaceTimeValue('a', '1', 1, 'days'),
            SpaceTimeValue('a', '2', 2, 'days'),
            SpaceTimeValue('b', '1', 3, 'days'),
            SpaceTimeValue('b', '2', 4, 'days'),
        ]

        list_b = [
            SpaceTimeValue('b', '2', 14, 'days'),
            SpaceTimeValue('b', '1', 13, 'days'),
            SpaceTimeValue('a', '2', 12, 'days'),
            SpaceTimeValue('a', '1', 11, 'days'),
        ]

        expected = [
            SpaceTimeValue('a', '1', 12, 'days'),
            SpaceTimeValue('a', '2', 14, 'days'),
            SpaceTimeValue('b', '1', 16, 'days'),
            SpaceTimeValue('b', '2', 18, 'days'),
        ]
        actual = SosModel.add_data_series(list_a, list_b)
        assert actual == expected

    def test_dependency_aggregation(self, get_sos_model_with_summed_dependency):
        sos_model = get_sos_model_with_summed_dependency
        sos_model.run()


@fixture(scope='function')
def get_config_data(setup_project_folder, setup_region_data):
    path = setup_project_folder
    water_supply_wrapper_path = str(
        path.join(
            'models', 'water_supply', '__init__.py'
        )
    )
    return {
        "timesteps": [2010, 2011, 2012],
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
                    'units': 'ml',
                    'region': 'oxford',
                    'interval': 1
                }
            ]
        },
        "region_sets": {'LSOA': setup_region_data['features']},
        "interval_sets": {
            'annual': [
                {
                    'id': 1,
                    'start': 'P0Y',
                    'end': 'P1Y'
                }
            ]
        },
        "resolution_mapping": {
            'scenario': {
                'raininess': {
                    'temporal_resolution': 'annual',
                    'spatial_resolution': 'LSOA'
                }
            }
        }
    }


class TestSosModelBuilder():

    def test_builder(self, setup_project_folder):

        builder = SosModelBuilder()

        builder.add_timesteps([2010, 2011, 2012])

        model = WaterSupplySectorModel()
        model.name = 'water_supply'
        model_data = {
            'name': model.name,
            'interventions': [
                {"name": "water_asset_a", "location": "oxford"},
                {"name": "water_asset_b", "location": "oxford"},
                {"name": "water_asset_c", "location": "oxford"}
            ],
            'initial_conditions': [],
            'inputs': [],
            'outputs': [],
        }

        builder.add_model(model)
        builder.add_model_data(model, model_data)
        assert isinstance(builder.sos_model.models['water_supply'],
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

    def test_construct(self, get_config_data):
        """Test constructing from single dict config
        """
        config = get_config_data
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()

        assert isinstance(sos_model, SosModel)
        assert sos_model.sector_models == ['water_supply']
        assert isinstance(sos_model.models['water_supply'], SectorModel)
        assert sos_model.timesteps == [2010, 2011, 2012]

    def test_missing_planning_asset(self, get_config_data):
        config = get_config_data
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

    def test_missing_planning_timeperiod(self, get_config_data):
        config = get_config_data
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

    def test_scenario_dependency(self, get_config_data, setup_region_data):
        """Expect successful build with dependency on scenario data

        Should raise error if no spatial or temporal sets are defined
        """
        config = get_config_data
        config["sector_model_data"][0]["inputs"] = [
            {
                'name': 'raininess',
                'spatial_resolution': 'blobby',
                'temporal_resolution': 'mega'
            }
        ]

        builder = SosModelBuilder()
        builder.construct(config)

        with raises(ValueError):
            builder.finish()

        builder.load_region_sets({'blobby': setup_region_data['features']})

        interval_data = [
            {
                'id': 'ultra',
                'start': 'P0Y',
                'end': 'P1Y'
            }
        ]
        builder.load_interval_sets({'mega': interval_data})
        builder.finish()

    def test_build_valid_dependencies(self, one_dependency,
                                      get_config_data, setup_region_data):
        builder = SosModelBuilder()
        builder.construct(get_config_data)

        ws = WaterSupplySectorModel()
        ws.name = "water_supply_broken"
        ws.inputs = one_dependency
        builder.add_model(ws)

        with raises(AssertionError) as error:
            builder.finish()

        msg = "Missing dependency: water_supply_broken depends on macguffins produced" + \
              ", which is not supplied."
        assert str(error.value) == msg

    def test_cyclic_dependencies(self, setup_region_data):
        a_inputs = [
            {
                'name': 'b value',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual'
            }
        ]

        a_outputs = [
            {
                'name': 'a value',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual'
            }
        ]

        b_inputs = [
            {
                'name': 'a value',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual'
            }
        ]

        b_outputs = [
            {
                'name': 'b value',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual'
            }
        ]

        builder = SosModelBuilder()
        builder.add_timesteps([2010])
        builder.add_planning([])
        builder.load_region_sets({'LSOA': setup_region_data['features']})
        interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
        builder.load_interval_sets({'annual': interval_data})

        a_model = WaterSupplySectorModel()
        a_model.name = "a_model"
        a_model.inputs = a_inputs
        a_model.outputs = a_outputs
        builder.add_model(a_model)

        b_model = WaterSupplySectorModel()
        b_model.name = "b_model"
        b_model.inputs = b_inputs
        b_model.outputs = b_outputs
        builder.add_model(b_model)

        builder.finish()

    def test_nest_scenario_data(self, setup_country_data):
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
        interval_data = [
            {'id': 'winter', 'start': 'P0M', 'end': 'P2M'},
            {'id': 'spring', 'start': 'P2M', 'end': 'P5M'},
            {'id': 'summer', 'start': 'P5M', 'end': 'P8M'},
            {'id': 'autumn', 'start': 'P8M', 'end': 'P11M'},
            {'id': 'winter', 'start': 'P11M', 'end': 'P1Y'},
        ]
        builder.load_interval_sets({'annual': interval_data})
        builder.load_region_sets({'country': setup_country_data['features']})
        builder.add_resolution_mapping({
            'scenario': {
                'mass': {
                    'spatial_resolution': 'country',
                    'temporal_resolution': 'annual'
                }
            }
        })
        builder.add_scenario_data(data)

        actual = builder.sos_model._scenario_data

        assert actual[2015]["mass"] == [
            SpaceTimeValue("GB", "spring", 3, 'kg'),
            SpaceTimeValue('GB', 'summer', 5, 'kg'),
            SpaceTimeValue('NI', 'spring', 1, 'kg')
        ]
        assert actual == expected

        expected_value = [3, 5, 1]
        expected_region = ['GB', 'GB', 'NI']

        for exp_val, exp_reg, entry in zip(expected_value,
                                           expected_region,
                                           actual[2015]['mass']):
            assert entry.value == exp_val
            assert entry.region == exp_reg
            assert entry.units == 'kg'

    def test_scenario_data_defaults(self, setup_region_data):
        data = {
            "length": [
                {
                    'year': 2015,
                    'interval': 1,
                    'value': 3.14,
                    'units': 'm',
                    'region': 'oxford'
                }
            ]
        }

        expected = {2015: {"length": [SpaceTimeValue("oxford", 1, 3.14, 'm')]}}

        builder = SosModelBuilder()
        interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
        builder.load_interval_sets({'annual': interval_data})
        builder.load_region_sets({'LSOA': setup_region_data['features']})
        builder.add_resolution_mapping({
            'scenario': {
                'length': {
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual'
                }
            }
        })
        builder.add_scenario_data(data)
        assert builder.sos_model._scenario_data == expected

    def test_scenario_data_missing_year(self, setup_region_data):
        data = {
            "length": [
                {
                    'value': 3.14,
                    'units': 'm'
                }
            ]
        }

        builder = SosModelBuilder()
        interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
        builder.load_interval_sets({'annual': interval_data})
        builder.load_region_sets({'LSOA': setup_region_data['features']})
        builder.add_resolution_mapping({
            'scenario': {
                'length': {
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual'
                }
            }
        })

        msg = "Scenario data item missing year"
        with raises(ValueError) as ex:
            builder.add_scenario_data(data)
        assert msg in str(ex.value)

    def test_scenario_data_missing_param_mapping(self):
        data = {
            "length": [
                {
                    'value': 3.14,
                    'units': 'm',
                    'year': 2015
                }
            ]
        }

        builder = SosModelBuilder()

        msg = "Parameter length not registered in resolution mapping"
        with raises(ValueError) as ex:
            builder.add_scenario_data(data)
        assert msg in str(ex.value)

    def test_scenario_data_missing_param_region(self, setup_region_data):
        data = {
            "length": [
                {
                    'value': 3.14,
                    'units': 'm',
                    'region': 'missing',
                    'interval': 1,
                    'year': 2015
                }
            ]
        }

        builder = SosModelBuilder()
        interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
        builder.load_interval_sets({'annual': interval_data})
        builder.load_region_sets({'LSOA': setup_region_data['features']})
        builder.add_resolution_mapping({
            'scenario': {
                'length': {
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual'
                }
            }
        })

        msg = "Region missing not defined in set LSOA for parameter length"
        with raises(ValueError) as ex:
            builder.add_scenario_data(data)
        assert msg in str(ex)

    def test_scenario_data_missing_param_interval(self, setup_region_data):
        data = {
            "length": [
                {
                    'value': 3.14,
                    'units': 'm',
                    'region': 'oxford',
                    'interval': 'missing',
                    'year': 2015
                }
            ]
        }

        builder = SosModelBuilder()
        interval_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
        builder.load_interval_sets({'annual': interval_data})
        builder.load_region_sets({'LSOA': setup_region_data['features']})
        builder.add_resolution_mapping({
            'scenario': {
                'length': {
                    'spatial_resolution': 'LSOA',
                    'temporal_resolution': 'annual'
                }
            }
        })

        msg = "Interval missing not defined in set annual for parameter length"
        with raises(ValueError) as ex:
            builder.add_scenario_data(data)
        assert msg in str(ex)

    def test_inputs_property(self,
                             get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        actual = sos_model.inputs

        expected = {'raininess': ['water_supply', 'water_supply_2']}

        assert isinstance(actual, dict)

        for key, value in expected.items():
            assert key in actual.keys()
            for entry in value:
                assert entry in actual[key]

    def test_outputs_property(self,
                              get_sos_model_only_scenario_dependencies):
        sos_model = get_sos_model_only_scenario_dependencies
        actual = sos_model.outputs

        expected = {'cost': ['water_supply']}

        assert isinstance(actual, dict)

        for key, value in expected.items():
            assert key in actual.keys()
            for entry in value:
                assert entry in actual[key]

    def test_single_dependency(self,
                               get_sos_model_with_model_dependency):
        sos_model = get_sos_model_with_model_dependency
        outputs = sos_model.outputs

        expected_outputs = {'water': ['water_supply']}

        assert isinstance(outputs, dict)

        for key, value in expected_outputs.items():
            assert key in outputs.keys()
            for entry in value:
                assert entry in outputs[key]

        inputs = sos_model.inputs

        expected_inputs = {
            'raininess': ['water_supply'],
            'water': ['water_supply_2']
        }

        assert isinstance(inputs, dict)

        for key, value in expected_inputs.items():
            assert key in inputs.keys()
            for entry in value:
                assert entry in inputs[key]

        sos_model.run()
