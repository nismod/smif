# -*- coding: utf-8 -*-

from pytest import raises
from smif.sector_model import SectorModel
from smif.sos_model import SosModel, SosModelBuilder

from .fixtures.water_supply import WaterSupplySectorModel


class TestSosModelBuilder():

    def test_sos_builder(self, setup_project_folder):

        builder = SosModelBuilder()

        builder.add_timesteps([2010, 2011, 2012])
        builder.add_planning([])

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

    def test_build_api(self):
        builder = SosModelBuilder()
        builder.add_timesteps([2010, 2011, 2012])
        builder.add_planning([])

        ws_model = WaterSupplySectorModel()
        ws_model.name = 'water_supply'
        builder.add_model(ws_model)

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']

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
