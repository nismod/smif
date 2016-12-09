# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from pytest import raises
from smif.controller import Controller, SosModel, SosModelBuilder
from smif.sector_model import SectorModel
from fixtures.water_supply import one_dependency, one_input, WaterSupplySectorModel


class TestController():
    # TODO replace setup with builder; possibly use fixture for test controller
    def test_run_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))

        cont.run_sector_model('water_supply')

    def test_invalid_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))
        with raises(AssertionError):
            cont.run_sector_model('invalid_sector_model')


class TestSosModelBuilder():

    def test_sos_builder(self, setup_project_folder):

        project_path = setup_project_folder
        builder = SosModelBuilder()

        timesteps_config = project_path.join('config', 'timesteps.yaml')
        builder.load_timesteps(str(timesteps_config))

        planning_path = project_path.join('planning', 'pre-specified.yaml')
        builder.load_planning([str(planning_path)])

        builder.add_planning({})

        model = WaterSupplySectorModel()
        builder.add_model(model)
        assert isinstance(builder.sos_model.model_list['water_supply'], SectorModel)

        builder.add_models([model])

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']
        assert sos_model.all_assets == ['water_asset_a', 'water_asset_b',
                                        'water_asset_c']

    def test_build_api(self, one_input):
        builder = SosModelBuilder()
        builder.add_timesteps([2010, 2011, 2012])

        ws_model = WaterSupplySectorModel()
        ws_model.name = 'water_supply'
        ws_model.inputs = one_input
        builder.add_model(ws_model)

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']

    def test_build_valid_dependencies(self, one_dependency):
        builder = SosModelBuilder()
        builder.add_timesteps([2010])

        ws = WaterSupplySectorModel()
        ws.inputs = one_dependency
        builder.add_model(ws)

        with raises(AssertionError) as e:
            builder.finish()
            assert e.msg == ""
