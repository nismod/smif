# -*- coding: utf-8 -*-

import os
from pytest import raises

from smif.controller import Controller, SosModel, SosModelBuilder
from smif.cli.parse_model_config import SosModelReader
from smif.sector_model import SectorModel
from fixtures.water_supply import one_dependency, one_input, WaterSupplySectorModel


class TestController():
    # TODO replace setup with builder; possibly use fixture for test controller
    def test_run_sector_model(self, setup_project_folder):
        config_file_path = os.path.join(str(setup_project_folder), "config", "model.yaml")
        reader = SosModelReader(config_file_path)
        reader.load()
        cont = Controller(reader.data)
        cont.run_sector_model('water_supply')

    def test_invalid_sector_model(self, setup_project_folder):
        config_file_path = os.path.join(str(setup_project_folder), "config", "model.yaml")
        reader = SosModelReader(config_file_path)
        reader.load()
        cont = Controller(reader.data)
        with raises(AssertionError):
            cont.run_sector_model('invalid_sector_model')


class TestSosModelBuilder():

    def test_sos_builder(self, setup_project_folder):

        builder = SosModelBuilder()


        planning_path = project_path.join('planning', 'pre-specified.yaml')
        builder.load_planning([str(planning_path)])

        builder.add_timesteps([2010, 2011, 2012])

        builder.add_planning({})

        model = WaterSupplySectorModel()
        model.name = 'water_supply'

        builder.add_model(model)
        assert isinstance(builder.sos_model.model_list['water_supply'], SectorModel)

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']
        # TODO check if there is a requirement to report all assets in the system

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
