# -*- coding: utf-8 -*-

import os

from pytest import raises
from smif.cli import read_sector_model_data_from_config
from smif.cli.parse_model_config import SosModelReader
from smif.controller import Controller, SosModel, SosModelBuilder
from smif.sector_model import SectorModel

from .fixtures.water_supply import WaterSupplySectorModel


class TestController():
    # TODO replace setup with builder; possibly use fixture for test controller
    def test_run_sector_model(self, setup_project_folder):
        config_file_path = os.path.join(str(setup_project_folder), "config", "model.yaml")
        reader = SosModelReader(config_file_path)
        reader.load()
        config_data = reader.data

        main_config_dir = os.path.dirname(config_file_path)
        config_data['sector_model_data'] = read_sector_model_data_from_config(
            main_config_dir,
            config_data['sector_model_config']
        )

        cont = Controller(config_data)
        cont.run_sector_model('water_supply')

    def test_invalid_sector_model(self, setup_project_folder):
        config_file_path = os.path.join(str(setup_project_folder), "config", "model.yaml")
        reader = SosModelReader(config_file_path)
        reader.load()
        config_data = reader.data

        main_config_dir = os.path.dirname(config_file_path)
        config_data['sector_model_data'] = read_sector_model_data_from_config(
            main_config_dir,
            config_data['sector_model_config']
        )

        cont = Controller(config_data)
        with raises(AssertionError):
            cont.run_sector_model('invalid_sector_model')


class TestSosModelBuilder():

    def test_sos_builder(self, setup_project_folder):

        builder = SosModelBuilder()

        builder.add_timesteps([2010, 2011, 2012])
        builder.add_planning([])

        model = WaterSupplySectorModel()
        model.name = 'water_supply'

        builder.add_model(model)
        assert isinstance(builder.sos_model.model_list['water_supply'], SectorModel)

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']
        # TODO check if there is a requirement to report all assets in the system

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
