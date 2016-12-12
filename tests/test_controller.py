from unittest.mock import MagicMock

from pytest import raises
from smif.controller import (Controller, SectorConfigReader,
                             SectorModelBuilder, SosModel, SoSModelBuilder,
                             SoSModelReader)
from smif.inputs import ModelInputs
from smif.sectormodel import SectorModel


class TestController():

    def test_model_list(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = ['water_supply']
        actual = cont.model.sector_models
        assert actual == expected

    def test_timesteps(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = [2010, 2011, 2012]
        actual = cont.model.timesteps
        assert actual == expected

    def test_timesteps_alternate_file(self, setup_project_folder,
                                      setup_config_file_timesteps_two,
                                      setup_timesteps_file_two,
                                      setup_pre_specified_planning_two):

        cont = Controller(str(setup_project_folder))

        expected = [2015, 2020, 2025]
        actual = cont.model.timesteps
        assert actual == expected

    def test_timesteps_invalid(self, setup_project_folder,
                               setup_timesteps_file_invalid):

        with raises(ValueError):
            Controller(str(setup_project_folder))

    def test_assets(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = cont.model.all_assets
        assert actual == expected

    def test_assets_two_asset_files(self, setup_project_folder,
                                    setup_assets_file_two,
                                    setup_config_file_two,
                                    setup_water_asset_d):

        cont = Controller(str(setup_project_folder))

        expected = ['water_asset_a', 'water_asset_b',
                    'water_asset_c', 'water_asset_d']
        actual = cont.model.all_assets
        assert actual == expected


class TestRunModel():

    def test_run_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))

        # Monkey patching inputs as run.py fixture cannot access smif.inputs
        inputs = cont.model.model_list['water_supply'].model.inputs
        model_inputs = ModelInputs(inputs)
        cont.model.model_list['water_supply'].model.inputs = model_inputs

        cont.model.run_sector_model('water_supply')

    def test_invalid_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))
        with raises(AssertionError):
            cont.model.run_sector_model('invalid_sector_model')


class TestBuildSosModel():

    def test_read_sos_model(self, setup_project_folder):

        project_path = setup_project_folder

        reader = SoSModelReader(str(project_path))
        mock_builder = MagicMock()
        reader.builder = mock_builder

        reader.construct()

        timesteps_config = project_path.join('config', 'timesteps.yaml')
        name, args, _ = reader.builder.mock_calls[0]
        assert name == 'load_timesteps'
        assert args[0] == str(timesteps_config)
        name, args, _ = reader.builder.mock_calls[1]
        assert name == 'load_models'
        assert args == (['water_supply'], str(project_path))
        name, args, _ = reader.builder.mock_calls[2]
        assert name == 'load_planning'

    def test_sos_builder(self, setup_project_folder):

        project_path = setup_project_folder
        builder = SoSModelBuilder()

        timesteps_config = project_path.join('config', 'timesteps.yaml')
        builder.load_timesteps(str(timesteps_config))

        planning_path = project_path.join('planning', 'pre-specified.yaml')
        builder.load_planning([str(planning_path)])

        model = builder.load_model('water_supply', str(project_path))
        assert isinstance(model, SectorModel)

        builder.load_models(['water_supply'], str(project_path))

        sos_model = builder.finish()
        assert isinstance(sos_model, SosModel)

        assert sos_model.timesteps == [2010, 2011, 2012]
        assert sos_model.sector_models == ['water_supply']
        assert sos_model.all_assets == ['water_asset_a', 'water_asset_b',
                                        'water_asset_c']


class TestBuildSectorModel():

    def test_sector_model_builder(self, setup_project_folder):
        project_path = setup_project_folder
        builder = SectorModelBuilder()

        builder.name_model('a model name')

        attributes = {name: str(project_path.join('models',
                                                  'water_supply', 'assets',
                                                  "{}.yaml".format(name)))
                      for name in ['water_asset_a']}

        builder.load_attributes(attributes)

        wrapper_path = str(project_path.join('models',
                                             'water_supply',
                                             'run.py'))

        builder.load_wrapper(wrapper_path)
        # builder.load_inputs()
        # builder.load_outputs()
        # builder.validate()
        model = builder.finish()
        assert isinstance(model, SectorModel)

        assert model.name == 'a model name'

        assert model.assets == ['water_asset_a']

        ext_attr = {'water_asset_a': {'capital_cost': {'unit': '£/kW',
                                                       'value': 1000
                                                       },
                                      'economic_lifetime': 25,
                                      'operational_lifetime': 25
                                      }
                    }

        assert model.attributes == ext_attr

    def test_sector_config_reader(self, setup_project_folder):
        project_folder = setup_project_folder

        reader = SectorConfigReader('water_supply', str(project_folder))

        mock_builder = MagicMock()
        reader.builder = mock_builder

        reader.construct()

        name, args, _ = reader.builder.mock_calls[0]


class TestSoSBuilderValidation():

    def test_invalid_assets_in_pre_spec_plan(self, setup_project_folder,
                                             setup_config_conflict_assets):
        msg = "Asset '{}' in planning file not found in sector assets"
        with raises(AssertionError, message=msg.format('water_asset_z')):
            Controller(str(setup_project_folder))

    def test_invalid_period_in_pre_spec_plan(self, setup_project_folder,
                                             setup_config_conflict_periods):
        msg = "Timeperiod '{}' in planning file not found model config"
        with raises(AssertionError, message=msg.format('2010')):
            Controller(str(setup_project_folder))
