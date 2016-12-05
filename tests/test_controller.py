from pytest import raises
from smif.controller import Controller
from smif.inputs import ModelInputs


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
                                      setup_timesteps_file_two):

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
