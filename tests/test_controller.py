import os
from pytest import raises
from smif.controller import Controller


class TestController():

    def test_model_list(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = ['water_supply']
        actual = cont.model_list
        assert actual == expected

    def test_timesteps(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = [2010, 2011, 2012]
        actual = cont.timesteps
        assert actual == expected

    def test_assets(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = cont.all_assets
        assert actual == expected

    def test_assets_two_asset_files(self, setup_project_folder,
                                    setup_assets_file_two,
                                    setup_config_file_two,
                                    setup_water_asset_d):

        cont = Controller(str(setup_project_folder))

        expected = ['water_asset_a', 'water_asset_b',
                    'water_asset_c', 'water_asset_d']
        actual = cont.all_assets
        assert actual == expected


class TestRunModel():

    def test_run_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))
        assert os.path.exists(os.path.join(str(setup_project_folder),
                                           'models',
                                           'water_supply',
                                           'run.py'))

        cont.run_sector_model('water_supply')

    def test_invalid_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))
        with raises(AssertionError):
            cont.run_sector_model('invalid_sector_model')
