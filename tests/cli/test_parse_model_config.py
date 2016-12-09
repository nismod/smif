# -*- coding: utf-8 -*-
from pytest import raises
from smif.cli.parse_model_config import SosModelReader

class TestSosModelReader():

    def test_read_sos_model(self, setup_project_folder):

        project_path = setup_project_folder

        reader = SosModelReader(str(project_path))
        reader.load()

        # TODO assert top level config is as expectied (pointers to timesteps, sector models)

    def test_model_list(self, setup_project_folder):

        reader = SosModelReader(str(setup_project_folder))

        expected = ['water_supply']
        actual = reader.sector_models
        assert actual == expected

    def test_timesteps(self, setup_project_folder):

        reader = SosModelReader(str(setup_project_folder))

        expected = [2010, 2011, 2012]
        actual = reader.timesteps
        assert actual == expected

    def test_timesteps_alternate_file(self, setup_project_folder,
                                      setup_config_file_timesteps_two,
                                      setup_timesteps_file_two):

        reader = SosModelReader(str(setup_project_folder))

        expected = [2015, 2020, 2025]
        actual = reader.timesteps
        assert actual == expected

    def test_timesteps_invalid(self, setup_project_folder,
                               setup_timesteps_file_invalid):

        with raises(ValueError):
            SosModelReader(str(setup_project_folder))

# TODO test for clear error messages from reading
