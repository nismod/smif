# -*- coding: utf-8 -*-
import os
from pytest import raises
from smif.cli.parse_model_config import SosModelReader

class TestSosModelReader():

    def _get_model_config(self, folder):
        return os.path.join(str(folder), "config", "model.yaml")

    def test_read_sos_model(self, setup_project_folder):

        reader = SosModelReader(self._get_model_config(setup_project_folder))
        reader.load()

        # TODO assert top level config is as expectied (pointers to timesteps, sector models)

    def test_model_list(self, setup_project_folder):

        reader = SosModelReader(self._get_model_config(setup_project_folder))
        reader.load()

        expected = [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply"
            }
        ]
        actual = reader.sector_model_data
        assert actual == expected

    def test_timesteps(self, setup_project_folder):

        reader = SosModelReader(self._get_model_config(setup_project_folder))
        reader.load()

        expected = [2010, 2011, 2012]
        actual = reader.timesteps
        assert actual == expected

    def test_timesteps_alternate_file(self, setup_project_folder,
                                      setup_config_file_timesteps_two,
                                      setup_timesteps_file_two):

        reader = SosModelReader(self._get_model_config(setup_project_folder))
        reader.load()

        expected = [2015, 2020, 2025]
        actual = reader.timesteps
        assert actual == expected

    def test_timesteps_invalid(self, setup_project_folder,
                               setup_timesteps_file_invalid):

        reader = SosModelReader(self._get_model_config(setup_project_folder))

        with raises(ValueError):
            reader.load()

# TODO test for clear error messages from reading
