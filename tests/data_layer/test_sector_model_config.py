# -*- coding: utf-8 -*-
import os
from unittest.mock import MagicMock

from smif.data_layer.sector_model_config import SectorModelReader


class TestSectorModelReader(object):
    def _model_config_dir(self, project_folder):
        return os.path.join(str(project_folder), 'data', 'water_supply')

    def _model_path(self, project_folder):
        return os.path.join(str(project_folder), 'models', 'water_supply', 'water_supply.py')

    def _reader(self, project_folder):
        model_name = 'water_supply'
        model_path = self._model_path(project_folder)
        model_classname = ''
        config_dir = self._model_config_dir(project_folder)
        return SectorModelReader({
            "model_name": model_name,
            "model_path": model_path,
            "model_classname": model_classname,
            "model_config_dir": config_dir,
            "initial_conditions": [
                os.path.join(config_dir, "initial_conditions/assets_1.yaml")
            ],
            "interventions": [
                os.path.join(config_dir, "interventions/water_asset_abc.yaml")
            ]
        })

    def test_reader_without_initial_config(self):
        reader = SectorModelReader()
        assert reader.model_name is None
        reader.model_name = "test_model"
        assert reader.model_name == "test_model"

    def test_load(self, setup_project_folder):
        reader = self._reader(setup_project_folder)
        reader.load()

        # could check actual data from conftest.py:setup_water_inputs
        assert reader.data["inputs"] is not None
        assert reader.data["outputs"] is not None

    def test_load_errors(self, setup_project_missing_model_config):
        reader = self._reader(setup_project_missing_model_config)
        reader.logger.warning = MagicMock()

        # expect a warning if loading no inputs
        inputs = reader.load_inputs()
        args, kwargs = reader.logger.warning.call_args

        assert args[0] == "No %s provided for '%s' model: %s not found"
        assert args[1] == "inputs"
        assert args[2] == "water_supply"
        assert "inputs.yaml" in args[3]
        assert inputs == []

        outputs = reader.load_outputs()
        args, kwargs = reader.logger.warning.call_args

        assert args[0] == "No %s provided for '%s' model: %s not found"
        assert args[1] == "outputs"
        assert args[2] == "water_supply"
        assert "outputs.yaml" in args[3]
        assert outputs == []

    def test_load_empty(self, setup_project_empty_model_io):
        reader = self._reader(setup_project_empty_model_io)
        reader.logger.warning = MagicMock()

        # expect a warning if loading no inputs
        inputs = reader.load_inputs()
        args, kwargs = reader.logger.warning.call_args

        assert args[0] == "No %s provided for '%s' model: %s was empty"
        assert args[1] == "inputs"
        assert args[2] == "water_supply"
        assert "inputs.yaml" in args[3]
        assert inputs == []

        outputs = reader.load_outputs()
        args, kwargs = reader.logger.warning.call_args

        assert args[0] == "No %s provided for '%s' model: %s was empty"
        assert args[1] == "outputs"
        assert args[2] == "water_supply"
        assert "outputs.yaml" in args[3]
        assert outputs == []

    def test_load_interventions(self, setup_project_folder):

        reader = self._reader(setup_project_folder)
        reader.load()

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = [asset['name'] for asset in reader.interventions]
        assert actual == expected

    def test_load_interventions_two_files(self, setup_project_folder,
                                          setup_config_file_two,
                                          setup_water_intervention_d):

        model_name = 'water_supply'
        model_path = self._model_path(setup_project_folder)
        model_classname = ''
        config_dir = self._model_config_dir(setup_project_folder)
        reader = SectorModelReader({
            "model_name": model_name,
            "model_path": model_path,
            "model_classname": model_classname,
            "model_config_dir": config_dir,
            "initial_conditions": [
                os.path.join(config_dir, "initial_conditions/assets_1.yaml")
            ],
            "interventions": [
                os.path.join(config_dir, "interventions/water_asset_abc.yaml"),
                os.path.join(config_dir, "interventions/water_asset_d.yaml")
            ]
        })

        reader.load()

        expected = ['water_asset_a', 'water_asset_b',
                    'water_asset_c', 'water_asset_d']
        actual = [asset['name'] for asset in reader.interventions]
        assert actual == expected
