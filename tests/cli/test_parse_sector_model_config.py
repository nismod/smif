# -*- coding: utf-8 -*-
import os
from pytest import raises
from smif.cli.parse_sector_model_config import SectorModelReader


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
        return SectorModelReader(model_name, model_path, model_classname,
                                 config_dir)

    def test_load(self, setup_project_folder):
        reader = self._reader(setup_project_folder)
        reader.load()

        # could check actual data from conftest.py:setup_water_inputs
        assert reader.data["inputs"] is not None
        assert reader.data["outputs"] is not None
        assert reader.data["time_intervals"] is not None
        assert reader.data["regions"] is not None

    def test_load_errors(self, setup_project_missing_model_config):
        reader = self._reader(setup_project_missing_model_config)

        with raises(FileNotFoundError) as ex:
            reader._load_inputs()
        msg = "inputs config file not found for water_supply model"
        assert msg in str(ex.value)

        with raises(FileNotFoundError) as ex:
            reader._load_outputs()
        msg = "outputs config file not found for water_supply model"
        assert msg in str(ex.value)

        with raises(FileNotFoundError) as ex:
            reader._load_time_intervals()
        msg = "time_intervals config file not found for water_supply model"
        assert msg in str(ex.value)

        with raises(FileNotFoundError) as ex:
            reader._load_regions()
        msg = "regions config file not found for water_supply model"
        assert msg in str(ex.value)
