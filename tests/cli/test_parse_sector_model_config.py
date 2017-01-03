# -*- coding: utf-8 -*-
import os
from smif.cli.parse_sector_model_config import SectorModelReader

class TestSectorModelReader(object):
    def _model_config_dir(self, project_folder):
        return os.path.join(str(project_folder), 'data', 'water_supply')

    def test_assets(self, setup_project_folder):

        reader = SectorModelReader(
            'water_supply',
            'path/to/water_supply.py',
            'WaterSupplySectorModel',
            self._model_config_dir(setup_project_folder)
        )
        reader.load()

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = [asset['name'] for asset in reader.asset_types]
        assert actual == expected

    def test_assets_two_asset_files(self, setup_project_folder,
                                    setup_config_file_two,
                                    setup_water_asset_d):

        reader = SectorModelReader(
            'water_supply',
            'path/to/water_supply.py',
            'WaterSupplySectorModel',
            self._model_config_dir(setup_project_folder)
        )

        reader.load()

        expected = ['water_asset_a', 'water_asset_b',
                    'water_asset_c', 'water_asset_d']
        actual = [asset['name'] for asset in reader.asset_types]
        assert actual == expected
