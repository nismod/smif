from smif.cli.parse_sector_model_config import SectorModelReader

class TestSectorModelReader(object):

    def test_assets(self, setup_project_folder):

        reader = SectorModelReader(
            'water_supply',
            str(setup_project_folder),
            'WaterSupplySectorModel')

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = reader.assets
        assert actual == expected

    def test_assets_two_asset_files(self, setup_project_folder,
                                    setup_assets_file_two,
                                    setup_config_file_two,
                                    setup_water_asset_d):

        reader = SectorModelReader(
            'water_supply',
            str(setup_project_folder),
            'WaterSupplySectorModel')

        expected = ['water_asset_a', 'water_asset_b',
                    'water_asset_c', 'water_asset_d']
        actual = reader.assets
        assert actual == expected
