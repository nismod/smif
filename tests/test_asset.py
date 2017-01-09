import pytest
from smif.asset import Asset

class TestAsset:
    def get_wtp(self):
        asset_type = 'water_treatment_plant'
        data = {
            'capacity': {
                'units': 'ML/day',
                'value': 5
            }
        }
        return Asset(
            asset_type,
            data
        )

    def test_create_asset(self):
        water_treatment_plant = self.get_wtp()
        assert water_treatment_plant.asset_type == 'water_treatment_plant'

    def test_create_asset_with_full_data(self):
        asset_type = 'water_treatment_plant'
        data = {
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'build_date': 2020,
            'location': "POINT(51.1 -1.7)"
        }
        wtp = Asset(
            asset_type,
            data
        )
        assert wtp.location == "POINT(51.1 -1.7)"
        assert wtp.build_date == 2020

    def test_no_location(self):
        water_treatment_plant = self.get_wtp()
        assert water_treatment_plant.location is None

    def test_set_location(self):
        water_treatment_plant = self.get_wtp()
        water_treatment_plant.location = "POINT(51.1 -1.7)"
        assert water_treatment_plant.location == "POINT(51.1 -1.7)"

    def test_no_build_date(self):
        water_treatment_plant = self.get_wtp()
        assert water_treatment_plant.build_date is None

    def test_set_build_date(self):
        water_treatment_plant = self.get_wtp()
        water_treatment_plant.build_date = 2020
        assert water_treatment_plant.build_date == 2020

    def test_get_data(self):
        water_treatment_plant = self.get_wtp()
        water_treatment_plant.location = "POINT(51.1 -1.7)"
        water_treatment_plant.build_date = 2020
        water_treatment_plant.data["name"] = "oxford treatment plant"

        assert water_treatment_plant.data == {
            'name': 'oxford treatment plant',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'location': "POINT(51.1 -1.7)",
            'build_date': 2020
        }
