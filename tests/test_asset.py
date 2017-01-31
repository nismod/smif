import pytest
from smif.asset import Asset, AssetRegister

def get_wtp():
    asset_type = 'water_treatment_plant'
    data = {
        'sector': 'water_supply',
        'capacity': {
            'units': 'ML/day',
            'value': 5
        }
    }
    return Asset(
        asset_type,
        data
    )

class TestAsset:

    def test_create_asset(self):
        water_treatment_plant = get_wtp()
        assert water_treatment_plant.asset_type == 'water_treatment_plant'

    def test_create_asset_with_full_data(self):
        asset_type = 'water_treatment_plant'
        data = {
            'sector': 'water_supply',
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
        water_treatment_plant = get_wtp()
        assert water_treatment_plant.location is None

    def test_set_location(self):
        water_treatment_plant = get_wtp()
        water_treatment_plant.location = "POINT(51.1 -1.7)"
        assert water_treatment_plant.location == "POINT(51.1 -1.7)"

    def test_no_build_date(self):
        water_treatment_plant = get_wtp()
        assert water_treatment_plant.build_date is None

    def test_set_build_date(self):
        water_treatment_plant = get_wtp()
        water_treatment_plant.build_date = 2020
        assert water_treatment_plant.build_date == 2020

    def test_get_data(self):
        water_treatment_plant = get_wtp()
        water_treatment_plant.location = "POINT(51.1 -1.7)"
        water_treatment_plant.build_date = 2020
        water_treatment_plant.data["name"] = "oxford treatment plant"

        assert water_treatment_plant.data == {
            'asset_type': 'water_treatment_plant',
            'sector': 'water_supply',
            'name': 'oxford treatment plant',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'location': "POINT(51.1 -1.7)",
            'build_date': 2020
        }

    def test_hash(self):
        water_treatment_plant = get_wtp()
        data_str = '{"asset_type": "water_treatment_plant", "capacity": ' + \
                   '{"units": "ML/day", "value": 5}, "sector": "water_supply"}'

        repr_str = 'Asset("water_treatment_plant", {"asset_type": ' + \
                   '"water_treatment_plant", "capacity": {"units": "ML/day", ' + \
                   '"value": 5}, "sector": "water_supply"})'

        # should be able to reproduce sha1sum by doing
        # `printf "data_str..." | sha1sum` on the command line
        sha1sum = "3569207430472b3c5348abffa7cfe165c89fa56e"
        assert str(water_treatment_plant) == data_str
        assert repr(water_treatment_plant) == repr_str
        assert water_treatment_plant.sha1sum() == sha1sum


class TestAssetSerialiser:
    def test_register_asset(self):
        water_treatment_plant = get_wtp()
        register = AssetRegister()
        register.register(water_treatment_plant)

        assert len(register.asset_types) == 1
        assert sorted(register.attribute_keys) == [
            "asset_type",
            "capacity",
            "sector"
        ]

        attr_idx = register.attribute_index("asset_type")
        possible = register.attribute_possible_values[attr_idx]
        assert possible == [None, "water_treatment_plant"]

        attr_idx = register.attribute_index("capacity")
        possible = register.attribute_possible_values[attr_idx]
        assert possible == [None, {'units': 'ML/day', 'value': 5}]

    def test_retrieve_asset(self):
        water_treatment_plant = get_wtp()
        register = AssetRegister()
        register.register(water_treatment_plant)

        # pick an asset from the list - this is what the optimiser will do
        numeric_asset = register.asset_types[0]

        asset = register.numeric_to_asset(numeric_asset)
        assert asset.asset_type == "water_treatment_plant"

        assert asset.data == {
            'asset_type': 'water_treatment_plant',
            'sector': 'water_supply',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            }
        }
