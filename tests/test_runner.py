# -*- coding: utf-8 -*-
from smif.sector_model import SectorModel

class EmptySectorModel(SectorModel):
    def simulate(self, static_inputs, decision_variables):
        pass

    def extract_obj(self, results):
        return 0


def test_assets_load(self):
    assets = {
        'water_asset_a': {},
        'water_asset_b': {},
        'water_asset_c': {}
    }
    model = EmptySectorModel()
    model.assets = assets

    asset_names = model.asset_names

    assert len(asset_names) == 3
    assert 'water_asset_a' in asset_names
    assert 'water_asset_b' in asset_names
    assert 'water_asset_c' in asset_names


def test_attributes_load(self):

    assets = {
        'water_asset_a': {
            'capital_cost': {
                'value': 1000,
                'unit': '£/kW'
            },
            'economic_lifetime': 25,
            'operational_lifetime': 25
        },
        'water_asset_b': {
            'capital_cost': {
                'value': 1500,
                'unit': '£/kW'
            }
        },
        'water_asset_c': {
            'capital_cost': {
                'value': 3000,
                'unit': '£/kW'
            }
        }
    }
    model = EmptySectorModel()
    model.assets = assets
    actual = model.assets
    expected = {
        'water_asset_a': {
            'capital_cost': {
                'value': 1000,
                'unit': '£/kW'
            },
            'economic_lifetime': 25,
            'operational_lifetime': 25
        },
        'water_asset_b': {
            'capital_cost': {
                'value': 1500,
                'unit': '£/kW'
            }
        },
        'water_asset_c': {
            'capital_cost': {
                'value': 3000,
                'unit': '£/kW'
            }
        }
    }

    assert actual == expected

    for asset in model.asset_names:
        assert asset in expected.keys()
