# -*- coding: utf-8 -*-
from smif.sectormodel import SectorModel

class EmptySectorModel(SectorModel):
    def simulate(self, static_inputs, decision_variables):
        pass

    def extract_obj(self, results):
        return 0

class TestAssetLoad(object):

    def test_assets_load(self):
        attributes = {'water_asset_a': {},
                      'water_asset_b': {},
                      'water_asset_c': {}}
        model = EmptySectorModel()
        model.attributes = attributes
        assert model.assets == ['water_asset_a',
                                'water_asset_b',
                                'water_asset_c']


class TestAttributesLoad(object):

    def test_attributes_load(self):

        attributes = {
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
        model.attributes = attributes
        actual = model.attributes
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

        for asset in model.assets:
            assert asset in expected.keys()
