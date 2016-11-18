from smif.runner import ModelRunner


class TestAssetLoad:

    def test_assets_load(self):
        attributes = {'water_asset_a': {},
                      'water_asset_b': {},
                      'water_asset_c': {}}
        model = ModelRunner('water_supply', attributes)
        assert model.assets == ['water_asset_a',
                                'water_asset_b',
                                'water_asset_c']


class TestAttributesLoad:

    def test_attributes_load(self):

        attributes = \
            {'water_asset_a': {'capital_cost': {'value': 1000,
                                                'unit': '£/kW'},
                               'economic_lifetime': 25,
                               'operational_lifetime': 25
                               },
             'water_asset_b': {'capital_cost': {'value': 1500,
                                                'unit': '£/kW'}},
             'water_asset_c': {'capital_cost': {'value': 3000,
                                                'unit': '£/kW'}}}
        model = ModelRunner('water_supply', attributes)
        actual = model.attributes
        expected = \
            {'water_asset_a': {'capital_cost': {'value': 1000,
                                                'unit': '£/kW'},
                               'economic_lifetime': 25,
                               'operational_lifetime': 25
                               },
             'water_asset_b': {'capital_cost': {'value': 1500,
                                                'unit': '£/kW'}},
             'water_asset_c': {'capital_cost': {'value': 3000,
                                                'unit': '£/kW'}}}

        assert actual == expected

        for asset in model.assets:
            assert asset in expected.keys()
