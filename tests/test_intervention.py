from pytest import fixture, raises
from smif.intervention import (Asset, AssetRegister, Intervention,
                               InterventionRegister)


@fixture(scope='function')
def get_wtp():
    name = 'water_treatment_plant'
    data = {
        'sector': 'water_supply',
        'capacity': {
            'units': 'ML/day',
            'value': 5
        },
        'location': 'oxford',
        'build_date': 2016
    }
    return Asset(
        name=name,
        data=data
    )


@fixture(scope='function')
def get_intervention():
    name = 'water_treatment_plant'
    data = {
        'sector': 'water_supply',
        'capacity': {
            'units': 'ML/day',
            'value': 5
        },
        'location': 'oxford'
    }
    return Intervention(
        name=name,
        data=data
    )


@fixture(scope='function')
def build_intervention_ws():
    data = {
        'sector': 'water_supply',
        'name': 'oxford treatment plant',
        'capacity': {
            'units': 'ML/day',
            'value': 450
            },
        'capital_cost': {
            'units': 'M£',
            'value': 500
        },
        'location': "POINT(51.1 -1.7)"
        }

    return data


@fixture(scope='function')
def build_intervention_es():
    data = {
        'sector': 'energy_supply',
        'name': 'London Array',
        'capacity': {
            'units': 'GW',
            'value': 2
            },
        'capital_cost': {
            'units': 'M£',
            'value': 2500
        },
        'location': "POINT(50.1 -1.4)"
        }

    return data


@fixture(scope='function')
def build_register_two(build_intervention_es, build_intervention_ws):
    water = Intervention(data=build_intervention_ws)
    energy = Intervention(data=build_intervention_es)

    register = InterventionRegister()
    register.register(energy)
    register.register(water)
    return register


class TestIntervention:

    def test_intervention_init_sector(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        assert actual.sector == 'water_supply'

    def test_intervention_init_build_date(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        with raises(AttributeError):
            assert actual.build_date

    def test_intervention_init_location(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        assert actual.location == "POINT(51.1 -1.7)"


class TestAsset:

    def test_create_asset(self, get_wtp):
        water_treatment_plant = get_wtp
        assert water_treatment_plant.name == 'water_treatment_plant'

    def test_create_asset_with_sector(self):
        data = {
            'sector': 'water_supply',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'build_date': 2020,
            'location': "POINT(51.1 -1.7)"
        }
        asset = Asset(data=data)
        assert asset.sector == 'water_supply'

    def test_create_asset_without_sector(self):
        data = {
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'build_date': 2020,
            'location': "POINT(51.1 -1.7)"
        }
        asset = Asset(data=data, sector='water_supply')
        assert asset.sector == 'water_supply'

    def test_create_asset_with_full_data(self, get_wtp):
        name = 'water_treatment_plant'
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
            name,
            data
        )
        assert wtp.location == "POINT(51.1 -1.7)"
        assert wtp.build_date == 2020

    def test_location(self, get_wtp):
        water_treatment_plant = get_wtp
        assert water_treatment_plant.location == 'oxford'

    def test_set_location(self, get_wtp):
        water_treatment_plant = get_wtp
        water_treatment_plant.location = "POINT(51.1 -1.7)"
        assert water_treatment_plant.location == "POINT(51.1 -1.7)"

    def test_build_date(self, get_wtp):
        water_treatment_plant = get_wtp
        assert water_treatment_plant.build_date == 2016

    def test_set_build_date(self, get_wtp):
        water_treatment_plant = get_wtp
        water_treatment_plant.build_date = 2020
        assert water_treatment_plant.build_date == 2020

    def test_get_data(self, get_wtp):
        water_treatment_plant = get_wtp
        water_treatment_plant.location = "POINT(51.1 -1.7)"
        water_treatment_plant.build_date = 2020
        water_treatment_plant.data["name"] = "oxford treatment plant"

        assert water_treatment_plant.data == {
            'sector': 'water_supply',
            'name': 'oxford treatment plant',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'location': "POINT(51.1 -1.7)",
            'build_date': 2020
        }

    def test_hash(self, get_wtp):
        water_treatment_plant = get_wtp

        data_str = '{"build_date": 2016, ' + \
                   '"capacity": {"units": "ML/day", "value": 5}, ' + \
                   '"location": "oxford", ' + \
                   '"name": "water_treatment_plant", ' + \
                   '"sector": "water_supply"}'

        repr_str = 'Asset("water_treatment_plant", {' + \
                   '"build_date": 2016, ' + \
                   '"capacity": {"units": "ML/day", "value": 5}, ' + \
                   '"location": "oxford", ' + \
                   '"name": "water_treatment_plant", ' + \
                   '"sector": "water_supply"})'

        # should be able to reproduce sha1sum by doing
        # `printf "data_str..." | sha1sum` on the command line
        sha1sum = "c2c6e0860718e11bc85913ef4701d629e604c41f"
        assert str(water_treatment_plant) == data_str
        assert repr(water_treatment_plant) == repr_str
        assert water_treatment_plant.sha1sum() == sha1sum


class TestAssetRegister:

    def test_fail_add_intervention(self, get_intervention):
        not_an_asset = get_intervention
        register = AssetRegister()
        with raises(TypeError):
            register.register(not_an_asset)

    def test_register_assert_length(self, get_wtp):
        water = get_wtp
        register = AssetRegister()
        register.register(water)

        assert len(register) == 1

    def test_register_assert_keys(self, get_wtp):
        water = get_wtp
        register = AssetRegister()
        register.register(water)

        assert sorted(register._attribute_keys) == [
            "build_date",
            "capacity",
            "location",
            "name",
            "sector"
        ]


class TestInterventionRegister:

    def test_register_intervention(self, get_intervention):
        water_treatment_plant = get_intervention
        register = InterventionRegister()
        register.register(water_treatment_plant)

        assert len(register) == 1
        assert sorted(register._attribute_keys) == [
            "capacity",
            "location",
            "name",
            "sector"
        ]

        attr_idx = register.attribute_index("name")
        possible = register._attribute_possible_values[attr_idx]
        assert possible == [None, "water_treatment_plant"]

        attr_idx = register.attribute_index("capacity")
        possible = register._attribute_possible_values[attr_idx]
        assert possible == [None, {'units': 'ML/day', 'value': 5}]

    def test_retrieve_intervention(self, get_intervention):
        water_treatment_plant = get_intervention
        register = InterventionRegister()
        register.register(water_treatment_plant)

        # pick an asset from the list - this is what the optimiser will do
        numeric_asset = register._numeric_keys[0]

        asset = register.numeric_to_intervention(numeric_asset)
        assert asset.name == "water_treatment_plant"

        assert asset.data == {
            'name': 'water_treatment_plant',
            'sector': 'water_supply',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'location': 'oxford'
        }

    def test_retrieve_intervention_by_name(self, get_intervention):
        water_treatment_plant = get_intervention
        register = InterventionRegister()
        register.register(water_treatment_plant)

        # pick an asset from the list - this is what the optimiser will do
        asset = register.get_intervention('water_treatment_plant')
        assert asset.name == "water_treatment_plant"

        assert asset.data == {
            'name': 'water_treatment_plant',
            'sector': 'water_supply',
            'capacity': {
                'units': 'ML/day',
                'value': 5
            },
            'location': 'oxford'
        }

    def test_error_when_retrieve_by_name(self, get_intervention):
        water_treatment_plant = get_intervention
        register = InterventionRegister()
        register.register(water_treatment_plant)
        with raises(ValueError) as excinfo:
            register.get_intervention('not_here')
        expected = "Intervention 'not_here' not found in register"
        assert str(excinfo.value) == expected

    def test_iterate_over_interventions(self, get_intervention):
        """Test __iter___ method of AssetRegister class

        """
        asset_one = get_intervention
        register = InterventionRegister()
        register.register(asset_one)

        for asset in register:
            assert asset.sha1sum() == asset_one.sha1sum()

    def test_add_duplicate_intervention(self, get_intervention):
        """Tests that only unique interventions are retained

        """
        asset_one = get_intervention
        asset_two = get_intervention
        register = InterventionRegister()
        register.register(asset_one)
        register.register(asset_two)

        assert len(register) == 1

    def test_register_len_one(self, build_intervention_ws):
        water = Intervention(data=build_intervention_ws)
        register = InterventionRegister()
        register.register(water)
        assert len(register) == 1

    def test_register_len_two(self, build_register_two):
        register = build_register_two
        assert len(register) == 2
