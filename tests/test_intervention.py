from pytest import fixture, raises
from smif.intervention import Intervention, InterventionRegister


@fixture(scope='function')
def get_intervention_plant():
    name = 'water_treatment_plant'
    data = {
        'capacity': {
            'units': 'ML/day',
            'value': 5
        },
        'location': 'oxford'
    }
    return Intervention(
        name=name,
        data=data,
        sector='water_supply'
    )


@fixture(scope='function')
def get_intervention_power():
    name = 'power_plant'
    data = {
        'capacity': {
            'units': 'GW',
            'value': 1
        },
        'location': 'oxford'
    }
    return Intervention(
        name=name,
        data=data,
        sector='energy_supply'
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

    def test_serialisation(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws).as_dict()

        expected = {
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

        assert actual == expected

    def test_populate_with_arguments(self):
        actual = Intervention(name='a name', data=None, sector='a sector',
                              location='a location')
        assert actual.data == {'name': 'a name', 'sector': 'a sector',
                               'location': 'a location'}

    def test_validation(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        actual.data = {'required_attribute': 'test_name',
                       'missing_attribute': 'missing'}
        with raises(ValueError):
            actual._validate(['required_attribute'], ['missing_attribute'])

    def test_validation_expected(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        actual.data = {'unrequired_attribute': 'test_name'}
        with raises(ValueError):
            actual._validate(['required_attribute'], ['missing_attribute'])

    def test_validation_omitted(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        actual.data = {'required_attribute': 'test_name'}
        assert actual._validate(['required_attribute'], ['missing_attribute'])

    def test_set_location_attribute(self, build_intervention_ws):
        actual = Intervention(data=build_intervention_ws)
        actual.location = 'blobby'
        assert actual.location == 'blobby'
        assert actual.data['location'] == 'blobby'


class TestInterventionRegister:

    def test_register_intervention(self, get_intervention_plant):
        water_treatment_plant = get_intervention_plant
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

    def test_register_multiple_interventions(self,
                                             get_intervention_plant,
                                             get_intervention_power):
        register = InterventionRegister()
        register.register(get_intervention_plant)
        register.register(get_intervention_power)
        attr_idx = register.attribute_index("capacity")
        possible = register._attribute_possible_values[attr_idx]
        assert possible == [None, {'units': 'ML/day', 'value': 5},
                                  {'units': 'GW', 'value': 1}]

    def test_raise_error_on_register_non_intervention(self):
        register = InterventionRegister()
        with raises(TypeError):
            register.register(['not', 'an', 'intervention'])

    def test_retrieve_intervention(self, get_intervention_plant):
        water_treatment_plant = get_intervention_plant
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

    def test_retrieve_intervention_by_name(self, get_intervention_plant):
        water_treatment_plant = get_intervention_plant
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

    def test_error_when_retrieve_by_name(self, get_intervention_plant):
        water_treatment_plant = get_intervention_plant
        register = InterventionRegister()
        register.register(water_treatment_plant)
        with raises(ValueError) as excinfo:
            register.get_intervention('not_here')
        expected = "Intervention 'not_here' not found in register"
        assert str(excinfo.value) == expected

    def test_iterate_over_interventions(self, get_intervention_plant):
        """Test __iter___ method of AssetRegister class

        """
        asset_one = get_intervention_plant
        register = InterventionRegister()
        register.register(asset_one)

        for asset in register:
            assert asset.sha1sum() == asset_one.sha1sum()

    def test_adding_duplicate_intervention_raises_warning(self, get_intervention_plant):
        """Tests that adding a duplicate intervention raises a warning

        """
        asset_one = get_intervention_plant
        asset_two = get_intervention_plant
        register = InterventionRegister()
        register.register(asset_one)

        with raises(ValueError) as excinfo:
            register.register(asset_two)

        msg = "Attempted registering of duplicate intervention: " + \
              "'water_treatment_plant' for 'water_supply'"
        assert str(excinfo.value) == msg

        assert len(register) == 1

    def test_register_len_one(self, build_intervention_ws):
        water = Intervention(data=build_intervention_ws)
        register = InterventionRegister()
        register.register(water)
        assert len(register) == 1

    def test_register_len_two(self, build_register_two):
        register = build_register_two
        assert len(register) == 2
