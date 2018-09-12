# -*- coding: utf-8 -*-

from copy import copy
from unittest.mock import Mock

from pytest import fixture, raises
from smif.controller.modelrun import ModelRunner
from smif.metadata import Spec
from smif.model.dependency import Dependency
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel
from smif.model.sos_model import SosModel


class EmptySectorModel(SectorModel):
    """Simulate nothing
    """
    def simulate(self, data):
        return data


@fixture(scope='function')
def empty_sector_model():
    """SectorModel ready for customisation
    """
    return EmptySectorModel('test')


@fixture(scope='function')
def scenario_model():
    """ScenarioModel providing precipitation
    """
    model = ScenarioModel('climate')
    model.add_output(
        Spec.from_dict({
            'name': 'precipitation',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'ml'
        })
    )
    model.scenario = 'UKCP09 High emissions'
    return model


@fixture(scope='function')
def sector_model():
    """SectorModel requiring precipitation and cost, providing water
    """
    model = EmptySectorModel('water_supply')
    model.add_input(
        Spec.from_dict({
            'name': 'precipitation',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'ml'
        })
    )
    model.add_input(
        Spec.from_dict({
            'name': 'rGVA',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'million GBP'
        })
    )
    model.add_output(
        Spec.from_dict({
            'name': 'water',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'Ml'
        })
    )
    return model


@fixture(scope='function')
def economic_model():
    """SectorModel requiring precipitation and cost, providing water
    """
    model = EmptySectorModel('economic_model')
    model.add_output(
        Spec.from_dict({
            'name': 'gva',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'million GBP'
        })
    )
    return model


@fixture(scope='function')
def sos_model(sector_model, scenario_model, economic_model):
    """SosModel with one scenario and one sector model
    """
    sos_model = SosModel('test_sos_model')
    sos_model.add_model(scenario_model)
    sos_model.add_model(economic_model)
    sos_model.add_model(sector_model)
    sector_model.add_dependency(
        scenario_model, 'precipitation', 'precipitation')
    sector_model.add_dependency(
        economic_model, 'gva', 'rGVA')
    return sos_model


@fixture(scope='function')
def scenario_only_sos_model_dict():
    """Config for a SosModel with one scenario
    """
    return {
        'name': 'test_sos_model',
        'description': 'Readable description of the sos model',
        'scenarios': [
            {
                'name': 'climate',
                'scenario': 'UKCP09 High emissions',
                'outputs': [
                    {
                        'name': 'precipitation',
                        'dims': ['LSOA'],
                        'coords': {'LSOA': [1, 2, 3]},
                        'dtype': 'float',
                        'unit': 'ml'
                    }
                ]
            }
        ],
        'sector_models': [],
        'dependencies': []
    }


@fixture(scope='function')
def sos_model_dict(scenario_only_sos_model_dict):
    """Config for a SosModel with one scenario and one sector model
    """
    config = scenario_only_sos_model_dict
    config['sector_models'] = [
        {
            'name': 'economic_model',
            'inputs': [],
            'parameters': [],
            'outputs': [
                {
                    'name': 'gva',
                    'dims': ['LSOA'],
                    'coords': {'LSOA': [1, 2, 3]},
                    'dtype': 'float',
                    'unit': 'million GBP'
                }
            ]
        },
        {
            'name': 'water_supply',
            'inputs': [
                {
                    'name': 'precipitation',
                    'dims': ['LSOA'],
                    'coords': {'LSOA': [1, 2, 3]},
                    'dtype': 'float',
                    'unit': 'ml'
                },
                {
                    'name': 'rGVA',
                    'dims': ['LSOA'],
                    'coords': {'LSOA': [1, 2, 3]},
                    'dtype': 'float',
                    'unit': 'million GBP'
                }
            ],
            'parameters': [],
            'outputs': [
                {
                    'name': 'water',
                    'dims': ['LSOA'],
                    'coords': {'LSOA': [1, 2, 3]},
                    'dtype': 'float',
                    'unit': 'Ml'
                }
            ]
        }
    ]
    config['scenario_dependencies'] = [
        {
            'source': 'climate',
            'source_output': 'precipitation',
            'sink_input': 'precipitation',
            'sink': 'water_supply'
        }
    ]
    config['model_dependencies'] = [
        {
            'source': 'economic_model',
            'source_output': 'gva',
            'sink_input': 'rGVA',
            'sink': 'water_supply'
        }
    ]
    return config


class TestSosModel():
    """Construct from config or compose from objects
    """
    def test_construct(self, sos_model_dict, scenario_model, sector_model, economic_model):
        """Constructing from config of the form::

            {
                'name': 'sos_model_name',
                'description': 'friendly description of the sos model',
                'dependencies': [
                    {
                        'source': str (Model.name),
                        'source_output': str (Metadata.name),
                        'sink': str (Model.name),
                        'sink_output': str (Metadata.name)
                    }
                ]
            }

        With list of child SectorModel/ScenarioModel instances passed in alongside.
        """
        sos_model = SosModel.from_dict(
            sos_model_dict, [scenario_model, sector_model, economic_model])

        assert isinstance(sos_model, SosModel)
        assert list(sos_model.scenario_models.keys()) == ['climate']
        assert isinstance(sos_model.models['climate'], ScenarioModel)
        assert sorted(list(sos_model.sector_models.keys())) == \
            ['economic_model', 'water_supply']
        assert isinstance(sos_model.models['economic_model'], SectorModel)
        assert isinstance(sos_model.models['water_supply'], SectorModel)

    def test_optional_description(self, sos_model_dict):
        """Default to empty description
        """
        del sos_model_dict['description']
        sos_model = SosModel.from_dict(sos_model_dict)
        assert sos_model.description == ''

    def test_dependencies_fields(self, sos_model_dict, scenario_model, sector_model,
                                 economic_model):
        """Compose dependencies from scenario- and model- fields
        """
        a = {
            'source': 'climate',
            'source_output': 'precipitation',
            'sink_input': 'precipitation',
            'sink': 'water_supply'
        }
        b = {
            'source': 'economic_model',
            'source_output': 'gva',
            'sink_input': 'rGVA',
            'sink': 'water_supply'
        }

        del sos_model_dict['dependencies']
        sos_model_dict['scenario_dependencies'] = [a]
        sos_model_dict['model_dependencies'] = [b]
        sos_model = SosModel.from_dict(
            sos_model_dict, [scenario_model, sector_model, economic_model])
        actual = sos_model.as_dict()
        assert actual['scenario_dependencies'] == [a]
        assert actual['model_dependencies'] == [b]

    def test_dependencies_fields_alt(self, sos_model_dict, scenario_model, sector_model,
                                     economic_model):
        """Draw dependencies from single field
        """
        a = {
            'source': 'climate',
            'source_output': 'precipitation',
            'sink_input': 'precipitation',
            'sink': 'water_supply'
        }
        b = {
            'source': 'economic_model',
            'source_output': 'gva',
            'sink_input': 'rGVA',
            'sink': 'water_supply'
        }
        sos_model_dict['dependencies'] = [a, b]
        del sos_model_dict['scenario_dependencies']
        del sos_model_dict['model_dependencies']
        sos_model = SosModel.from_dict(
            sos_model_dict, [scenario_model, sector_model, economic_model])
        actual = sos_model.as_dict()
        assert actual['scenario_dependencies'] == [a]
        assert actual['model_dependencies'] == [b]

    def test_compose(self, sos_model):
        with raises(NotImplementedError) as ex:
            sos_model.add_model(SosModel('test'))
        assert "Nesting of CompositeModels (including SosModels) is not supported" in str(ex)

    def test_as_dict(self, sos_model):
        """as_dict correctly returns configuration as a dictionary, with child models as_dict
        similarly
        """
        actual = sos_model.as_dict()
        expected = {
            'name': 'test_sos_model',
            'description': '',
            'scenarios': ['climate'],
            'sector_models': ['economic_model', 'water_supply'],
            'scenario_dependencies': [{
                'source': 'climate',
                'source_output': 'precipitation',
                'sink': 'water_supply',
                'sink_input': 'precipitation'
            }],
            'model_dependencies': [{
                'source': 'economic_model',
                'source_output': 'gva',
                'sink': 'water_supply',
                'sink_input': 'rGVA'
            }]
        }
        assert actual == expected

    def test_add_dependency(self, empty_sector_model):
        """Add models, connect via dependency
        """
        spec = Spec(
            name='hourly_value',
            dims=['hours'],
            coords={'hours': range(24)},
            dtype='int'
        )
        sink_model = copy(empty_sector_model)
        sink_model.add_input(spec)

        source_model = copy(empty_sector_model)
        source_model.add_output(spec)

        sink_model.add_dependency(source_model, 'hourly_value', 'hourly_value')

        sos_model = SosModel('test')
        sos_model.add_model(source_model)
        sos_model.add_model(sink_model)

    def test_before_model_run_fails(self, sos_model):
        """Before model run should raise
        """
        data_handle = Mock()
        with raises(NotImplementedError):
            sos_model.before_model_run(data_handle)

    def test_simulate_fails(self, sos_model):
        """Simulate should raise
        """
        data_handle = Mock()
        with raises(NotImplementedError):
            sos_model.simulate(data_handle)


class TestSosModelProperties():
    """SosModel has inputs, outputs, parameters

    Convergence settings (should perhaps move to runner)
    """

    def test_model_inputs(self, sos_model):
        spec = sos_model.models['water_supply'].inputs['precipitation']
        assert isinstance(spec, Spec)

    def test_model_outputs(self, sos_model):
        spec = sos_model.models['water_supply'].outputs['water']
        assert isinstance(spec, Spec)

    def test_run_with_global_parameters(self, sos_model):
        sos_model = sos_model
        sos_model.add_parameter(Spec.from_dict({
            'name': 'sos_model_param',
            'description': 'A global parameter passed to all contained models',
            'abs_range': (0, 100),
            'exp_range': (3, 10),
            'default': 3,
            'dtype': 'float',
            'unit': '%'
        }))
        assert 'sos_model_param' in sos_model.parameters

    def test_run_with_sector_parameters(self, sos_model):
        sos_model = sos_model
        sector_model = sos_model.models['water_supply']
        sector_model.add_parameter(Spec.from_dict({
            'name': 'sector_model_param',
            'description': 'A model parameter passed to a specific model',
            'abs_range': (0, 100),
            'exp_range': (3, 10),
            'dtype': 'float',
            'default': 3,
            'unit': '%'
        }))
        assert 'sector_model_param' in sector_model.parameters

    def test_add_parameters(self, sos_model, sector_model):
        expected = Spec.from_dict({
            'name': 'sos_model_param',
            'description': 'A global parameter passed to all contained models',
            'abs_range': (0, 100),
            'exp_range': (3, 10),
            'dtype': 'float',
            'default': 3,
            'unit': '%'
        })
        sos_model.add_parameter(expected)

        assert sos_model.parameters['sos_model_param'] == expected
        assert list(sos_model.parameters.keys()) == ['sos_model_param']

        sector_model.add_parameter(Spec.from_dict({
            'name': 'sector_model_param',
            'description': 'Required for the sectormodel',
            'abs_range': (0, 100),
            'exp_range': (3, 10),
            'default': 3,
            'dtype': 'float',
            'unit': '%'
        }))
        sos_model.add_model(sector_model)

        # SosModel contains only its own parameters
        assert 'sos_model_param' in sos_model.parameters.keys()

        # SectorModel has its own ParameterList, gettable by param name
        assert 'sector_model_param' in sector_model.parameters.keys()


class TestSosModelDependencies(object):
    """SosModel can represent data flow as defined by model dependencies
    """
    def test_simple_dependency(self, sos_model):
        """Dependency graph construction
        """
        graph = ModelRunner.get_dependency_graph(sos_model.models)

        scenario = sos_model.models['climate']
        model = sos_model.models['water_supply']

        assert scenario in [node[1]['model'] for node in graph.nodes.data()]
        assert model in [node[1]['model'] for node in graph.nodes.data()]

        actual = sos_model.models['water_supply'].deps['precipitation']
        expected = Dependency(
            scenario,
            scenario.outputs['precipitation'],
            model,
            model.inputs['precipitation']
        )
        assert actual == expected

    def test_data_not_present(self, sos_model_dict, sector_model):
        """Raise a NotImplementedError if an input is defined but no dependency links
        it to a data source
        """
        sos_model_dict['scenario_dependencies'] = []
        sos_model_dict['model_dependencies'] = []
        with raises(NotImplementedError):
            SosModel.from_dict(sos_model_dict, [sector_model])

    def test_undefined_unit_conversion(self, sos_model_dict, sector_model, scenario_model,
                                       economic_model):
        """Error on invalid dependency
        """
        sector_model.inputs['precipitation'] = Spec(
            name='precipitation',
            dims=['LSOA'],
            coords={'LSOA': [1, 2, 3]},
            dtype='float',
            unit='incompatible'
        )

        with raises(ValueError) as ex:
            SosModel.from_dict(sos_model_dict, [sector_model, scenario_model, economic_model])
        assert "ml!=incompatible" in str(ex)

    def test_invalid_unit_conversion(self, sos_model_dict, sector_model, scenario_model,
                                     economic_model):
        """Error on invalid dependency
        """
        scenario_model.outputs['precipitation'] = Spec(
            name='precipitation',
            dims=['LSOA'],
            coords={'LSOA': [1, 2, 3]},
            dtype='float',
            unit='meter'
        )

        with raises(ValueError) as ex:
            SosModel.from_dict(sos_model_dict, [sector_model, scenario_model, economic_model])
        assert "meter!=ml" in str(ex.value)
