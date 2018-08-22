# -*- coding: utf-8 -*-

from copy import copy
from unittest.mock import Mock

from pytest import fixture, raises
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
    scenario_model = ScenarioModel('climate')
    scenario_model.add_output(
        Spec.from_dict({
            'name': 'precipitation',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'ml'
        })
    )
    scenario_model.scenario = 'UKCP09 High emissions'
    return scenario_model


@fixture(scope='function')
def sector_model():
    """SectorModel requiring precipitation and cost, providing water
    """
    sector_model = EmptySectorModel('water_supply')
    sector_model.add_input(
        Spec.from_dict({
            'name': 'precipitation',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'ml'
        })
    )
    sector_model.add_output(
        Spec.from_dict({
            'name': 'cost',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'million GBP'
        })
    )
    sector_model.add_output(
        Spec.from_dict({
            'name': 'water',
            'dims': ['LSOA'],
            'coords': {'LSOA': [1, 2, 3]},
            'dtype': 'float',
            'unit': 'Ml'
        })
    )
    return sector_model


@fixture(scope='function')
def sos_model(sector_model, scenario_model):
    """SosModel with one scenario and one sector model
    """
    sos_model = SosModel('test_sos_model')
    sos_model.add_model(sector_model)
    sos_model.add_model(scenario_model)
    sector_model.add_dependency(
        scenario_model, 'precipitation', 'precipitation')
    return sos_model


@fixture(scope='function')
def scenario_only_sos_model_dict():
    """Config for a SosModel with one scenario
    """
    return {
        'name': 'energy_sos_model',
        'description': 'Readable description of the sos model',
        # 'scenario_sets': [
        #     {
        #         'name': 'climate',
        #         'scenario': 'UKCP09 High emissions',
        #         'outputs': [
        #             {
        #                 'name': 'precipitation',
        #                 'dims': ['LSOA'],
        #                 'coords': {'LSOA': [1, 2, 3]},
        #                 'dtype': 'float',
        #                 'unit': 'ml'
        #             }
        #         ]
        #     }
        # ],
        # 'sector_models': [],
        'dependencies': []
    }


@fixture(scope='function')
def sos_model_dict(scenario_only_sos_model_dict):
    """Config for a SosModel with one scenario and one sector model
    """
    sos_model_dict = scenario_only_sos_model_dict
    # sos_model_dict['sector_models'] = [
    #     {
    #         'name': 'water_supply',
    #         'inputs': [
    #             {
    #                 'name': 'precipitation',
    #                 'dims': ['LSOA'],
    #                 'coords': {'LSOA': [1, 2, 3]},
    #                 'dtype': 'float',
    #                 'unit': 'ml'
    #             },
    #             {
    #                 'name': 'cost',
    #                 'dims': ['LSOA'],
    #                 'coords': {'LSOA': [1, 2, 3]},
    #                 'dtype': 'float',
    #                 'unit': 'million GBP'
    #             }
    #         ],
    #         'parameters': [],
    #         'outputs': [
    #             {
    #                 'name': 'water',
    #                 'dims': ['LSOA'],
    #                 'coords': {'LSOA': [1, 2, 3]},
    #                 'dtype': 'float',
    #                 'unit': 'Ml'
    #             }
    #         ]
    #     }
    # ]
    sos_model_dict['dependencies'] = [
        {
            'source': 'climate',
            'source_output': 'precipitation',
            'sink_input': 'precipitation',
            'sink': 'water_supply'
        }
    ]
    return sos_model_dict


class TestSosModel():
    """Construct from config or compose from objects
    """

    def test_construct(self, sos_model_dict, scenario_model, sector_model):
        """Constructing from config of the form::

            {
                'name': 'sos_model_name',
                'description': 'friendly description of the sos model',
                'max_iterations': int,
                'convergence_absolute_tolerance': float,
                'convergence_relative_tolerance': float,
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
        sos_model = SosModel.from_dict(sos_model_dict, [scenario_model, sector_model])

        assert isinstance(sos_model, SosModel)
        assert list(sos_model.scenario_models.keys()) == ['climate']
        assert isinstance(sos_model.models['climate'], ScenarioModel)
        assert list(sos_model.sector_models.keys()) == ['water_supply']
        assert isinstance(sos_model.models['water_supply'], SectorModel)

    def test_as_dict(self, sos_model, scenario_model, sector_model):
        """as_dict correctly returns configuration as a dictionary, with child models as_dict
        similarly
        """
        sos_model = sos_model
        actual = sos_model.as_dict()
        expected = {
            'name': 'test_sos_model',
            'description': '',
            'scenario_sets': ['climate'],
            'sector_models': ['water_supply'],
            'dependencies': [{
                'source': 'climate',
                'source_output': 'precipitation',
                'sink': 'water_supply',
                'sink_input': 'precipitation'
            }],
            'max_iterations': 25,
            'convergence_absolute_tolerance': 1e-8,
            'convergence_relative_tolerance': 1e-5
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

    def test_run_sequential(self, sos_model):
        """Simulate should exist
        """
        sos_model = sos_model
        data_handle = Mock()
        data_handle.timesteps = [2010, 2011, 2012]
        data_handle.get_state = Mock(return_value={})

        data_handle._current_timestep = 2010
        sos_model.simulate(data_handle)
        data_handle._current_timestep = 2011
        sos_model.simulate(data_handle)
        data_handle._current_timestep = 2012
        sos_model.simulate(data_handle)


class TestSosModelProperties():
    """SosModel has inputs, outputs, parameters

    Convergence settings (should perhaps move to runner)
    """

    def test_model_inputs(self, sos_model):
        spec = sos_model.models['water_supply'].inputs['precipitation']
        assert isinstance(spec, Spec)

    def test_model_outputs(self, sos_model):
        spec = sos_model.models['water_supply'].outputs['cost']
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

    def test_set_max_iterations(self, sos_model_dict):
        """Test constructing from single dict config
        """
        sos_model_dict['max_iterations'] = 125
        sos_model = SosModel.from_dict(sos_model_dict)
        assert sos_model.max_iterations == 125

    def test_set_convergence_absolute_tolerance(self, sos_model_dict):
        """Test constructing from single dict config
        """
        sos_model_dict['convergence_absolute_tolerance'] = 0.0001
        sos_model = SosModel.from_dict(sos_model_dict)
        assert sos_model.convergence_absolute_tolerance == 0.0001

    def test_set_convergence_relative_tolerance(self, sos_model_dict):
        """Test constructing from single dict config
        """
        sos_model_dict['convergence_relative_tolerance'] = 0.1
        sos_model = SosModel.from_dict(sos_model_dict)
        assert sos_model.convergence_relative_tolerance == 0.1


class TestSosModelDependencies(object):
    """SosModel can represent data flow as defined by model dependencies
    """

    def test_simple_dependency(self, sos_model_dict, sector_model, scenario_model):
        """Dependency graph construction
        """
        sos_model = SosModel.from_dict(sos_model_dict, [sector_model, scenario_model])

        graph = SosModel.make_dependency_graph(sos_model.models)

        scenario = sos_model.models['climate']
        model = sos_model.models['water_supply']

        assert scenario in graph.nodes()
        assert model in graph.nodes()

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
        sos_model_dict['dependencies'] = []
        with raises(NotImplementedError):
            SosModel.from_dict(sos_model_dict, [sector_model])

    def test_undefined_unit_conversion(self, sos_model_dict, sector_model, scenario_model):
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
            SosModel.from_dict(sos_model_dict, [sector_model, scenario_model])
        assert "ml!=incompatible" in str(ex)

    def test_invalid_unit_conversion(self, sos_model_dict, sector_model, scenario_model):
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
            SosModel.from_dict(sos_model_dict, [sector_model, scenario_model])
        assert "meter!=ml" in str(ex.value)
