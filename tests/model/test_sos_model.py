# -*- coding: utf-8 -*-

from copy import copy
from unittest.mock import Mock

import numpy as np
import pytest
from pytest import fixture, raises
from smif.metadata import MetadataSet
from smif.model.dependency import Dependency
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel
from smif.model.sos_model import SosModel, SosModelBuilder

from ..fixtures.water_supply import WaterSupplySectorModel


@fixture(scope='function')
def get_scenario_model_object():

    scenario_model = ScenarioModel('test_scenario_model')
    scenario_model.add_output('raininess',
                              scenario_model.regions.get_entry('LSOA'),
                              scenario_model.intervals.get_entry('annual'),
                              'ml')
    scenario_model.scenario_set = 'raininess'
    return scenario_model


@fixture(scope='function')
def get_sector_model_object(get_empty_sector_model):

    sector_model = get_empty_sector_model('water_supply')

    regions = sector_model.regions
    intervals = sector_model.intervals

    sector_model.add_input('raininess',
                           regions.get_entry('LSOA'),
                           intervals.get_entry('annual'),
                           'ml')

    sector_model.add_output('cost',
                            regions.get_entry('LSOA'),
                            intervals.get_entry('annual'),
                            'million GBP')

    sector_model.add_output('water',
                            regions.get_entry('LSOA'),
                            intervals.get_entry('annual'),
                            'Ml')

    return sector_model


@fixture(scope='function')
def get_sos_model_object(get_sector_model_object,
                         get_scenario_model_object):

    sos_model = SosModel('test_sos_model')
    sector_model = get_sector_model_object
    scenario_model = get_scenario_model_object
    sos_model.add_model(sector_model)
    sos_model.add_model(scenario_model)
    sector_model.add_dependency(scenario_model, 'raininess', 'raininess')

    return sos_model


@fixture(scope='function')
def get_sos_model_with_summed_dependency(oxford_region):
    scenario_model = get_scenario_model_object

    builder = SosModelBuilder()
    builder.load_scenario_models([scenario_model])

    sos_model = builder.finish()

    region_register = sos_model.regions
    interval_register = sos_model.intervals

    raininess_model = sos_model.models['raininess']

    ws = WaterSupplySectorModel('water_supply')
    ws.add_input(
        'raininess',
        region_register.get_entry('LSOA'),
        interval_register.get_entry('annual'),
        'ml')
    ws.add_output(
        'water',
        region_register.get_entry('LSOA'),
        interval_register.get_entry('annual'),
        'Ml')
    ws.add_dependency(raininess_model, 'raininess', 'raininess')
    sos_model.add_model(ws)

    ws2 = WaterSupplySectorModel('water_supply_2')
    ws2.add_input(
        'raininess',
        region_register.get_entry('LSOA'),
        interval_register.get_entry('annual'),
        'ml')
    ws2.add_output(
        'water',
        region_register.get_entry('LSOA'),
        interval_register.get_entry('annual'),
        'Ml')
    ws2.add_dependency(raininess_model, 'raininess', 'raininess')
    sos_model.add_model(ws2)

    ws3 = WaterSupplySectorModel('water_supply_3')
    ws3.add_input(
        'water',
        region_register.get_entry('LSOA'),
        interval_register.get_entry('annual'),
        'Ml')
    # TODO implement summed dependency
    # ws3.add_dependency(ws, 'water', 'water')
    ws3.add_dependency(ws2, 'water', 'water')
    sos_model.add_model(ws3)

    return sos_model


@fixture(scope='function')
def get_empty_sector_model():

    class EmptySectorModel(SectorModel):
        """Simulate nothing
        """
        def simulate(self, data):
            return data

        def extract_obj(self, results):
            return 0

    return EmptySectorModel


class TestSosModelProperties():

    def test_model_inputs_property(self, get_sos_model_object):
        sos_model = get_sos_model_object

        expected = {'raininess': 'water_supply'}

        for key, value in expected.items():
            inputs = sos_model.models[value].inputs
            assert isinstance(inputs, MetadataSet)
            assert key in inputs.names

    def test_model_outputs_property(self, get_sos_model_object):
        sos_model = get_sos_model_object

        expected = {'cost': 'water_supply'}

        for key, value in expected.items():
            outputs = sos_model.models[value].outputs
            assert isinstance(outputs, MetadataSet)
            assert key in outputs.names


class TestSosModel():

    def test_serialise_configuration(self, get_sos_model_object):
        """Tests that as_dict function correctly returns configuration
        as a dictionary
        """
        sos_model = get_sos_model_object
        actual = sos_model.as_dict()
        expected = {
            'name': 'test_sos_model',
            'description': '',
            'scenario_sets': ['raininess'],
            'sector_models': ['water_supply'],
            'dependencies': [{
                'source_model': 'test_scenario_model',
                'source_model_output': 'raininess',
                'sink_model': 'water_supply',
                'sink_model_input': 'raininess'
            }],
            'max_iterations': 25,
            'convergence_absolute_tolerance': 1e-8,
            'convergence_relative_tolerance': 1e-5
        }
        assert actual == expected

    def test_run_with_global_parameters(self, get_sos_model_object):
        sos_model = get_sos_model_object
        sos_model.add_parameter({
            'name': 'sos_model_param',
            'description': 'A global parameter passed to all contained models',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        })
        assert 'sos_model_param' in sos_model.parameters

    def test_run_with_sector_parameters(self, get_sos_model_object):
        sos_model = get_sos_model_object
        sector_model = sos_model.models['water_supply']
        sector_model.add_parameter({
            'name': 'sector_model_param',
            'description': 'A model parameter passed to a specific model',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        })
        assert 'sector_model_param' in sector_model.parameters

    def test_add_parameters(self, get_empty_sector_model):
        sos_model = SosModel('global')
        sos_model_param = {
            'name': 'sos_model_param',
            'description': 'A global parameter passed to all contained models',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        }
        sos_model.add_parameter(sos_model_param)
        expected = sos_model_param

        assert sos_model.parameters['sos_model_param'].as_dict() == expected
        assert sos_model.parameters.names == ['sos_model_param']

        sector_model = get_empty_sector_model('source_model')
        sector_model.add_parameter({
            'name': 'sector_model_param',
            'description': 'Required for the sectormodel',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%'
        })
        sos_model.add_model(sector_model)

        # SosModel contains only its own parameters
        assert 'sos_model_param' in sos_model.parameters.names

        # SectorModel has its own ParameterList, gettable by param name
        assert 'sector_model_param' in sector_model.parameters.names

    def test_add_dependency(self, get_empty_sector_model):

        regions = Mock()
        regions.name = 'test_regions'
        intervals = Mock()
        intervals.name = 'test_intervals'
        units = 'test_units'

        sink_model = get_empty_sector_model('sink_model')
        sink_model.add_input('input_name', regions, intervals, units)

        source_model = get_empty_sector_model('source_model')
        source_model.add_output('output_name', regions, intervals, units)

        sink_model.add_dependency(source_model, 'output_name', 'input_name')

        sos_model = SosModel('test')
        sos_model.add_model(source_model)
        sos_model.add_model(sink_model)

    def test_timestep_before(self):
        sos_model = SosModel('test')
        sos_model.timesteps = [2010, 2011, 2012]
        assert sos_model.timestep_before(2010) is None
        assert sos_model.timestep_before(2011) == 2010
        assert sos_model.timestep_before(2012) == 2011
        assert sos_model.timestep_before(2013) is None

    def test_timestep_after(self):
        sos_model = SosModel('test')
        sos_model.timesteps = [2010, 2011, 2012]
        assert sos_model.timestep_after(2010) == 2011
        assert sos_model.timestep_after(2011) == 2012
        assert sos_model.timestep_after(2012) is None
        assert sos_model.timestep_after(2013) is None

    def test_run_sequential(self, get_sos_model_object):
        sos_model = get_sos_model_object
        data_handle = Mock()
        data_handle.get_state = Mock(return_value={})
        data_handle.timesteps = [2010, 2011, 2012]
        data_handle._current_timestep = 2010
        sos_model.simulate(data_handle)
        data_handle._current_timestep = 2011
        sos_model.simulate(data_handle)
        data_handle._current_timestep = 2012
        sos_model.simulate(data_handle)

    @pytest.mark.xfail(reason="Summed dependencies not yet implemented")
    def test_dependency_aggregation(self, get_sos_model_with_summed_dependency):
        sos_model = get_sos_model_with_summed_dependency

        data = {
            2010: {
                'decisions': [],
                'raininess': np.array([[1, 1]]),
                'water': np.array([2, 2])
            }
        }

        sos_model.simulate(2010, data)


@fixture(scope='function')
def get_sos_model_config(get_scenario_model_object):

    scenario_model = get_scenario_model_object
    config_data = {
        'name': 'energy_sos_model',
        'description': 'description of a sos model',
        'scenario_sets': [scenario_model],
        'sector_models': [],
        'dependencies': []
    }

    return config_data


@fixture
def get_sos_model_config_with_dep(get_sos_model_config,
                                  get_sector_model_object):

    sector_model = get_sector_model_object
    dependency_config = [{'source_model': 'test_scenario_model',
                          'source_model_output': 'raininess',
                          'sink_model': 'water_supply',
                          'sink_model_input': 'raininess'}]

    config_data = get_sos_model_config
    config_data['sector_models'].append(sector_model)
    config_data['dependencies'] = dependency_config

    return config_data


@fixture
def get_sos_model_config_with_summed_dependency(get_sos_model_config, get_sector_model_object):

    config = get_sos_model_config

    water_model_one = get_sector_model_object
    config['sector_models'].append(water_model_one)

    water_model_two = copy(config['sector_models'][0])
    water_model_two.name = 'water_supply_two'

    regions = water_model_two.regions
    intervals = water_model_two.intervals

    water_model_three = copy(config['sector_models'][0])
    water_model_three.name = 'water_supply_three'

    water_model_three.add_input('water',
                                regions.get_entry('LSOA'),
                                intervals.get_entry('annual'),
                                'Ml')
    config['dependencies'] = [{'source_model': 'water_supply',
                               'source_model_output': 'water',
                               'sink_model': 'water_supply_three',
                               'sink_model_input': 'water'},
                              {'source_model': 'water_supply_two',
                               'source_model_output': 'water',
                               'sink_model': 'water_supply_three',
                               'sink_model_input': 'water'},
                              {'source_model': 'test_scenario_model',
                               'source_model_output': 'raininess',
                               'sink_model': 'water_supply',
                               'sink_model_input': 'raininess'},
                              {'source_model': 'test_scenario_model',
                               'source_model_output': 'raininess',
                               'sink_model': 'water_supply_two',
                               'sink_model_input': 'raininess'}]
    config['sector_models'].append(water_model_two)
    config['sector_models'].append(water_model_three)
    return config


class TestSosModelBuilderComponents():

    def test_construct(self, get_sos_model_config):
        """Test constructing from single dict config
        """
        config = get_sos_model_config
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()

        assert isinstance(sos_model, SosModel)
        assert list(sos_model.scenario_models.keys()) == ['test_scenario_model']
        assert isinstance(sos_model.models['test_scenario_model'], ScenarioModel)

    def test_set_max_iterations(self, get_sos_model_config):
        """Test constructing from single dict config
        """
        config = get_sos_model_config
        config['max_iterations'] = 125
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()
        assert sos_model.max_iterations == 125

    def test_set_convergence_absolute_tolerance(self, get_sos_model_config):
        """Test constructing from single dict config
        """
        config = get_sos_model_config
        config['convergence_absolute_tolerance'] = 0.0001
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()
        assert sos_model.convergence_absolute_tolerance == 0.0001

    def test_set_convergence_relative_tolerance(self, get_sos_model_config):
        """Test constructing from single dict config
        """
        config = get_sos_model_config
        config['convergence_relative_tolerance'] = 0.1
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()
        assert sos_model.convergence_relative_tolerance == 0.1


class TestSosModelBuilder():
    """Tests that the correct SosModel structure is created from a configuration
    dictionary::

        {
            'name': 'sos_model_name',
            'description': 'friendly description of the sos model',
            'sector_models': list of Model,
            'scenario_sets': list of ScenarioModel,
            'max_iterations': int,
            'convergence_absolute_tolerance': float,
            'convergence_relative_tolerance': float,
            'dependencies': [
                {
                    'source_model': str (Model.name),
                    'source_model_output': str (Metadata.name),
                    'sink_model': str (Model.name),
                    'sink_model_output': str (Metadata.name)
                }
            ]
        }

    """
    def test_scenarios(self, get_sos_model_config):
        """Test constructing from single dict config
        """
        config = get_sos_model_config
        builder = SosModelBuilder()
        builder.construct(config)
        sos_model = builder.finish()

        assert isinstance(sos_model, SosModel)
        assert isinstance(sos_model.models['test_scenario_model'], ScenarioModel)

    def test_simple_dependency(self, get_sos_model_config_with_dep):

        config_data = get_sos_model_config_with_dep

        builder = SosModelBuilder()
        builder.construct(config_data)
        sos_model = builder.finish()

        sos_model.make_dependency_graph()
        graph = sos_model.dependency_graph

        scenario = sos_model.models['test_scenario_model']
        model = sos_model.models['water_supply']

        assert 'water_supply' in sos_model.models
        assert sos_model.models['water_supply'] in graph.nodes()
        deps = sos_model.models['water_supply'].deps
        assert 'raininess' in deps.keys()
        expected = Dependency(
            scenario,
            scenario.outputs['raininess'],
            model.inputs['raininess']
        )
        assert deps['raininess'] == expected

        assert 'test_scenario_model' in sos_model.models
        assert sos_model.models['test_scenario_model'] in graph.nodes()

    def test_data_not_present(self, get_sos_model_config_with_dep):
        """Raise a NotImplementedError if an input is defined but no dependency links
        it to a data source
        """
        config_data = get_sos_model_config_with_dep
        config_data['scenario_sets'] = []
        config_data['dependencies'] = []
        with raises(NotImplementedError):
            builder = SosModelBuilder()
            builder.construct(config_data)
            builder.finish()

    def test_undefined_unit_conversion(self, get_sos_model_config_with_dep):

        config_data = get_sos_model_config_with_dep
        sector_model = config_data['sector_models'][0]
        sector_model.inputs['raininess'].units = 'incompatible'

        with raises(ValueError) as ex:
            builder = SosModelBuilder()
            builder.construct(config_data)
            builder.finish()

        assert "Cannot convert to undefined unit 'incompatible'" in str(ex.value)

    def test_invalid_unit_conversion(self, get_sos_model_config_with_dep):

        config_data = get_sos_model_config_with_dep
        scenario = config_data['scenario_sets'][0]

        scenario.outputs['raininess'].units = 'meter'

        with raises(ValueError) as ex:
            builder = SosModelBuilder()
            builder.construct(config_data)
            builder.finish()

        assert "Cannot convert from meter to milliliter" in str(ex.value)

    def test_cyclic_dependencies(self, get_sos_model_config_with_summed_dependency):
        config_data = get_sos_model_config_with_summed_dependency

        builder = SosModelBuilder()

        with raises(NotImplementedError):
            builder.construct(config_data)
        # sos_model = builder.finish()
        # sos_model.check_dependencies()
        # graph = sos_model.dependency_graph

        # scenario = sos_model.models['raininess']
        # water_one = sos_model.models['water_supply']
        # water_two = sos_model.models['water_supply_two']
        # water_three = sos_model.models['water_supply_three']

        # print(graph.edges())

        # assert (scenario, water_one) in graph.edges()
        # assert (scenario, water_two) in graph.edges()
        # assert (water_one, water_three) in graph.edges()
        # assert (water_two, water_three) in graph.edges()
