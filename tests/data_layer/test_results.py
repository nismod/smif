"""Test the Store interface

Many methods simply proxy to config/metadata/data store implementations, but there is some
cross-coordination and there are some convenience methods implemented at this layer.
"""

import os

from pytest import fixture, raises
from smif.data_layer import Results, Store
from smif.data_layer.memory_interface import (MemoryConfigStore,
                                              MemoryDataStore,
                                              MemoryMetadataStore)
from smif.exception import SmifDataNotFoundError


@fixture
def store():
    """Store fixture
    """
    return Store(
        config_store=MemoryConfigStore(),
        metadata_store=MemoryMetadataStore(),
        data_store=MemoryDataStore()
    )


@fixture
def results(store):
    """Results fixture
    """
    return Results(store=store)


@fixture
def results_with_model_run(store, model_run):
    """Results fixture
    """
    store.write_model_run(model_run)
    return Results(store=store)


class TestNoResults:

    def test_exceptions(self, store):

        # No arguments is not allowed
        with raises(AssertionError) as e:
            Results()
        assert 'either a details dict or a store' in str(e.value)

        # Both arguments is also not allowed
        with raises(AssertionError) as e:
            Results(details_dict={'some': 'dict'}, store=store)
        assert 'either a details dict or a store' in str(e.value)

        # Check that constructing with just a store works fine
        Results(store=store)

        # Check that valid configurations do work (but expect a SmifDataNotFoundError
        # because the store creation will fall over
        with raises(SmifDataNotFoundError) as e:
            Results(details_dict={'interface': 'local_csv', 'dir': '.'})
        assert 'Expected configuration folder' in str(e.value)

        with raises(SmifDataNotFoundError) as e:
            Results(details_dict={'interface': 'local_parquet', 'dir': '.'})
        assert 'Expected configuration folder' in str(e.value)

        # Interface left blank will default to local_csv
        with raises(SmifDataNotFoundError) as e:
            Results(details_dict={'dir': '.'})
        assert 'Expected configuration folder' in str(e.value)

        # Dir left blank will default to '.'
        with raises(SmifDataNotFoundError) as e:
            Results(details_dict={'interface': 'local_parquet'})
        assert 'Expected configuration folder' in str(e.value)

        # Invalid interface will raise a ValueError
        with raises(ValueError) as e:
            Results(details_dict={'interface': 'invalid', 'dir': '.'})
        assert 'Unsupported interface "invalid"' in str(e.value)

        # Invalid directory will raise a ValueError
        with raises(ValueError) as e:
            invalid_dir = os.path.join(os.path.dirname(__file__), 'does', 'not', 'exist')
            Results(details_dict={'interface': 'local_csv', 'dir': invalid_dir})
        assert 'to be a valid directory' in str(e.value)

    def test_list_model_runs(self, results, model_run):

        # Should be no model runs in an empty Results()
        assert results.list_model_runs() == []

        model_run_a = model_run.copy()
        model_run_a['name'] = 'a_model_run'

        model_run_b = model_run.copy()
        model_run_b['name'] = 'b_model_run'

        results._store.write_model_run(model_run_a)
        results._store.write_model_run(model_run_b)

        assert results.list_model_runs() == ['a_model_run', 'b_model_run']

    def test_available_results(self, results_with_model_run):

        available = results_with_model_run.available_results('unique_model_run_name')

        assert available['model_run'] == 'unique_model_run_name'
        assert available['sos_model'] == 'energy'
        assert available['sector_models'] == dict()


class TestSomeResults:

    def test_available_results(self, results_with_model_run, sample_results):

        results_with_model_run._store.write_results(
            sample_results, 'model_run_name', 'model_name', 0
        )

        available = results_with_model_run.available_results('unique_model_run_name')
        assert available

        # assert (available['model_run'] == 'energy_central')
        # assert (available['sos_model'] == 'energy')
        #
        # sec_models = available['sector_models']
        # assert (sorted(sec_models.keys()) == ['energy_demand'])
        #
        # outputs = sec_models['energy_demand']['outputs']
        # assert (sorted(outputs.keys()) == ['cost', 'water_demand'])
        #
        # output_answer = {1: [2010], 2: [2010], 3: [2015], 4: [2020]}
        #
        # assert outputs['cost'] == output_answer
        # assert outputs['water_demand'] == output_answer

    def test_read_exceptions(self, results_with_model_run):

        # Passing anything other than one sector model or output is current not implemented
        with raises(NotImplementedError) as e:
            results_with_model_run.read(
                model_run_names=['one', 'two'],
                sec_model_names=[],
                output_names=['one']
            )
        assert 'requires exactly one sector model' in str(e.value)

        with raises(NotImplementedError) as e:
            results_with_model_run.read(
                model_run_names=['one', 'two'],
                sec_model_names=['one', 'two'],
                output_names=['one']
            )
        assert 'requires exactly one sector model' in str(e.value)

        with raises(NotImplementedError) as e:
            results_with_model_run.read(
                model_run_names=['one', 'two'],
                sec_model_names=['one'],
                output_names=[]
            )
        assert 'requires exactly one output' in str(e.value)

        with raises(NotImplementedError) as e:
            results_with_model_run.read(
                model_run_names=['one', 'two'],
                sec_model_names=['one'],
                output_names=['one', 'two']
            )
        assert 'requires exactly one output' in str(e.value)
