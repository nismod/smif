"""Test the Results interface
"""

import os

from pytest import fixture, raises
from smif.data_layer import Results
from smif.exception import SmifDataNotFoundError


@fixture
def results(empty_store, model_run):
    """Results fixture
    """
    empty_store.write_model_run(model_run)
    return Results(store=empty_store)


@fixture
def results_with_results(empty_store, model_run, sample_results):
    """Results fixture with a model run and fictional results
    """
    empty_store.write_model_run(model_run)

    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2010, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2015, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2020, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2015, 1)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2020, 1)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2015, 2)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'a_model', 2020, 2)

    empty_store.write_results(sample_results, 'unique_model_run_name', 'b_model', 2010, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'b_model', 2015, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'b_model', 2020, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'b_model', 2025, 0)
    empty_store.write_results(sample_results, 'unique_model_run_name', 'b_model', 2030, 0)

    return Results(store=empty_store)


class TestNoResults:

    def test_exceptions(self, empty_store):

        # No arguments is not allowed
        with raises(AssertionError) as e:
            Results()
        assert 'either a details dict or a store' in str(e.value)

        # Both arguments is also not allowed
        with raises(AssertionError) as e:
            Results(details_dict={'some': 'dict'}, store=empty_store)
        assert 'either a details dict or a store' in str(e.value)

        # Check that constructing with just a store works fine
        Results(store=empty_store)

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

    def test_list_model_runs(self, empty_store, model_run):

        # Should be no model runs in an empty Results()
        results = Results(store=empty_store)
        assert results.list_model_runs() == []

        model_run_a = model_run.copy()
        model_run_a['name'] = 'a_model_run'

        model_run_b = model_run.copy()
        model_run_b['name'] = 'b_model_run'

        empty_store.write_model_run(model_run_a)
        empty_store.write_model_run(model_run_b)

        assert results.list_model_runs() == ['a_model_run', 'b_model_run']

    def test_available_results(self, results):

        available = results.available_results('unique_model_run_name')

        assert available['model_run'] == 'unique_model_run_name'
        assert available['sos_model'] == 'energy'
        assert available['sector_models'] == dict()


class TestSomeResults:

    def test_available_results(self, results_with_results):

        available = results_with_results.available_results('unique_model_run_name')

        assert available['model_run'] == 'unique_model_run_name'
        assert available['sos_model'] == 'energy'

        sec_models = available['sector_models']
        assert sorted(sec_models.keys()) == ['a_model', 'b_model']

        # Check a_model outputs are correct
        outputs_a = sec_models['a_model']['outputs']
        assert sorted(outputs_a.keys()) == ['energy_use']

        output_answer_a = {0: [2010, 2015, 2020], 1: [2015, 2020], 2: [2015, 2020]}
        assert outputs_a['energy_use'] == output_answer_a

        # Check b_model outputs are correct
        outputs_b = sec_models['b_model']['outputs']
        assert sorted(outputs_b.keys()) == ['energy_use']

        output_answer_b = {0: [2010, 2015, 2020, 2025, 2030]}
        assert outputs_b['energy_use'] == output_answer_b

    def test_read_validate_names(self, results_with_results):

        # Passing anything other than one sector model or output is current not implemented
        with raises(NotImplementedError) as e:
            results_with_results.read(
                model_run_names=['one', 'two'],
                sec_model_names=[],
                output_names=['one']
            )
        assert 'requires exactly one sector model' in str(e.value)

        with raises(NotImplementedError) as e:
            results_with_results.read(
                model_run_names=['one', 'two'],
                sec_model_names=['one', 'two'],
                output_names=['one']
            )
        assert 'requires exactly one sector model' in str(e.value)

        with raises(ValueError) as e:
            results_with_results.read(
                model_run_names=[],
                sec_model_names=['one'],
                output_names=['one']
            )
        assert 'requires at least one sector model name' in str(e.value)

        with raises(ValueError) as e:
            results_with_results.read(
                model_run_names=['one'],
                sec_model_names=['one'],
                output_names=[]
            )
        assert 'requires at least one output name' in str(e.value)

    def test_read(self):
        # This is difficult to test without fixtures defining an entire canonical project.
        # See smif issue #304 (https://github.com/nismod/smif/issues/304).  For now, we rely
        # on tests of the underling get_results() method on the Store.
        pass
