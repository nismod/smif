"""Test the Store interface

Many methods simply proxy to config/metadata/data store implementations, but there is some
cross-coordination and there are some convenience methods implemented at this layer.
"""

import os
import subprocess

from pytest import fixture, raises
from smif.data_layer import Results


@fixture(scope="session")
def tmp_sample_project_no_results(tmpdir_factory):
    test_folder = tmpdir_factory.mktemp("smif")
    subprocess.run(
        ['smif', 'setup', '-d', str(test_folder), '-v'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return str(test_folder)


@fixture(scope="session")
def tmp_sample_project_with_results(tmpdir_factory):
    test_folder = tmpdir_factory.mktemp("smif")
    subprocess.run(
        ['smif', 'setup', '-d', str(test_folder), '-v'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    subprocess.run(
        ['smif', 'run', '-d', str(test_folder), 'energy_central'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return str(test_folder)


class TestNoResults:

    def test_exceptions(self, tmp_sample_project_no_results):
        # Check that invalid interface is dealt with properly
        with raises(ValueError) as e:
            Results(interface='unexpected')
        assert ('Unsupported interface' in str(e.value))

        # Check that invalid directories are dealt with properly
        with raises(ValueError) as e:
            fake_path = os.path.join(tmp_sample_project_no_results, 'not', 'valid')
            Results(model_base_dir=fake_path)
            assert ('to be a valid directory' in str(e.value))

        # Check that valid options DO work
        Results(interface='local_csv', model_base_dir=tmp_sample_project_no_results)
        Results(interface='local_parquet', model_base_dir=tmp_sample_project_no_results)

    def test_list_model_runs(self, tmp_sample_project_no_results):
        res = Results(interface='local_csv', model_base_dir=tmp_sample_project_no_results)
        model_runs = res.list_model_runs()

        assert ('energy_central' in model_runs)
        assert ('energy_water_cp_cr' in model_runs)
        assert (len(model_runs) == 2)

    def test_available_results(self, tmp_sample_project_no_results):
        res = Results(interface='local_csv', model_base_dir=tmp_sample_project_no_results)
        available = res.available_results('energy_central')

        assert (available['model_run'] == 'energy_central')
        assert (available['sos_model'] == 'energy')
        assert (available['sector_models'] == dict())


class TestSomeResults:

    def test_available_results(self, tmp_sample_project_with_results):
        res = Results(interface='local_csv', model_base_dir=tmp_sample_project_with_results)
        available = res.available_results('energy_central')

        assert (available['model_run'] == 'energy_central')
        assert (available['sos_model'] == 'energy')

        sec_models = available['sector_models']
        assert (sorted(sec_models.keys()) == ['energy_demand'])

        outputs = sec_models['energy_demand']['outputs']
        assert (sorted(outputs.keys()) == ['cost', 'water_demand'])

        output_answer = {1: [2010], 2: [2010], 3: [2015], 4: [2020]}

        assert outputs['cost'] == output_answer
        assert outputs['water_demand'] == output_answer
