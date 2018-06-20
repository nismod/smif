
import numpy as np
from pytest import fixture
from smif.data_layer import DataHandle, MemoryInterface
from smif.model.sos_model import ModelSet, SectorModel


@fixture(scope='function')
def get_empty_sector_model():
    """Return model with static outputs
    """
    class EmptySectorModel(SectorModel):
        """Static water supply model
        """
        def simulate(self, data):
            data['cost'] = np.array([[2]])
            data['water'] = np.array([[0.5]])
            return data

        def extract_obj(self, results):
            return 0

    return EmptySectorModel


@fixture(scope='function')
def get_sector_model_object(get_empty_sector_model):
    """Return model configured with two outputs, no inputs
    """
    sector_model = get_empty_sector_model('water_supply')

    regions = sector_model.regions
    intervals = sector_model.intervals

    sector_model.add_output('cost',
                            regions.get_entry('LSOA'),
                            intervals.get_entry('annual'),
                            'million GBP')

    sector_model.add_output('water',
                            regions.get_entry('LSOA'),
                            intervals.get_entry('annual'),
                            'Ml')

    return sector_model


def get_data_handle(model):
    """Return a data handle for the model
    """
    store = MemoryInterface()
    store.write_sos_model_run({
        'name': 'test',
        'narratives': {}
    })
    return DataHandle(
        store,
        'test',  # modelrun_name
        2010,  # current_timestep
        [2010, 2011],  # timesteps
        model
    )


class TestModelSet:
    """Test public interface to ModelSet, with side effects in DataHandle
    """
    def test_guess_outputs_zero(self, get_sector_model_object):
        """If no previous timestep has results, guess outputs as zero
        """
        ws_model = get_sector_model_object
        model_set = ModelSet({ws_model.name: ws_model})
        data_handle = get_data_handle(model_set)
        model_set.simulate(data_handle)
        expected = {
            "cost": np.zeros((1, 1)),
            "water": np.zeros((1, 1))
        }
        actual = {
            "cost": data_handle.get_results('cost', ws_model.name, 0),
            "water": data_handle.get_results('water', ws_model.name, 0)
        }
        assert actual == expected

    def test_guess_outputs_last_year(self, get_sector_model_object):
        """If a previous timestep has results, guess outputs as identical
        """
        ws_model = get_sector_model_object
        model_set = ModelSet({ws_model.name: ws_model})

        expected = {
            "cost": np.array([[3.14]]),
            "water": np.array([[2.71]])
        }

        # set up data as though from previous timestep simulation
        data_handle = get_data_handle(model_set)
        data_handle._store.write_results(
            'test', 'water_supply', 'cost', expected['cost'], 'LSOA', 'annual', 2010, 0)
        data_handle._store.write_results(
            'test', 'water_supply', 'water', expected['water'], 'LSOA', 'annual', 2010, 0)

        data_handle._current_timestep = 2011
        model_set.simulate(data_handle)
        actual = {
            "cost": data_handle.get_results('cost', ws_model.name, 0),
            "water": data_handle.get_results('water', ws_model.name, 0)
        }
        assert actual == expected

    def test_converged_first_iteration(self, get_sector_model_object):
        """Should not report convergence after a single iteration
        """
        ws_model = get_sector_model_object
        model_set = ModelSet({ws_model.name: ws_model})
        data_handle = get_data_handle(model_set)
        model_set.simulate(data_handle)
        assert model_set.max_iteration > 0

    def test_converged_two_identical(self, get_sector_model_object):
        """Should report converged if the last two output sets are identical
        """
        ws_model = get_sector_model_object
        model_set = ModelSet({ws_model.name: ws_model})

        data_handle = get_data_handle(model_set)
        model_set.simulate(data_handle)
        # ws_model will always return identical results, so we expect:
        # 0: guessed zeroes
        # 1: returned values
        # 2: returned identical values
        assert model_set.max_iteration == 2
