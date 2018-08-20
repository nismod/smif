
import numpy as np
from pytest import fixture
from smif.data_layer import DataHandle, MemoryInterface
from smif.metadata import Spec
from smif.model.sos_model import ModelSet, SectorModel


class EmptySectorModel(SectorModel):
    """Static water supply model
    """
    def simulate(self, data):
        data['cost'] = np.array([[2]])
        data['water'] = np.array([[0.5]])
        return data


@fixture(scope='function')
def empty_sector_model():
    """Return model with static outputs
    """
    return EmptySectorModel('water_supply')


@fixture(scope='function')
def sector_model(empty_sector_model):
    """Return model configured with two outputs, no inputs
    """
    model = empty_sector_model
    model.add_output(Spec(
        name='cost',
        dims=['LSOA', 'annual'],
        coords={'LSOA': [1], 'annual': [1]},
        dtype='float',
        unit='million GBP'
    ))

    model.add_output(Spec(
        name='water',
        dims=['LSOA', 'annual'],
        coords={'LSOA': [1], 'annual': [1]},
        dtype='float',
        unit='Ml'
    ))

    return model


def get_data_handle(model):
    """Return a data handle for the model
    """
    store = MemoryInterface()
    store.write_model_run({
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
    def test_guess_outputs_zero(self, sector_model):
        """If no previous timestep has results, guess outputs as zero
        """
        model_set = ModelSet({sector_model.name: sector_model})
        data_handle = get_data_handle(model_set)
        model_set.simulate(data_handle)
        expected = {
            "cost": np.zeros((1, 1)),
            "water": np.zeros((1, 1))
        }
        actual = {
            "cost": data_handle.get_results(('water_supply', 'cost'), sector_model.name, 0),
            "water": data_handle.get_results(('water_supply', 'water'), sector_model.name, 0)
        }
        assert actual == expected

    def test_guess_outputs_last_year(self, sector_model):
        """If a previous timestep has results, guess outputs as identical
        """
        model_set = ModelSet({sector_model.name: sector_model})

        expected = {
            "cost": np.array([[3.14]]),
            "water": np.array([[2.71]])
        }

        # set up data as though from previous timestep simulation
        data_handle = get_data_handle(model_set)
        data_handle._store.write_results(
            expected['cost'], 'test', 'water_supply', sector_model.outputs['cost'], 2010, 0)
        data_handle._store.write_results(
            expected['water'], 'test', 'water_supply', sector_model.outputs['water'], 2010, 0)

        data_handle._current_timestep = 2011
        model_set.simulate(data_handle)
        actual = {
            "cost": data_handle.get_results(('water_supply', 'cost'), sector_model.name, 0),
            "water": data_handle.get_results(('water_supply', 'water'), sector_model.name, 0)
        }
        assert actual == expected

    def test_converged_first_iteration(self, sector_model):
        """Should not report convergence after a single iteration
        """
        model_set = ModelSet({sector_model.name: sector_model})
        data_handle = get_data_handle(model_set)
        model_set.simulate(data_handle)
        assert model_set.max_iteration > 0

    def test_converged_two_identical(self, sector_model):
        """Should report converged if the last two output sets are identical
        """
        model_set = ModelSet({sector_model.name: sector_model})

        data_handle = get_data_handle(model_set)
        model_set.simulate(data_handle)
        # sector_model will always return identical results, so we expect:
        # 0: guessed zeroes
        # 1: returned values
        # 2: returned identical values
        assert model_set.max_iteration == 2
