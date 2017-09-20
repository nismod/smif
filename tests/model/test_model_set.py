
import numpy as np
from pytest import fixture
from smif.model.sos_model import ModelSet, SectorModel


@fixture(scope='function')
def get_empty_sector_model():

    class EmptySectorModel(SectorModel):

        def initialise(self, initial_conditions):
            pass

        def simulate(self, timestep, data=None):
            return {'output_name': 'some_data', 'timestep': timestep}

        def extract_obj(self, results):
            return 0

    return EmptySectorModel


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


class TestModelSet:

    def test_guess_outputs_zero(self, get_sector_model_object):
        """If no previous timestep has results, guess outputs as zero
        """
        ws_model = get_sector_model_object
        model_set = ModelSet([ws_model])

        results = model_set.guess_results(ws_model, 2010, {})
        expected = {
            "cost": np.zeros((1, 1)),
            "water": np.zeros((1, 1))
        }
        assert results == expected

    def test_guess_outputs_last_year(self, get_sector_model_object):
        """If a previous timestep has results, guess outputs as identical
        """
        ws_model = get_sector_model_object
        model_set = ModelSet([ws_model])

        expected = {
            "cost": np.array([[3.14]]),
            "water": np.array([[2.71]])
        }

        # set up data as though from previous timestep simulation
        data = {
            2010: {
                'water_supply': expected
            },
            2011: {}
        }

        results = model_set.guess_results(ws_model, 2011, data)
        assert results == expected

    def test_converged_first_iteration(self, get_sector_model_object):
        """Should not report convergence after a single iteration
        """
        ws_model = get_sector_model_object
        model_set = ModelSet([ws_model])

        results = model_set.guess_results(ws_model, 2010, {})
        model_set.iterated_results = [{ws_model.name: results}]

        assert not model_set.converged()

    def test_converged_two_identical(self, get_sector_model_object):
        """Should report converged if the last two output sets are identical
        """
        ws_model = get_sector_model_object
        model_set = ModelSet([ws_model])

        results = model_set.guess_results(ws_model, 2010, {})
        model_set.iterated_results = [
            {
                "water_supply": results
            },
            {
                "water_supply": results
            }
        ]

        assert model_set.converged()
