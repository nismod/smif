"""Wrap and solve a set of interdependent models

Given a directed graph of dependencies between models, any cyclic
dependencies are contained within the strongly-connected components of the
graph.

A ModelSet corresponds to the set of models within a single strongly-
connected component. This class provides the machinery necessary
to find a solution to each of the interdependent models.

The current implementation first estimates the outputs for each model in the
set, guaranteeing that each model will then be able to run, then begins
iterating, running every model in the set at each iteration, monitoring the
model outputs over the iterations, and stopping at timeout, divergence or
convergence.
"""

import numpy as np
from smif.data_layer import DataHandle
from smif.model import CompositeModel, element_before


class ModelSet(CompositeModel):
    """Wraps a set of interdependent models

    Parameters
    ----------
    models : list
        A list of :class:`smif.model.Model`
    max_iterations : int, default=25
        The maximum number of iterations that the model set will run before
        returning results
    relative_tolerance : float, default=1e-05
        Used to calculate when the model interations have converged
    absolute_tolerance : float, default=1e-08
        Used to calculate when the model interations have converged

    Attributes
    ----------
    max_iterations: int
        The maximum number of iterations
    models: list
        The list of :class:`smif.model.Model` subclasses
    """
    def __init__(self, models, max_iterations=25, relative_tolerance=1e-05,
                 absolute_tolerance=1e-08):
        name = "<->".join(sorted(model.name for model in models))
        super().__init__(name)
        self.models = models
        self._model_names = {model.name for model in models}
        self._derive_deps_from_models()
        self._current_iteration = 0

        self.max_iterations = int(max_iterations)
        # tolerance for convergence assessment - see numpy.allclose docs
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)

    def _derive_deps_from_models(self):
        for model in self.models:
            for sink, dep in model.deps.items():
                if dep.source_model not in self.models:
                    self.deps[sink] = dep
                    self.inputs.add_metadata(model.inputs[sink])

    def simulate(self, data_handle):
        """Runs a set of one or more models

        Arguments
        ---------
        timestep : int
        data : dict, default=None
        """
        self.logger.info("Simulating %s", self.name)

        # - keep track of intermediate results (iterations within the timestep)
        # - stop iterating according to near-equality condition
        for i in range(self.max_iterations):
            self._current_iteration = i
            if self.converged(data_handle):
                break
            else:
                self._run_iteration(i, data_handle)
        else:
            raise TimeoutError("Model evaluation exceeded max iterations")

    def _run_iteration(self, i, data_handle):
        """Run all models within the set

        Arguments
        ---------
        i : int
            Iteration counter
        data_handle : smif.data_layer.DataHandle
        """
        for model in self.models:
            self.logger.info("Simulating %s, iteration %s", model.name, i)
            model_data_handle = DataHandle(
                data_handle._store,
                data_handle._modelrun_name,
                data_handle._current_timestep,
                data_handle._timesteps,
                model,
                i,
                data_handle._decision_iteration
            )
            # Start by running all models in set with best guess
            # - zeroes
            # - last year's inputs
            if i == 0:
                self.guess_results(model, model_data_handle)
            else:
                model.simulate(model_data_handle)

    def guess_results(self, model, data_handle):
        """Dependency-free guess at a model's input result set.

        Initially, guess zeroes, or the previous timestep's results.

        Arguments
        ---------
        model : smif.model.composite.Model
        data_handle : smif.data_layer.DataHandle

        Returns
        -------
        results : dict
        """
        timestep_before = element_before(
            data_handle.current_timestep,
            data_handle.timesteps
        )
        if timestep_before is not None:
            # last iteration of previous timestep results
            for output in model.outputs.metadata:
                data_handle.set_results(
                    data_handle.get_data(output.name, timestep_before)
                )
        else:
            # generate zero-values for each parameter/region/interval combination
            for output in model.outputs.metadata:
                regions = output.get_region_names()
                intervals = output.get_interval_names()
                data_handle.set_results(
                    output.name,
                    np.zeros((len(regions), len(intervals)))
                )

    def converged(self, data_handle):
        """Check whether the results of a set of models have converged.

        Returns
        -------
        converged: bool
            True if the results have converged to within a tolerance

        Raises
        ------
        DiverganceError
            If the results appear to be diverging
        """
        if self._current_iteration < 2:
            # must have at least two result sets per model to assess convergence
            return False

        # iterated_results is a list of dicts with
        #   str key (model name) =>
        #       list of data output from models
        #
        # each data output is a dict with
        #   str key (parameter name) =>
        #       np.ndarray value (regions x intervals)
        if all(
                self._model_converged(
                    model,
                    data_handle.get_iteration_data(model.name, -1),
                    data_handle.get_iteration_data(model.name, -2)
                )
                for model in self.models):
            # if all most recent are almost equal to penultimate, must have converged
            return True

        # TODO check for divergence and raise error

        return False

    def _model_converged(self, model, latest_results, previous_results):
        """Check a single model's output for convergence

        Compare data output for each param over recent iterations.

        Parameters
        ----------
        model: :class:`smif.model.Model`
        latest_results: dict
            dict of data output from model with str key (parameter name) =>
            np.ndarray value (with dimensions regions x intervals)
        previous_results: dict
            dict of data output from the model in the previous iteration

        Returns
        -------
        bool
            True if converged otherwise, False
        """
        return all(
            np.allclose(
                latest_results[param.name],
                previous_results[param.name],
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance
            )
            for param in model.outputs.metadata
        )
