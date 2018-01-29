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
    models : dict
        A dict of model_name str => model :class:`smif.model.Model`
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
        name = "<->".join(sorted(model.name for model in models.values()))
        super().__init__(name)
        self.models = models
        self._model_names = list(models.keys())
        self._derive_deps_from_models()
        self._current_iteration = 0
        self._did_converge = False

        self.max_iterations = int(max_iterations)
        # tolerance for convergence assessment - see numpy.allclose docs
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)

    def _derive_deps_from_models(self):
        for model in self.models.values():
            for sink, dep in model.deps.items():
                if dep.source_model not in self.models.values():
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
            if self._converged(data_handle):
                self._did_converge = True
                break
            else:
                self._run_iteration(i, data_handle)
        else:
            raise TimeoutError("Model evaluation exceeded max iterations")

    @property
    def max_iteration(self):
        """The maximum iteration reached before convergence
        """
        if self._did_converge:
            return self._current_iteration
        else:
            return None

    def _run_iteration(self, i, data_handle):
        """Run all models within the set

        Arguments
        ---------
        i : int
            Iteration counter
        data_handle : smif.data_layer.DataHandle
        """
        for model in self.models.values():
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
                self._guess_results(model, model_data_handle)
            else:
                model.simulate(model_data_handle)

    def _guess_results(self, model, data_handle):
        """Dependency-free guess at a model's input result set.

        Initially, guess zeroes, or the previous timestep's results.

        Arguments
        ---------
        model : smif.model.composite.Model
        data_handle : smif.data_layer.DataHandle

        Returns
        -------
        data_handle : smif.data_layer.DataHandle
        """
        timestep_before = element_before(
            data_handle.current_timestep,
            data_handle.timesteps
        )
        if timestep_before is not None:
            # last iteration of previous timestep results
            print(timestep_before)
            for output in model.outputs.metadata:
                data_handle.set_results(
                    output.name,
                    data_handle.get_results(output.name, timestep=timestep_before)
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
        return data_handle

    def _converged(self, data_handle):
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

        # each data output is a dict with
        #   str key (parameter name) =>
        #       np.ndarray value (regions x intervals)
        converged = []
        for model in self.models.values():
            model_data_handle = DataHandle(
                data_handle._store,
                data_handle._modelrun_name,
                data_handle._current_timestep,
                data_handle._timesteps,
                model,
                self._current_iteration,
                data_handle._decision_iteration
            )
            converged.append(self._model_converged(model, model_data_handle))

        if all(converged):
            # if all most recent are almost equal to penultimate, must have converged
            return True

        # TODO check for divergence and raise error

        return False

    def _model_converged(self, model, data_handle):
        """Check a single model's output for convergence

        Compare data output for each param over recent iterations.

        Parameters
        ----------
        model: :class:`smif.model.Model`
        data_handle: :class:`smif.data_layer.DataHandle`

        Returns
        -------
        bool
            True if converged otherwise, False
        """
        prev_data_handle = DataHandle(
            data_handle._store,
            data_handle._modelrun_name,
            data_handle._current_timestep,
            data_handle._timesteps,
            model,
            self._current_iteration - 1,  # access previous iteration
            data_handle._decision_iteration
        )
        return all(
            np.allclose(
                data_handle.get_data(param.name),
                prev_data_handle.get_data(param.name),
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance
            )
            for param in model.inputs.metadata
        )
