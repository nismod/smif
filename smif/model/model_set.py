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
from smif.model import CompositeModel, element_before


class ModelSet(CompositeModel):
    """Wraps a set of interdependent models

    Arguments
    ---------
    models : list
        A list of smif.model.composite.Model
    max_iterations : int, default=25
        The maximum number of iterations that the model set will run before
        returning results
    relative_tolerance : float, default=1e-05
        Used to calculate when the model interations have converged
    absolute_tolerance : float, default=1e-08
        Used to calculate when the model interations have converged
    """
    def __init__(self, models, max_iterations=25, relative_tolerance=1e-05,
                 absolute_tolerance=1e-08):
        name = "-".join(sorted(model.name for model in models))
        super().__init__(name)
        self.models = models
        self._model_names = {model.name for model in models}
        self._derive_deps_from_models()

        self.timestep = None
        self.iterated_results = None
        self.max_iterations = max_iterations
        # tolerance for convergence assessment - see numpy.allclose docs
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def _derive_deps_from_models(self):
        for model in self.models:
            for sink, dep in model.deps.items():
                if dep.source_model not in self.models:
                    self.deps[sink] = dep
                    self.model_inputs.add_metadata_object(model.model_inputs[sink])

    def simulate(self, timestep, data=None):
        """Runs a set of one or more models

        Arguments
        ---------
        timestep : int
        data : dict, default=None
        """
        # Start by running all models in set with best guess
        # - zeroes
        # - last year's inputs
        self.iterated_results = [{}]
        self.timestep = timestep
        if data is None:
            data = {}

        for model in self.models:
            sim_data = {}
            sim_data = self._get_parameter_values(model, sim_data, data)

            results = self.guess_results(model, timestep, sim_data)
            self.iterated_results[-1][model.name] = results

        # - keep track of intermediate results (iterations within the timestep)
        # - stop iterating according to near-equality condition
        for i in range(self.max_iterations):
            if self.converged():
                break
            else:
                self._run_iteration(i, data)
        else:
            raise TimeoutError("Model evaluation exceeded max iterations")

        return self.get_last_iteration_results()

    def _run_iteration(self, i, data):
        """Run all models within the set

        Arguments
        ---------
        i : int
            Iteration counter
        data : dict
            The data passed into the model within the set
        """
        self.iterated_results.append({})
        for model in self.models:
            model_data = {}
            for input_name, dep in model.deps.items():
                input_ = model.model_inputs[input_name]
                if input_ in self.model_inputs:
                    # if external dependency
                    dep_data = data[dep.source.name]
                else:
                    # else, pull from iterated results
                    dep_data = \
                        self.iterated_results[-2][dep.source_model.name][dep.source.name]
                model_data[input_name] = dep.convert(dep_data, input_)
            results = model.simulate(self.timestep, model_data)

            self.logger.debug("Iteration %s, model %s, results: %s",
                              i, model.name, results)
            for model_name, model_results in results.items():
                self.iterated_results[-1][model_name] = model_results

    def get_last_iteration_results(self):
        """Return results from the last iteration

        Returns
        -------
        results : dict
            Dictionary of Model results, keyed by model name
        """
        return self.iterated_results[-1]

    def guess_results(self, model, timestep, data):
        """Dependency-free guess at a model's result set.

        Initially, guess zeroes, or the previous timestep's results.

        Arguments
        ---------
        model : smif.model.composite.Model
        timestep : int
        data : dict

        Returns
        -------
        results : dict
        """
        timesteps = sorted(list(data.keys()))
        timestep_before = element_before(timestep, timesteps)
        if timestep_before is not None:
            # last iteration of previous timestep results
            results = data[timestep_before][model.name]
        else:
            # generate zero-values for each parameter/region/interval combination
            results = {}
            for output in model.model_outputs.metadata:
                regions = output.get_region_names()
                intervals = output.get_interval_names()
                results[output.name] = np.zeros((len(regions), len(intervals)))
        return results

    def converged(self):
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
        if len(self.iterated_results) < 2:
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
                    self.iterated_results[-1][model.name],
                    self.iterated_results[-2][model.name])
                for model in self.models):
            # if all most recent are almost equal to penultimate, must have converged
            return True

        # TODO check for divergance and raise error

        return False

    def _model_converged(self, model, latest_results, previous_results):
        """Check a single model's output for convergence

        Compare data output for each param over recent iterations.

        Parameters
        ----------
        model: Model
        latest_results: dict
            dict of data output from model with str key (parameter name) =>
            np.ndarray value (with dimensions regions x intervals)
        previous_results: dict
            dict of data output from the model in the previous iteration

        Returns
        -------
        bool
            True if convergedm otherwise, False
        """
        return all(
            np.allclose(
                latest_results[param.name],
                previous_results[param.name],
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance
            )
            for param in model.model_outputs.metadata
        )
