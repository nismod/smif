from scipy.optimize import minimize
from smif.abstract import ModelInputs


class SectorModel(object):
    """An abstract representation of the sector model with inputs and outputs

    Parameters
    ==========
    schema : dict
        A dictionary of parameter, asset and exogenous data names with expected
        types. Used for validating presented data.

    Attributes
    ==========
    model : :class:`smif.abstract.AbstractModelWrapper`
        An instance of a wrapped simulation model

    """
    def __init__(self, model):
        self.model = model
        self._inputs = None
        self._schema = None

    @property
    def inputs(self):
        """The inputs to the model

        Returns
        =======
        :class:`smif.abstract.ModelInputs`

        """
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """The inputs to the model

        Arguments
        =========
        value : dict
            A dictionary of inputs to the model. This may include parameters,
            assets and exogenous data.

        """
        self._inputs = ModelInputs(value)

    def optimise(self):
        """Performs a static optimisation for a particular model instance

        Uses an off-the-shelf optimisation algorithm from the scipy library

        Returns
        =======
        dict
            A set of optimised simulation results


        """
        assert self.inputs, "Inputs to the model not yet specified"

        v_names = self.inputs.decision_variable_names
        v_initial = self.inputs.decision_variable_values
        v_bounds = self.inputs.decision_variable_bounds

        cons = self.model.constraints(self.inputs.parameter_values)

        opts = {'disp': True}
        res = minimize(self._simulate_optimised,
                       v_initial,
                       options=opts,
                       method='SLSQP',
                       bounds=v_bounds,
                       constraints=cons
                       )

        # results = {x: y for x, y in zip(v_names, res.x)}
        results = self.simulate(res.x)

        if res.success:
            print("Solver exited successfully with obj: {}".format(res.fun))
            print("and with solution: {}".format(res.x))
            print("and bounds: {}".format(v_bounds))
            print("from initial values: {}".format(v_initial))
            print("for variables: {}".format(v_names))
        else:
            print("Solver failed")

        return results

    def _simulate_optimised(self, decision_variables):
        results = self.simulate(decision_variables)
        obj = self.model.extract_obj(results)
        return obj

    def simulate(self, decision_variables):
        """Performs an operational simulation of the sector model

        Arguments
        =========
        decision_variables : :class:`numpy.ndarray`

        Note
        ====
        The term simulation may refer to operational optimisation, rather than
        simulation-only. This process is described as simulation to distinguish
        from the definition of investments in capacity, versus operation using
        the given capacity
        """

        static_inputs = self.inputs.parameter_values
        results = self.model.simulate(static_inputs, decision_variables)
        return results
