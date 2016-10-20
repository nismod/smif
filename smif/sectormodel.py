from scipy.optimize import minimize
from smif.abstract import ModelAdapter, ModelInputs


class SectorModel(object):
    """An abstract representation of the sector model with inputs and outputs

    Parameters
    ==========
    schema : dict
        A dictionary of parameter, asset and exogenous data names with expected
        types. Used for validating presented data.

    Attributes
    ==========
    model
        An instance of the sector model

    """
    def __init__(self, model, adapter_function):
        self.model = None
        self.adapted = ModelAdapter(model, adapter_function)
        self._inputs = None
        self._schema = None

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """The inputs to the model

        value : dict
            A dictionary of inputs to the model. This may include parameters,
            assets and exogenous data.

        """
        self._inputs = ModelInputs(value)

    def optimise(self):
        """Performs a static optimisation for a particular model instance

        Uses an off-the-shelf optimisation algorithm from the scipy library

        """

        v_names = self.inputs.decision_variable_names
        v_initial = self.inputs.decision_variable_values
        v_bounds = self.inputs.decision_variable_bounds

        cons = self.adapted.constraints(self.inputs.parameter_values)

        opts = {'disp': True}
        res = minimize(self.simulate,
                       v_initial,
                       options=opts,
                       method='SLSQP',
                       bounds=v_bounds,
                       constraints=cons
                       )

        results = {x: y for x, y in zip(v_names, res.x)}

        if res.success:
            print("Solver exited successfully with obj: {}".format(res.fun))
            print("and with solution: {}".format(res.x))
            print("and bounds: {}".format(v_bounds))
            print("from initial values: {}".format(v_initial))
            print("for variables: {}".format(v_names))
        else:
            print("Solver failed")

        return results

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
        results = self.adapted.simulate(static_inputs, decision_variables)
        obj = self.adapted.extract_obj(results)
        return obj
