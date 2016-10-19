from abc import ABC, abstractmethod

from scipy.optimize import minimize
from smif.abstract import SectorModel


class AbstractModelWrapper(ABC):
    """Provides in interface to wrap any simulation model for optimisation
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def simulate(self, static_inputs, decision_variables):
        """This method should allow

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
        decision_variables : x-by-1 :class:`numpy.ndarray`
        """
        pass

    @abstractmethod
    def extract_obj(self, results):
        """Implement this method to return a scalar value objective function

        Arguments
        =========
        results : :class:`dict`
            The results from the `simulate` method

        Returns
        =======
        float
            A scalar component generated from the simulation model results
        """
        pass


class WaterModelAsset(SectorModel):

    def __init__(self, model, adapter_function):
        super().__init__()
        self.adapted = ModelAdapter(model, adapter_function)

    def initialise(self):
        pass

    def simulate(self, decision_variables):
        """

        Arguments
        =========
        decision_variables : :class:`numpy.ndarray`
        """

        static_inputs = self.inputs.parameter_values
        results = self.adapted.simulate(static_inputs, decision_variables)
        obj = self.adapted.extract_obj(results)
        return obj

    def optimise(self):
        """Performs a static optimisation for a particular model instance

        Uses an off-the-shelf optimisation algorithm from the scipy library

        Notes
        =====
        This constraint below expresses that water supply must be greater than
        or equal to 3.  ``x[0]`` is the decision variable for water treatment
        capacity, while the value ``p_values[0]`` in the min term is the value
        of the raininess parameter.

        """

        v_names = self.inputs.decision_variable_names
        v_initial = self.inputs.decision_variable_values
        v_bounds = self.inputs.decision_variable_bounds
        p_values = self.inputs.parameter_values

        cons = ({'type': 'ineq',
                 'fun': lambda x: min(x[0], p_values[0]) - 3}
                )

        fun = self.simulate
        x0 = v_initial
        bnds = v_bounds
        opts = {'disp': True}
        res = minimize(fun, x0,
                       options=opts,
                       method='SLSQP',
                       bounds=bnds,
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


class ModelAdapter(object):
    """Adapts a model so that it can be used by the optisation protocol

    Arguments
    =========
    model :
        An instance of a model
    simulate :
        The function to use for implementing a `simulate` method

    """

    def __init__(self, model, simulate):
        self.model = model
        self.simulate = simulate

    def __getattr__(self, attr):
        return getattr(self.model, attr)
