from abc import ABC, abstractmethod

import numpy as np
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


def get_decision_variables(model_inputs):
    """Extracts an array of decision variables from a dictionary of inputs

    Arguments
    =========
    inputs : :class:`dict`
        See notes below.

    Returns
    =======
    ordered_names : :class:`numpy.ndarray`
        The names of the decision variables in the order specified by the
        'index' key in the entries of the inputs
    bounds : :class:`numpy.ndarray`
        The bounds ordered by the index key
    initial : :class:`numpy.ndarray`
        The initial values ordered by the index key

    Notes
    =====
    The inputs are expected to be defined using the following keys::

        'decision variables': [<list of decision variable names>]
        'parameters': [<list of parameter names>]
        '<decision variable name>': {'bounds': (<tuple of upper and lower
                                                 bound>),
                                     'index': <scalar showing position in
                                               arguments>},
                                     'init': <scalar showing initial value for
                                              solver>
                                      },
        '<parameter name>': {'bounds': (<tuple of upper and lower range for
                                        sensitivity analysis>),
                             'index': <scalar showing position in arguments>,
                             'value': <scalar showing value for model>
                              },
    """

    names = model_inputs['decision variables']
    number_of_decision_variables = len(names)

    indices = [model_inputs[name]['index'] for name in names]
    assert len(indices) == number_of_decision_variables, \
        'Index entries do not match the number of decision variables'
    initial = np.zeros(number_of_decision_variables, dtype=np.float)
    bounds = np.zeros(number_of_decision_variables, dtype=(np.float, 2))
    ordered_names = np.zeros(number_of_decision_variables, dtype='U30')

    for name, index in zip(names, indices):
        initial[index] = model_inputs[name]['init']
        bounds[index] = model_inputs[name]['bounds']
        ordered_names[index] = name

    return ordered_names, initial, bounds


def get_parameter_values(model_inputs):
    """Extracts an array of parameters from a dictionary of inputs

    Arguments
    =========
    inputs : :class:`dict`

    Returns
    =======
    ordered_names : :class:`numpy.ndarray`
        The names of the parameters in the order specified by the
        'index' key in the entries of the inputs
    bounds : :class:`numpy.ndarray`
        The parameter bounds (or range) ordered by the index key
    values : :class:`numpy.ndarray`
        The parameter values ordered by the index key


    """
    names = model_inputs['parameters']
    number_of_parameters = len(names)

    indices = [model_inputs[name]['index'] for name in names]
    assert len(indices) == number_of_parameters, \
        'Index entries do not match the number of decision variables'
    values = np.zeros(number_of_parameters, dtype=np.float)
    bounds = np.zeros(number_of_parameters, dtype=(np.float, 2))
    ordered_names = np.zeros(number_of_parameters, dtype='U30')

    for name, index in zip(names, indices):
        values[index] = model_inputs[name]['value']
        bounds[index] = model_inputs[name]['bounds']
        ordered_names[index] = name

    return ordered_names, bounds, values


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

        static_inputs = self.static_data
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

        v_names, v_initial, v_bounds = get_decision_variables(self.inputs)
        p_names, p_bounds, p_values = get_parameter_values(self.inputs)
        self.static_data = p_values

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


class AbstractModel(ABC):
    """
    """

    def simulate(self):
        pass


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
