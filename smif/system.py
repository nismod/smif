#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Concrete implementation of classes for sector models

"""
from __future__ import absolute_import, division, print_function

import logging
from scipy.optimize import minimize
from smif.abstract import SectorModel

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class WaterModelAsset(SectorModel):

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
