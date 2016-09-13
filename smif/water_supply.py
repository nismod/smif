#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First approximation of a simulation model to be used in the framework

"""

from smif import SectorModel

class WaterSupply(SectorModel):
    raininess = None
    water = None
    cost = None

    def initialise(self,data):
        self.raininess = data["raininess"]

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        self.water = self.raininess
        self.cost = 1
        return {
            "water": self.water,
            "cost": self.cost
        }

    def model_executable(self):
        pass


