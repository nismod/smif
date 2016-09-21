#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

class ExampleWaterSupplySimulation():
    """An example simulation model used for testing purposes

    """
    def __init__(self, raininess):
        self.raininess = raininess
        self.water = None
        self.cost = None

    def simulate(self):
        self.water = self.raininess
        self.cost = 1
        print("water,{}".format(self.water))
        print("cost,{}".format(self.cost))

def argparse():
    parser = ArgumentParser()
    parser.add_argument("--raininess")
    return parser.parse_args()

def main():
    args = argparse()
    water_supply = ExampleWaterSupplySimulation(args.raininess)
    results = water_supply.simulate()
    print(results)
    return results

if __name__ == '__main__':
    main()
