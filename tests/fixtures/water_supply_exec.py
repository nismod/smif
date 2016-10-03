#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements example simulation model which can be run from the command line

Arguments
=========
raininess : int
    Sets the amount of rain

"""

from argparse import ArgumentParser

from tests.fixtures.water_supply import ExampleWaterSupplySimulation


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--raininess",
                        type=int,
                        help="Sets the amount of rain")
    return parser.parse_args()


def main():
    args = argparse()
    water_supply = ExampleWaterSupplySimulation(args.raininess)
    results = water_supply.simulate()
    for key, val in results.items():
        print("{},{}".format(key, val))


if __name__ == '__main__':
    main()
