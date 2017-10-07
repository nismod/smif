# -*- coding: utf-8 -*-
"""A command line interface to the system of systems framework

This command line interface implements a number of methods.

- `setup` creates a new project folder structure in a location
- `run` performs a simulation of an individual sector model, or the whole
        system of systems model
- `validate` performs a validation check of the configuration file

Folder structure
----------------

When configuring a system-of-systems model for the CLI, the folder structure
below should be used.  In this example, there is one sector model, called
``water_supply``::

    /main_config.yaml
    /timesteps.yaml
    /water_supply.yaml
    /data/all/inputs.yaml
    /data/water_supply/
    /data/water_supply/inputs.yaml
    /data/water_supply/outputs.yaml
    /data/water_supply/assets/assets1.yaml
    /data/water_supply/planning/
    /data/water_supply/planning/pre-specified.yaml

The ``data`` folder contains one subfolder for each sector model.

The sector model implementations can be installed independently of the model
run configuration. The main_config.yaml file specifies which sector models
should run, while each set of sector model config

"""
from __future__ import print_function
import logging
import logging.config
import os
import re
import sys
import traceback
from argparse import ArgumentParser

import smif
from smif.data_layer.load import dump

from smif.data_layer import DatafileInterface

from smif.convert.area import get_register as get_region_register
from smif.convert.area import RegionSet
from smif.convert.interval import get_register as get_interval_register
from smif.convert.interval import IntervalSet

from smif.modelrun import ModelRunBuilder
from smif.model.sos_model import SosModelBuilder
from smif.model.sector_model import SectorModelBuilder
from smif.model.scenario_model import ScenarioModelBuilder

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
        },
        'message': {
            'format': '\033[1;34m%(levelname)-8s\033[0m %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'smif.log',
            'mode': 'a',
            'encoding': 'utf-8'
        },
        'stream': {
            'class': 'logging.StreamHandler',
            'formatter': 'message',
            'level': 'DEBUG'
        }
    },
    'root': {
        'handlers': ['file', 'stream'],
        'level': 'DEBUG'
    }
}

# Configure logging once, outside of any dependency on argparse
VERBOSITY = None
if '--verbose' in sys.argv:
    VERBOSITY = sys.argv.count('--verbose')
else:
    for arg in sys.argv:
        if re.match(r'\A-v+\Z', arg):
            VERBOSITY = len(arg) - 1
            break

if VERBOSITY is None:
    LOGGING_CONFIG['root']['level'] = logging.WARNING
elif VERBOSITY == 1:
    LOGGING_CONFIG['root']['level'] = logging.INFO
else:
    LOGGING_CONFIG['root']['level'] = logging.DEBUG

logging.config.dictConfig(LOGGING_CONFIG)
LOGGER = logging.getLogger(__name__)
LOGGER.debug('Debug logging enabled.')

REGIONS = get_region_register()
INTERVALS = get_interval_register()


def setup_project_folder(project_path):
    """Creates folder structure in the target directory

    Arguments
    =========
    project_path : str
        Absolute path to an empty folder

    """
    folder_list = ['data']
    for folder in folder_list:
        folder_path = os.path.join(project_path, folder)
        if os.path.exists(folder_path):
            msg = "{} already exists, skipping...".format(folder_path)
        else:
            msg = "Creating {} folder in {}".format(folder, project_path)
            os.mkdir(folder_path)
        LOGGER.info(msg)


def setup_configuration(args):
    """Sets up the configuration files into the defined project folder

    """
    project_path = os.path.abspath(args.path)
    msg = "Set up the project folders in {}?".format(project_path)
    response = confirm(msg,
                       response=False)
    if response:
        msg = "Setting up the project folders in {}".format(project_path)
        setup_project_folder(project_path)
    else:
        msg = "Setup cancelled."
    LOGGER.info(msg)


def load_region_sets(handler):
    """Loads the region sets into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`

    """
    region_definitions = handler.read_region_definitions()
    for region_def in region_definitions:
        region_name = region_def['name']
        region_data = handler.read_region_definition_data(region_name)
        region_set = RegionSet(region_name, region_data)
        REGIONS.register(region_set)


def load_interval_sets(handler):
    """Loads the time-interval sets into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`

    """
    interval_definitions = handler.read_interval_definitions()
    for interval_def in interval_definitions:
        interval_name = interval_def['name']
        interval_data = handler.read_interval_definition_data(interval_name)
        interval_set = IntervalSet(interval_name, interval_data)
        INTERVALS.register(interval_set)


def get_model_run_definition(args):
    """Builds the model run

    Returns
    -------
    dict
        The complete sos_model_run configuration dictionary with contained
        ScenarioModel, SosModel and SectorModel objects

    """
    handler = DatafileInterface(args.path)
    load_region_sets(handler)
    load_interval_sets(handler)

    # HARDCODE selet the first model run only
    model_run_config = handler.read_sos_model_runs()[0]
    LOGGER.debug("Model Run: %s", model_run_config)
    sos_model_config = handler.read_sos_model(model_run_config['sos_model'])

    sector_model_objects = []
    for sector_model in sos_model_config['sector_models']:
        sector_model_config = handler.read_sector_model(sector_model)

        absolute_path = os.path.join(args.path,
                                     sector_model_config['path'])
        sector_model_config['path'] = absolute_path

        intervention_files = sector_model_config['interventions']
        intervention_list = []
        for intervention_file in intervention_files:
            interventions = handler.read_interventions(intervention_file)
            intervention_list.extend(interventions)
        sector_model_config['interventions'] = intervention_list

        initial_condition_files = sector_model_config['initial_conditions']
        initial_condition_list = []
        for initial_condition_file in initial_condition_files:
            initial_conditions = handler.read_initial_conditions(initial_condition_file)
            initial_condition_list.extend(initial_conditions)
        sector_model_config['initial_conditions'] = initial_condition_list

        sector_model_builder = SectorModelBuilder(sector_model_config['name'])
        LOGGER.debug("Sector model config: %s", sector_model_config)
        sector_model_builder.construct(sector_model_config)
        sector_model_object = sector_model_builder.finish()

        sector_model_objects.append(sector_model_object)
        LOGGER.debug("Model inputs: %s", sector_model_object.model_inputs.names)

    LOGGER.debug("Sector models: %s", sector_model_objects)
    sos_model_config['sector_models'] = sector_model_objects

    scenario_objects = []
    for scenario in model_run_config['scenarios']:
        LOGGER.debug("Finding data for '%s'", scenario[1])
        scenario_definition = handler.read_scenario_definition(scenario[1])
        scenario_data = handler.read_scenario_data(scenario[1])
        scenario_set = scenario_definition['scenario_set']
        scenario_model_builder = ScenarioModelBuilder(scenario_set)
        scenario_model_builder.construct(scenario_definition, scenario_data,
                                         model_run_config['timesteps'])
        scenario_objects.append(scenario_model_builder.finish())

    LOGGER.debug("Scenario models: %s", [model.name for model in scenario_objects])
    sos_model_config['scenario_sets'] = scenario_objects

    sos_model_builder = SosModelBuilder()
    sos_model_builder.construct(sos_model_config)
    sos_model_object = sos_model_builder.finish()

    LOGGER.debug("Model list: %s", list(sos_model_object.models.keys()))

    model_run_config['sos_model'] = sos_model_object

    return model_run_config


def build_model_run(model_run_config):

    try:
        builder = ModelRunBuilder()
        builder.construct(model_run_config)
        modelrun = builder.finish()
    except AssertionError as error:
        err_type, err_value, err_traceback = sys.exc_info()
        traceback.print_exception(err_type, err_value, err_traceback)
        err_msg = str(error)
        if err_msg:
            LOGGER.error("An AssertionError occurred (%s) see details above.", err_msg)
        else:
            LOGGER.error("An AssertionError occurred, see details above.")
        exit(-1)

    return modelrun


def execute_model_run(args):
    """Runs the model run

    Arguments
    ---------
    """
    model_run_config = get_model_run_definition(args)
    modelrun = build_model_run(model_run_config)

    LOGGER.info("Running model run %s", modelrun.name)
    modelrun.run()

    output_file = args.output_file
    LOGGER.info("Writing results to %s", output_file)
    dump(modelrun.sos_model.results, output_file)
    print("Model run complete")


def parse_arguments():
    """Parse command line arguments

    Returns
    =======
    :class:`argparse.ArgumentParser`

    """
    parser = ArgumentParser(description='Command line tools for smif')
    parser.add_argument('-V', '--version',
                        action='version',
                        version="smif " + smif.__version__,
                        help='show the current version of smif')
    parser.add_argument('-v', '--verbose',
                        action='count',
                        help='show messages: -v to see messages reporting on progress, ' +
                        '-vv to see debug messages.')
    subparsers = parser.add_subparsers(help='available commands')

    # SETUP
    parser_setup = subparsers.add_parser('setup',
                                         help='Setup the project folder')
    parser_setup.set_defaults(func=setup_configuration)
    parser_setup.add_argument('path',
                              help="Path to the project folder")

    # RUN
    parser_run = subparsers.add_parser('run',
                                       help='Run a model')
    parser_run.add_argument('-o', '--output-file',
                            default='results.yaml',
                            help='Output file')
    parser_run.set_defaults(func=execute_model_run)
    parser_run.add_argument('path',
                            help="Path to the main config file")

    return parser


def confirm(prompt=None, response=False):
    """Prompts for a yes or no response from the user

    Arguments
    ---------
    prompt : str, default=None
    response : bool, default=False

    Returns
    -------
    bool
        True for yes and False for no.

    Notes
    -----

    `response` should be set to the default value assumed by the caller when
    user simply types ENTER.

    Examples
    --------

    >>> confirm(prompt='Create Directory?', response=True)
    Create Directory? [y]|n:
    True
    >>> confirm(prompt='Create Directory?', response=False)
    Create Directory? [n]|y:
    False
    >>> confirm(prompt='Create Directory?', response=False)
    Create Directory? [n]|y: y
    True

    """

    if prompt is None:
        prompt = 'Confirm'

    if response:
        prompt = '{} [{}]|{}: '.format(prompt, 'y', 'n')
    else:
        prompt = '{} [{}]|{}: '.format(prompt, 'n', 'y')

    while True:
        ans = input(prompt)
        if not ans:
            return response
        if ans not in ['y', 'Y', 'n', 'N']:
            print('please enter y or n.')
            continue
        if ans in ['y', 'Y']:
            return True
        if ans in ['n', 'N']:
            return False


def main(arguments=None):
    """Parse args and run
    """
    parser = parse_arguments()
    args = parser.parse_args(arguments)

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main(sys.argv[1:])
