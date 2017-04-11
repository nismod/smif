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
from smif.sos_model import SosModelBuilder
from smif.data_layer.load import dump
from smif.data_layer.sos_model_config import SosModelReader
from smif.data_layer.sector_model_config import SectorModelReader
from smif.data_layer.validate import VALIDATION_ERRORS

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


def run_model(args):
    """Runs the model specified in the args.model argument

    """
    model_config = validate_config(args)

    try:
        builder = SosModelBuilder()
        builder.construct(model_config)
        sos_model = builder.finish()
    except AssertionError as error:
        err_type, err_value, err_traceback = sys.exc_info()
        traceback.print_exception(err_type, err_value, err_traceback)
        err_msg = str(error)
        if len(err_msg) > 0:
            LOGGER.error("An AssertionError occurred (%s) see details above.", err_msg)
        else:
            LOGGER.error("An AssertionError occurred, see details above.")
        exit(-1)

    if args.model == 'all':
        LOGGER.info("Running the system of systems model")
        sos_model.run()
    else:
        LOGGER.info("Running the %s sector model", args.model)
        model_name = args.model
        sos_model.run_sector_model(model_name)

    output_file = args.output_file
    LOGGER.info("Writing results to %s", output_file)
    dump(sos_model.results, output_file)
    print("Model run complete")


def validate_config(args):
    """Validates the model configuration file against the schema

    Arguments
    =========
    args :
        Parser arguments

    """
    config_path = os.path.abspath(args.path)

    if not os.path.exists(config_path):
        LOGGER.error("The model configuration file '%s' was not found", config_path)
        exit(-1)
    else:
        try:
            # read system-of-systems config
            reader = SosModelReader(config_path)
            reader.load()

            model_config = reader.data
            config_basepath = os.path.dirname(config_path)

            # read sector model data+config
            model_config['sector_model_data'] = read_sector_model_data(
                config_basepath,
                model_config['sector_model_config'])
        except Exception as error:
            # should not raise error, so exit
            log_validation_errors()
            LOGGER.exception("Unexpected error validating config: %s", error)
            exit(-1)

        log_validation_errors()
        if len(VALIDATION_ERRORS) > 0:
            print("The model configuration was invalid")
            exit(-1)
        else:
            print("The model configuration was valid")

        return model_config


def log_validation_errors():
    """Log validation errors
    """
    for error in VALIDATION_ERRORS:
        LOGGER.error(error)


def path_to_abs(relative_root, path):
    """Return an absolute path, given a possibly-relative path
    and the relative root"""
    if os.path.isabs(path):
        return os.path.normpath(path)
    else:
        return os.path.normpath(os.path.join(relative_root, path))


def read_sector_model_data(config_basepath, config):
    """Read sector-specific data from the sector config folders
    """
    data = []

    for model_config in config:
        # read from dir relative to main model config file
        config_dir = path_to_abs(config_basepath, model_config['config_dir'])
        path = path_to_abs(config_basepath, model_config['path'])
        initial_conditions_paths = list(map(
            lambda path: path_to_abs(config_basepath, path),
            model_config['initial_conditions']
        ))
        interventions_paths = list(map(
            lambda path: path_to_abs(config_basepath, path),
            model_config['interventions']
        ))

        # read each sector model config+data
        reader = SectorModelReader({
            "model_name": model_config['name'],
            "model_path": path,
            "model_classname": model_config['classname'],
            "model_config_dir": config_dir,
            "initial_conditions": initial_conditions_paths,
            "interventions": interventions_paths
        })
        try:
            reader.load()
            data.append(reader.data)
        except FileNotFoundError as error:
            LOGGER.error("%s: %s", model_config['name'], error)
            raise ValueError("missing sector model configuration")

    return data


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
    subparsers = parser.add_subparsers()

    # VALIDATE
    help_msg = 'Validate the model configuration file'
    parser_validate = subparsers.add_parser('validate',
                                            help=help_msg)
    parser_validate.set_defaults(func=validate_config)
    parser_validate.add_argument('path',
                                 help="Path to the main config file")

    # SETUP
    parser_setup = subparsers.add_parser('setup',
                                         help='Setup the project folder')
    parser_setup.set_defaults(func=setup_configuration)
    parser_setup.add_argument('path',
                              help="Path to the project folder")

    # RUN
    parser_run = subparsers.add_parser('run',
                                       help='Run a model')
    parser_run.add_argument('-m', '--model',
                            default='all',
                            help='The name of the model to run')
    parser_run.add_argument('-o', '--output-file',
                            default='results.yaml',
                            help='Output file')
    parser_run.set_defaults(func=run_model)
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
