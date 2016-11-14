"""A command line interface to the system of systems framework

This command line interface implements a number of methods.

- `setup` creates a new project folder structure in a location
- `run` performs a simulation of an individual sector model, or the whole
        system of systems model

"""
import logging
import os
import sys
from argparse import ArgumentParser
from smif.controller import Controller

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)

_log_format = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
logging.basicConfig(filename='cli.log',
                    level=logging.DEBUG,
                    format=_log_format,
                    filemode='a')


def setup_project_folder(project_path):
    """Creates folder structure in the target directory

    Arguments
    =========
    project_path : str
        Absolute path to an empty folder

    """
    folder_list = ['config', 'planning', 'models']
    for folder in folder_list:
        folder_path = os.path.join(project_path, folder)
        if os.path.exists(folder_path):
            msg = "{} already exists, skipping...".format(folder_path)
        else:
            msg = "Creating {} folder in {}".format(folder, project_path)
            os.mkdir(folder_path)
        logger.info(msg)


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
    logger.info(msg)


def run_model(args):
    """Runs the model specified in the args.model argument

    """
    if args.model == 'all':
        logger.info("Running the system of systems model")
        controller = Controller(os.getcwd())
        controller.run_sos_model()

    else:
        logger.info("Running the {} sector model".format(args.model))
        model_name = args.model
        controller = Controller(os.getcwd())
        controller.run_sector_model(model_name)


def parse_arguments(list_of_sector_models):
    """

    Arguments
    =========
    args : list
        Command line arguments
    list_of_sector_models : list
        A list of sector model names

    Returns
    =======
    :class:`argparse.ArgumentParser`

    """
    assert isinstance(list_of_sector_models, list)
    list_of_sector_models.append('all')

    parser = ArgumentParser(description='Command line tools for smif')
    parser.set_defaults(modellist=list_of_sector_models)

    subparsers = parser.add_subparsers()

    parser_setup = subparsers.add_parser('setup',
                                         help='Setup the project folder')
    parser_setup.set_defaults(func=setup_configuration)
    parser_setup.add_argument('path',
                              help="Path to the project folder")

    parser_run = subparsers.add_parser('run',
                                       help='Run a model')

    parser_run.add_argument('model',
                            choices=list_of_sector_models,
                            type=str,
                            help='The name of the model to run')
    parser_run.set_defaults(func=run_model)

    return parser


def confirm(prompt=None, response=False):
    """Prompts for a yes or no response from the user

    Arguments
    =========
    prompt : str, default=None
    response : bool, default=False


    Returns
    =======
    bool
        True for yes and False for no.


    Notes
    =====

    `response` should be set to the default value assumed by the caller when
    user simply types ENTER.

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
    list_of_sector_models = ['water_supply', 'solid_waste']
    parser = parse_arguments(list_of_sector_models)
    args = parser.parse_args(arguments)
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main(sys.argv[1:])
