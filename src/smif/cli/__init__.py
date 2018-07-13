# -*- coding: utf-8 -*-
"""A command line interface to the system of systems framework

This command line interface implements a number of methods.

- `setup` creates an example project with the recommended folder structure
- `run` performs a simulation of an individual sector model, or the whole system
        of systems model
- `validate` performs a validation check of the configuration file
- `app` runs the graphical user interface, opening in a web browser

Folder structure
----------------

When configuring a project for the CLI, the folder structure below should be
used.  In this example, there is one system-of-systems model, combining a water
supply and an energy demand model::

    /config
        project.yaml
        /sector_models
            energy_demand.yml
            water_supply.yml
        /sos_models
            energy_water.yml
        /sos_model_runs
            run_to_2050.yml
            short_test_run.yml
            ...
    /data
        /initial_conditions
            reservoirs.yml
        /interval_definitions
            annual_intervals.csv
        /interventions
            water_supply.yml
        /narratives
            high_tech_dsm.yml
        /region_definitions
            /oxfordshire
                regions.geojson
            /uk_nations_shp
                regions.shp
        /scenarios
            population.csv
            raininess.csv
        /water_supply
            /initial_system
    /models
        energy_demand.py
        water_supply.py
    /planning
        expected_to_2020.yaml
        national_infrastructure_pipeline.yml

The sector model implementations can be installed independently of the model run
configuration. The paths to python wrapper classes (implementing SectorModel)
should be specified in each ``sector_model/*.yml`` configuration.

The project.yaml file specifies the metadata shared by all elements of the
project; ``sos_models`` specify the combinations of ``sector_models`` and
``scenarios`` while individual ``model_runs`` specify the scenario, strategy
and narrative combinations to be used in each run of the models.

"""
from __future__ import print_function

try:
    import _thread
except ImportError:
    import thread as _thread

import logging
import logging.config
import os
import socket
import errno
import pkg_resources

try:
    import win32api
    USE_WIN32 = True
except ImportError:
    USE_WIN32 = False

from argparse import ArgumentParser

import smif
import smif.cli.log

from smif.controller import copy_project_folder, execute_model_run, Scheduler
from smif.http_api import create_app
from smif.data_layer import DatafileInterface


__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


LOGGER = logging.getLogger(__name__)


def list_model_runs(args):
    """List the model runs defined in the config
    """
    handler = DatafileInterface(args.directory)
    model_run_configs = handler.read_sos_model_runs()
    for run in model_run_configs:
        print(run['name'])


def run_model_runs(args):
    """Run the model runs as requested. Check if results exist and asks
    user for permission to overwrite

    Parameters
    ----------
    args
    """
    if args.batchfile:
        with open(args.modelrun, 'r') as f:
            model_run_ids = f.read().splitlines()
    else:
        model_run_ids = [args.modelrun]

    execute_model_run(model_run_ids, args.directory, args.interface, args.warm)


def _run_server(args):
    app_folder = pkg_resources.resource_filename('smif', 'app/dist')
    app = create_app(
        static_folder=app_folder,
        template_folder=app_folder,
        data_interface=DatafileInterface(args.directory),
        scheduler=Scheduler()
    )

    port = 5000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.bind(("0.0.0.0", port))
            break
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                port += 1
            else:
                raise Exception('Smif app server error')
    s.close()

    print("    Opening smif app\n")
    print("    Copy/paste this URL into your web browser to connect:")
    print("        http://localhost:" + str(port) + "\n")
    # add flush to ensure that text is printed before server thread starts
    print("    Close your browser then type Control-C here to quit.", flush=True)
    app.run(host='0.0.0.0', port=port, threaded=True)


def run_app(args):
    """Run the client/server application

    Parameters
    ----------
    args
    """
    # avoid one of two error messages from 'forrtl error(200)' when running
    # on windows cmd - seems related to scipy's underlying Fortran
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

    if USE_WIN32:
        # Set handler for CTRL-C. Necessary to avoid `forrtl: error (200):
        # program aborting...` crash on CTRL-C if we're runnging from Windows
        # cmd.exe
        def handler(dw_ctrl_type, hook_sigint=_thread.interrupt_main):
            """Handler for CTRL-C interrupt
            """
            if dw_ctrl_type == 0:  # CTRL-C
                hook_sigint()
                return 1  # don't chain to the next handler
            return 0  # chain to the next handler
        win32api.SetConsoleCtrlHandler(handler, 1)

    # Create backend server process
    _run_server(args)


def setup_project_folder(args):
    """Setup a sample project
    """
    copy_project_folder(args.directory)


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
    parser_setup.set_defaults(func=setup_project_folder)
    parser_setup.add_argument('-d', '--directory',
                              default='.',
                              help="Path to the project directory")

    # LIST
    parser_list = subparsers.add_parser('list',
                                        help='List available model runs')
    parser_list.set_defaults(func=list_model_runs)
    parser_list.add_argument('-d', '--directory',
                             default='.',
                             help="Path to the project directory")

    # APP
    parser_app = subparsers.add_parser('app',
                                       help='Open smif app')
    parser_app.set_defaults(func=run_app)
    parser_app.add_argument('-d', '--directory',
                            default='.',
                            help="Path to the project directory")

    # RUN
    parser_run = subparsers.add_parser('run',
                                       help='Run a model')
    parser_run.set_defaults(func=run_model_runs)
    parser_run.add_argument('-i', '--interface',
                            default='local_binary',
                            choices=['local_csv', 'local_binary'],
                            help="Select the data interface (default: %(default)s)")
    parser_run.add_argument('-w', '--warm',
                            action='store_true',
                            help="Use intermediate results from the last modelrun \
                                  and continue from where it had left")
    parser_run.add_argument('-d', '--directory',
                            default='.',
                            help="Path to the project directory")
    parser_run.add_argument('-b', '--batchfile',
                            action='store_true',
                            help="Use a batchfile instead of a modelrun name (a \
                                  list of modelrun names)")
    parser_run.add_argument('modelrun',
                            help="Name of the model run to run")

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
