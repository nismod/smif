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
import datetime
import logging
import logging.config
import os
import pkg_resources
import re
import shutil
import sys
import time
import traceback
import webbrowser
from argparse import ArgumentParser
from multiprocessing import Process
from threading import Timer

import smif
from smif.data_layer import DatafileInterface, DataNotFoundError
from smif.convert.register import Register
from smif.convert.area import get_register as get_region_register
from smif.convert.area import RegionSet
from smif.convert.interval import get_register as get_interval_register
from smif.convert.interval import IntervalSet
from smif.convert.unit import get_register as get_unit_register
from smif.http_api import create_app
from smif.parameters import Narrative
from smif.modelrun import ModelRunBuilder, ModelRunError
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
UNITS = get_unit_register()


def setup_project_folder(args):
    """Creates folder structure in the target directory

    Parameters
    ----------
    args
    """
    _recursive_overwrite('smif', 'sample_project', args.directory)
    if args.directory == ".":
        dirname = "the current directory"
    else:
        dirname = args.directory
    LOGGER.info("Created sample project in %s", dirname)


def _recursive_overwrite(pkg, src, dest):
    if pkg_resources.resource_isdir(pkg, src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        contents = pkg_resources.resource_listdir(pkg, src)
        for item in contents:
            _recursive_overwrite(pkg,
                                 os.path.join(src, item),
                                 os.path.join(dest, item))
    else:
        filename = pkg_resources.resource_filename(pkg, src)
        shutil.copyfile(filename, dest)


def load_region_sets(handler):
    """Loads the region sets into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`

    """
    region_definitions = handler.read_region_definitions()
    for region_def in region_definitions:
        region_name = region_def['name']
        LOGGER.info("Reading in region definition %s", region_name)
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
        LOGGER.info("Reading in interval definition %s", interval_name)
        interval_data = handler.read_interval_definition_data(interval_name)
        interval_set = IntervalSet(interval_name, interval_data)
        INTERVALS.register(interval_set)


def load_units(handler):
    """Loads the units into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`
    """
    unit_file = handler.read_units_file_name()
    if unit_file is not None:
        LOGGER.info("Loading units in from %s", unit_file)
        UNITS.register(unit_file)


def get_model_run_definition(args):
    """Builds the model run

    Returns
    -------
    dict
        The complete sos_model_run configuration dictionary with contained
        ScenarioModel, SosModel and SectorModel objects

    """
    handler = DatafileInterface(args.directory)
    Register.data_interface = handler
    load_region_sets(handler)
    load_interval_sets(handler)
    load_units(handler)

    try:
        model_run_config = handler.read_sos_model_run(args.modelrun)
    except DataNotFoundError:
        LOGGER.error("Model run %s not found. Run 'smif list' to see available model runs.",
                     args.modelrun)
        exit(-1)

    LOGGER.info("Running %s", model_run_config['name'])
    LOGGER.debug("Model Run: %s", model_run_config)
    sos_model_config = handler.read_sos_model(model_run_config['sos_model'])

    sector_model_objects = []
    for sector_model in sos_model_config['sector_models']:
        sector_model_config = handler.read_sector_model(sector_model)

        absolute_path = os.path.join(args.directory,
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
        # LOGGER.debug("Sector model config: %s", sector_model_config)
        sector_model_builder.construct(sector_model_config,
                                       model_run_config['timesteps'])
        sector_model_object = sector_model_builder.finish()

        sector_model_objects.append(sector_model_object)
        LOGGER.debug("Model inputs: %s", sector_model_object.inputs.names)

    LOGGER.debug("Sector models: %s", sector_model_objects)
    sos_model_config['sector_models'] = sector_model_objects

    scenario_objects = []
    for scenario_set, scenario_name in model_run_config['scenarios'].items():
        scenario_definition = handler.read_scenario_definition(scenario_name)
        LOGGER.debug("Scenario definition: %s", scenario_definition)

        scenario_model_builder = ScenarioModelBuilder(scenario_set)
        scenario_model_builder.construct(scenario_definition)
        scenario_objects.append(scenario_model_builder.finish())

    LOGGER.debug("Scenario models: %s", [model.name for model in scenario_objects])
    sos_model_config['scenario_sets'] = scenario_objects

    sos_model_builder = SosModelBuilder()
    sos_model_builder.construct(sos_model_config)
    sos_model_object = sos_model_builder.finish()

    LOGGER.debug("Model list: %s", list(sos_model_object.models.keys()))

    model_run_config['sos_model'] = sos_model_object
    narrative_objects = get_narratives(handler,
                                       model_run_config['narratives'])
    model_run_config['narratives'] = narrative_objects

    return model_run_config


def get_narratives(handler, narrative_config):
    """Load the narrative data from the sos model run configuration

    Arguments
    ---------
    handler: :class:`smif.data_layer.DataInterface`
    narrative_config: dict
        A dict with keys as narrative_set names and values as narrative names

    Returns
    -------
    list
        A list of :class:`smif.parameter.Narrative` objects populated with
        data

    """
    narrative_objects = []
    for narrative_set, narrative_names in narrative_config.items():
        LOGGER.info("Loading narrative data for narrative set '%s'",
                    narrative_set)
        for narrative_name in narrative_names:
            LOGGER.debug("Adding narrative entry '%s'", narrative_name)
            definition = handler.read_narrative_definition(narrative_name)
            narrative = Narrative(
                narrative_name,
                definition['description'],
                narrative_set
            )
            narrative.data = handler.read_narrative_data(narrative_name)
            narrative_objects.append(narrative)
    return narrative_objects


def list_model_runs(args):
    """List the model runs defined in the config
    """
    handler = DatafileInterface(args.directory)
    model_run_configs = handler.read_sos_model_runs()
    for run in model_run_configs:
        print(run['name'])


def build_model_run(model_run_config):
    """Builds the model run

    Arguments
    ---------
    model_run_config: dict
        A valid model run configuration dict with objects

    Returns
    -------
    `smif.modelrun.ModelRun`
    """
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

    Parameters
    ----------
    args
    """
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%dT%H%M%S')
    
    LOGGER.info("Getting model run definition")
    model_run_config = get_model_run_definition(args)

    LOGGER.info("Build model run from configuration data")
    modelrun = build_model_run(model_run_config)

    LOGGER.info("Running model run %s with timestamp %s", modelrun.name, timestamp)
    store = DatafileInterface(args.directory, args.interface, timestamp)

    try:
        if args.warm:
            modelrun.run(store, store.prepare_warm_start(modelrun.name))
        else:
            modelrun.run(store)
    except ModelRunError as ex:
        LOGGER.exception(ex)
        exit(1)

    print("Model run complete")


def make_get_data_interface(args):
    """Use args to make a function returning a suitable DataInterface
    """
    def getter():
        """Return a DataInterface
        """
        return DatafileInterface(args.directory)
    return getter


def _open_browser():
    print(" * Opening page in browser...")
    webbrowser.open("http://localhost:5000/")


def _run_server(args):
    app_folder = pkg_resources.resource_filename('smif', 'app/dist')
    get_data_interface = make_get_data_interface(args)
    app = create_app(
        static_folder=app_folder,
        template_folder=app_folder,
        get_data_interface=get_data_interface
    )
    app.run(host='localhost', port=5000, threaded=True)


def run_app(args):
    """Run the client/server application

    Parameters
    ----------
    args
    """
    print("Opening smif application")

    # avoid one of two error messages from 'forrtl error(200)' when running
    # on windows cmd - seems related to scipy's underlying Fortran
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

    # Create backend server process
    server = Process(target=_run_server, args=(args,))

    try:
        print(" * Type CTRL-C to stop")
        server.start()
        Timer(2, _open_browser).start()
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(" * Stopping...")
        server.terminate()
        server.join()


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
    parser_run.set_defaults(func=execute_model_run)
    parser_run.add_argument('-i', '--interface',
                            default='local_binary',
                            choices=['local_csv', 'local_binary'],
                            help="Select the data interface (default: %(default)s)")
    parser_run.add_argument('-w', '--warm',
                            action='store_true',
                            help="Use intermediate results from the last modelrun and continue from where it had left")
    parser_run.add_argument('-d', '--directory',
                            default='.',
                            help="Path to the project directory")
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


if __name__ == '__main__':
    main(sys.argv[1:])
