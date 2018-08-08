import logging
import os
import sys
import traceback

from smif.data_layer import DatafileInterface, DataNotFoundError
from smif.data_layer.model_loader import ModelLoader
from smif.model.scenario_model import ScenarioModel
from smif.model.sos_model import SosModel
from smif.modelrun import ModelRunBuilder

LOGGER = logging.getLogger(__name__)


def get_model_run_definition(directory, modelrun):
    """Builds the model run

    Arguments
    ---------
    directory : str
        Path to the project directory
    modelrun : str
        Name of the model run to run

    Returns
    -------
    dict
        The complete sos_model_run configuration dictionary with contained
        ScenarioModel, SosModel and SectorModel objects

    """
    handler = DatafileInterface(directory)
    try:
        model_run_config = handler.read_sos_model_run(modelrun)
    except DataNotFoundError:
        LOGGER.error("Model run %s not found. Run 'smif list' to see available model runs.",
                     modelrun)
        exit(-1)

    LOGGER.info("Running %s", model_run_config['name'])
    LOGGER.debug("Model Run: %s", model_run_config)
    sos_model_config = handler.read_sos_model(model_run_config['sos_model'])

    sector_models = get_sector_models(sos_model_config['sector_models'], handler)
    LOGGER.debug("Sector models: %s", sector_models)

    scenario_models = get_scenario_models(model_run_config['scenarios'], handler)
    LOGGER.debug("Scenario models: %s", [model.name for model in scenario_models])

    sos_model = SosModel.from_dict(sos_model_config, sector_models + scenario_models)
    model_run_config['sos_model'] = sos_model
    LOGGER.debug("Model list: %s", list(sos_model.models.keys()))

    strategies = get_strategies(sector_models, model_run_config, handler)
    model_run_config['strategies'] = strategies
    LOGGER.info("Added %s strategies to model run config", len(strategies))

    return model_run_config


def get_scenario_models(scenarios, handler):
    """Read in ScenarioModels

    Arguments
    ---------
    scenarios : dict
    handler : smif.data_layer.DataInterface

    Returns
    -------
    list of ScenarioModel
    """
    scenario_models = []
    for scenario_name in scenarios.values():
        scenario_definition = handler.read_scenario_definition(scenario_name)
        LOGGER.debug("Scenario definition: %s", scenario_definition)

        scenario_model = ScenarioModel.from_dict(scenario_definition)
        scenario_models.append(scenario_model)
    return scenario_models


def get_sector_models(sector_model_names, handler):
    """Read and build SectorModels

    Arguments
    ---------
    sector_model_names : list of str
    handler : smif.data_layer.DataInterface

    Returns
    -------
    list of SectorModel implementations
    """
    sector_models = []
    loader = ModelLoader()
    for sector_model_name in sector_model_names:
        sector_model_config = handler.read_sector_model(sector_model_name)

        # absolute path to be crystal clear for ModelLoader when loading python class
        sector_model_config['path'] = os.path.normpath(
            os.path.join(handler.base_folder, sector_model_config['path'])
        )
        sector_model = loader.load(sector_model_config)
        sector_models.append(sector_model)
        LOGGER.debug("Model inputs: %s", sector_model.inputs.keys())
    return sector_models


def get_strategies(sector_models, model_run_config, handler):

    strategies = []
    initial_conditions = get_initial_conditions_strategies(sector_models)

    strategies.extend(initial_conditions)

    pre_spec_strategies = get_pre_specified_planning_strategies(
        model_run_config, handler)
    strategies.extend(pre_spec_strategies)

    return strategies


def get_pre_specified_planning_strategies(model_run_config, handler):
    """Build pre-specified planning strategies for future investments

    Arguments
    ---------
    model_run_config : dict
    handler : smif.data_layer.DataInterface
        An instance of the data interface

    Returns
    -------
    list
    """
    strategies = []
    for strategy in model_run_config['strategies']:
        if strategy['strategy'] == 'pre-specified-planning':
            decisions = handler.read_strategies(strategy['filename'])
            if decisions is None:
                decisions = []
            del strategy['filename']
            strategy['interventions'] = decisions
            LOGGER.info("Added %s pre-specified planning interventions to %s",
                        len(decisions), strategy['model_name'])
            strategies.append(strategy)
    return strategies


def get_initial_conditions_strategies(sector_models):
    """Add pre-specified planning strategy for all initial conditions

    Arguments
    ---------
    sector_models : list
        A list of :class:`~smif.model.SectorModel`

    Returns
    -------
    list
    """
    strategies = []
    for sector_model in sector_models:
        if sector_model.initial_conditions:
            strategy = {}
            strategy['model_name'] = sector_model.name
            strategy['description'] = 'historical_decisions'
            strategy['strategy'] = 'pre-specified-planning'
            strategy['interventions'] = sector_model.initial_conditions
            LOGGER.info("Added %s pre-specified historical decisions to %s",
                        len(sector_model.initial_conditions),
                        strategy['model_name'])
            strategies.append(strategy)
    return strategies


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
