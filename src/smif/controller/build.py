import logging
import os
import sys
import traceback

from smif.data_layer import DatafileInterface, DataNotFoundError
from smif.model.scenario_model import ScenarioModelBuilder
from smif.model.sector_model import SectorModelBuilder
from smif.model.sos_model import SosModelBuilder
from smif.modelrun import ModelRunBuilder
from smif.parameters import Narrative

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

    sector_model_objects = get_sector_model_objects(
        sos_model_config, handler, model_run_config['timesteps'])
    LOGGER.debug("Sector models: %s", sector_model_objects)
    sos_model_config['sector_models'] = sector_model_objects

    scenario_objects = get_scenario_objects(
        model_run_config['scenarios'], handler)
    LOGGER.debug("Scenario models: %s", [model.name for model in scenario_objects])
    sos_model_config['scenario_sets'] = scenario_objects

    sos_model_builder = SosModelBuilder()
    sos_model_builder.construct(sos_model_config)
    sos_model_object = sos_model_builder.finish()
    model_run_config['sos_model'] = sos_model_object
    LOGGER.debug("Model list: %s", list(sos_model_object.models.keys()))

    strategies = get_strategies(sector_model_objects,
                                model_run_config, handler)
    model_run_config['strategies'] = strategies
    LOGGER.info("Added %s strategies to model run config", len(strategies))

    narrative_objects = get_narratives(handler,
                                       model_run_config['narratives'])
    model_run_config['narratives'] = narrative_objects

    return model_run_config


def get_scenario_objects(scenarios, handler):
    """
    Arguments
    ---------
    scenarios : dict
    handler : smif.data_layer.DataInterface

    Returns
    -------
    list
    """
    scenario_objects = []
    for scenario_set, scenario_name in scenarios.items():
        scenario_definition = handler.read_scenario_definition(scenario_name)
        LOGGER.debug("Scenario definition: %s", scenario_definition)

        scenario_model_builder = ScenarioModelBuilder(scenario_set)
        scenario_model_builder.construct(scenario_definition)
        scenario_objects.append(scenario_model_builder.finish())
    return scenario_objects


def get_sector_model_objects(sos_model_config, handler,
                             timesteps):
    sector_model_objects = []
    for sector_model_name in sos_model_config['sector_models']:
        sector_model_config = handler.read_sector_model(sector_model_name)

        sector_model_builder = SectorModelBuilder(sector_model_name)
        # absolute path to be crystal clear for SectorModelBuilder when loading python class
        sector_model_config['path'] = os.path.normpath(
            os.path.join(handler.base_folder, sector_model_config['path'])
        )
        sector_model_builder.construct(sector_model_config, timesteps)

        sector_model_object = sector_model_builder.finish()
        sector_model_objects.append(sector_model_object)
        LOGGER.debug("Model inputs: %s", sector_model_object.inputs.names)
    return sector_model_objects


def get_strategies(sector_model_objects, model_run_config, handler):

    strategies = []
    initial_conditions = get_initial_conditions_strategies(sector_model_objects)

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
            del strategy['filename']
            strategy['interventions'] = decisions
            LOGGER.info("Added %s pre-specified planning interventions to %s",
                        len(decisions), strategy['model_name'])
            strategies.append(strategy)
    return strategies


def get_initial_conditions_strategies(sector_model_objects):
    """Add pre-specified planning strategy for all initial conditions

    Arguments
    ---------
    sector_model_objects : list
        A list of :class:`~smif.model.SectorModel`

    Returns
    -------
    list
    """
    strategies = []
    for sector_model in sector_model_objects:
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
