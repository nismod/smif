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

    sector_model_objects = []
    for sector_model in sos_model_config['sector_models']:
        sector_model_config = handler.read_sector_model(sector_model)

        absolute_path = os.path.join(directory,
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

    strategies = []
    for strategy in model_run_config['strategies']:
        if strategy['strategy'] == 'pre-specified-planning':
            interventions = handler.read_strategies(strategy['filename'])
            del strategy['filename']
            strategy['interventions'] = interventions
            LOGGER.debug("Added %s pre-specified planning interventions to %s",
                         len(interventions), strategy['model_name'])
        strategies.append(strategy)
    sos_model_config['strategies'] = strategies

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
