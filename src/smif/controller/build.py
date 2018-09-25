import logging
import os
import sys
import traceback

from smif.controller.modelrun import ModelRunBuilder
from smif.data_layer import DatafileInterface
from smif.data_layer.model_loader import ModelLoader
from smif.exception import SmifDataNotFoundError
from smif.model.scenario_model import ScenarioModel
from smif.model.sos_model import SosModel

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
        The complete model_run configuration dictionary with contained
        ScenarioModel, SosModel and SectorModel objects

    """
    handler = DatafileInterface(directory)
    try:
        model_run_config = handler.read_model_run(modelrun)
    except SmifDataNotFoundError:
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

    model_run_config['strategies'] = handler.read_strategies(model_run_config['name'])
    LOGGER.debug("Strategies: %s", [s['type'] for s in model_run_config['strategies']])

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
    for scenario_name, variant_name in scenarios.items():
        scenario_definition = handler.read_scenario(scenario_name)

        # assign variant name to definition
        scenario_definition['scenario'] = variant_name

        # rename provides => outputs
        scenario_definition['outputs'] = scenario_definition['provides']
        del scenario_definition['provides']

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


def build_model_run(model_run_config):
    """Builds the model run

    Arguments
    ---------
    model_run_config: dict
        A valid model run configuration dict with objects

    Returns
    -------
    `smif.controller.modelrun.ModelRun`
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
