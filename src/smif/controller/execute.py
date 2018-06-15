import logging

from smif.controller.build import build_model_run, get_model_run_definition
from smif.controller.load import load_resolution_sets
from smif.data_layer import DatafileInterface
from smif.modelrun import ModelRunError

LOGGER = logging.getLogger(__name__)


def execute_model_run(model_run_ids, directory, interface='local_binary', warm=False):
    """Runs the model run

    Parameters
    ----------
    modelrun_ids: list
        Modelrun ids that should be executed sequentially
    """
    LOGGER.info("Loading resolution data")
    load_resolution_sets(directory)

    model_run_definitions = []
    for model_run in model_run_ids:
        LOGGER.info("Getting model run definition for '" + model_run + "'")
        model_run_definitions.append(get_model_run_definition(directory, model_run))

    for model_run_config in model_run_definitions:

        LOGGER.info("Build model run from configuration data")
        modelrun = build_model_run(model_run_config)

        LOGGER.info("Running model run %s", modelrun.name)
        store = DatafileInterface(directory, interface)

        try:
            if warm:
                modelrun.run(store, store.prepare_warm_start(modelrun.name))
            else:
                modelrun.run(store)
        except ModelRunError as ex:
            LOGGER.exception(ex)
            exit(1)

        print("Model run '" + modelrun.name + "' complete")
