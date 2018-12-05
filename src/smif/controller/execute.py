import logging
import sys

from smif.controller.build import build_model_run, get_model_run_definition
from smif.exception import SmifModelRunError

LOGGER = logging.getLogger(__name__)


def execute_model_run(model_run_ids, store, warm=False):
    """Runs the model run

    Parameters
    ----------
    modelrun_ids: list
        Modelrun ids that should be executed sequentially
    """
    model_run_definitions = []
    for model_run in model_run_ids:
        LOGGER.info("Getting model run definition for '%s'", model_run)
        model_run_definitions.append(get_model_run_definition(store, model_run))

    for model_run_config in model_run_definitions:

        LOGGER.info("Build model run from configuration data")
        modelrun = build_model_run(model_run_config)

        LOGGER.info("Running model run %s", modelrun.name)

        try:
            if warm:
                modelrun.run(store, store.prepare_warm_start(modelrun.name))
            else:
                modelrun.run(store)
        except SmifModelRunError as ex:
            LOGGER.exception(ex)
            exit(1)

        print("Model run '%s' complete" % modelrun.name)
        sys.stdout.flush()
