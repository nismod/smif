"""Execute a model run - discover and schedule all steps in order
"""
import logging
import sys

from smif.controller.build import build_model_run, get_model_run_definition
from smif.controller.job import SerialJobScheduler
from smif.exception import SmifModelRunError


def execute_model_run(model_run_ids, store, warm=False, dry=False):
    """Runs the model run

    Parameters
    ----------
    modelrun_ids: list
        Modelrun ids that should be executed sequentially
    """
    model_run_definitions = []
    for model_run in model_run_ids:
        logging.info("Getting model run definition for '%s'", model_run)
        model_run_definitions.append(get_model_run_definition(store, model_run))

    logging.debug("Initialising the job scheduler")
    job_scheduler = SerialJobScheduler(store=store)

    for model_run_config in model_run_definitions:

        logging.info("Build model run from configuration data")
        modelrun = build_model_run(model_run_config)

        logging.info("Running model run %s", modelrun.name)

        if dry:
            print("Dry run, stepping through model run without execution:")
            print("    smif decide {}".format(modelrun.name))

        try:
            if warm:
                modelrun.run(store, job_scheduler, store.prepare_warm_start(modelrun.name),
                             dry_run=dry)
            else:
                modelrun.run(store, job_scheduler, dry_run=dry)
        except SmifModelRunError as ex:
            logging.exception(ex)
            sys.exit(1)

        if not dry:
            print("Model run '%s' complete" % modelrun.name)
        sys.stdout.flush()
