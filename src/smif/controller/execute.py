import logging
import os
import sys

from smif.controller.build import build_model_run, get_model_run_definition
from smif.data_layer import DataHandle
from smif.data_layer.model_loader import ModelLoader
from smif.exception import SmifDataNotFoundError, SmifModelRunError
from smif.model import ModelOperation


def execute_model_run(model_run_ids, store, warm=False):
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

    for model_run_config in model_run_definitions:

        logging.info("Build model run from configuration data")
        modelrun = build_model_run(model_run_config)

        logging.info("Running model run %s", modelrun.name)

        try:
            if warm:
                modelrun.run(store, store.prepare_warm_start(modelrun.name))
            else:
                modelrun.run(store)
        except SmifModelRunError as ex:
            logging.exception(ex)
            exit(1)

        print("Model run '%s' complete" % modelrun.name)
        sys.stdout.flush()


def execute_model_step(model_run_id, model_name, timestep, decision, store,
                       operation=ModelOperation.SIMULATE):
    """Runs a single step of a model run

    This method is designed to be the single place where smif actually calls wrapped models.

    Loading model wrapper implementations (via ModelLoader) is also deferred to this method, so
    that the smif scheduler environment need not have all the python dependencies of all
    wrapped models.

    For example, in a scheduled containerised environment, the model run can be configured and
    set running (`smif run modelrun_name`) in one environment, and individual models can run
    (`smif run modelrun_name --model model_name --timestep 2050`) in their own environments.

    Parameters
    ----------
    model_run_id: str
        Modelrun id of overarching model run
    model_name: str
        Model to run
    timestep: int
        Timestep to run
    decision: int
        Decision to run
    store: Store
    operation: ModelOperation, optional
        Model operation to execute, either before_model_run or simulate
    """
    try:
        model_run_config = store.read_model_run(model_run_id)
    except SmifDataNotFoundError:
        logging.error(
            "Model run %s not found. Run 'smif list' to see available model runs.",
            model_run_id)
        exit(1)

    loader = ModelLoader()
    sector_model_config = store.read_model(model_name)
    # absolute path to be crystal clear for ModelLoader when loading python class
    sector_model_config['path'] = os.path.normpath(
        os.path.join(store.model_base_folder, sector_model_config['path'])
    )
    model = loader.load(sector_model_config)

    # DataHandle reads
    # - model run from store to find narratives and scenarios selected
    # - sos model from store to find dependencies and parameters
    # all in order to resolve *input* data locations and *parameter* defaults and values
    data_handle = DataHandle(
        store=store,
        model=model,
        modelrun_name=model_run_id,
        current_timestep=timestep,
        timesteps=model_run_config['timesteps'],
        decision_iteration=decision
    )

    if operation is ModelOperation.BEFORE_MODEL_RUN:
        # before_model_run may not be implemented by all jobs
        if hasattr(model, "before_model_run"):
            model.before_model_run(data_handle)

    elif operation is ModelOperation.SIMULATE:
        # run the model
        model.simulate(data_handle)

    else:
        raise ValueError("Unrecognised operation: {}".format(operation))
