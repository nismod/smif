"""Execute a single step directly
"""
import logging
import os
import sys

from smif.controller.build import get_model_run_definition
from smif.data_layer import DataHandle
from smif.data_layer.model_loader import ModelLoader
from smif.decision.decision import DecisionManager
from smif.exception import SmifDataNotFoundError


def execute_model_before_step(model_run_id, model_name, store, dry_run=False):
    """Runs model initialisation

    This method
    """
    if dry_run:
        print("    smif before_step {} --model {}".format(model_run_id, model_name))
    else:
        model, data_handle = _get_model_and_handle(store, model_run_id, model_name)

        # before_model_run may not be implemented by all jobs
        if hasattr(model, "before_model_run"):
            model.before_model_run(data_handle)


def execute_model_step(model_run_id, model_name, timestep, decision, store, dry_run=False):
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
    """
    if dry_run:
        print("    smif step {} --model {} --timestep {} --decision {}".format(
              model_run_id, model_name, timestep, decision))
    else:
        model, data_handle = _get_model_and_handle(
            store, model_run_id, model_name, timestep, decision)
        model.simulate(data_handle)


def _get_model_and_handle(store, model_run_id, model_name, timestep=None, decision=None):
    """Helper method to read model and set up appropriate data handle
    """
    try:
        model_run_config = store.read_model_run(model_run_id)
    except SmifDataNotFoundError:
        logging.error(
            "Model run %s not found. Run 'smif list' to see available model runs.",
            model_run_id)
        sys.exit(1)

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
    return model, data_handle


def execute_decision_step(model_run_id, decision, store):
    """Request the next set of decisions from the decision manager, including the bundle of
    timesteps and decision iterations to simulate next.

    Parameters
    ----------
    model_run_id: str
        Modelrun id of overarching model run
    decision: int
        Decision to run - should be 0 to start, or n+1 where n is the max decision executed so
        far
    store: Store
    """
    # get model run with sos model and all models loaded
    # DecisionManager needs a SosModel
    # - uses list of model names to read available interventions per model
    # - passes SosModel to create ResultsHandle
    #   - uses any given Model as object to look at outputs to find appropriate Spec to
    #     read
    model_run = get_model_run_definition(store, model_run_id)

    # decision loop gets and saves decisions into "state" files
    decision_manager = DecisionManager(
        store,
        model_run['timesteps'],
        model_run_id,
        model_run['sos_model'],
        decision
    )
    bundle = next(decision_manager.decision_loop())

    # print report
    print("Got decision bundle")
    print("    decision iterations", bundle['decision_iterations'])
    print("    timesteps", bundle['timesteps'])
    print("Run each model in order with commands like:")
    print("    smif step {} --model <model> --decision <d> --timestep <t>".format(
        model_run_id))
    print("To see a viable order, dry-run the whole model:")
    print("    smif run {} --dry-run".format(model_run_id))
