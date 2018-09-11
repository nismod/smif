"""The Model Run collects scenarios, timesteps, narratives, and
model collection into a package which can be built and passed to
the ModelRunner to run.

The ModelRunner is responsible for running a ModelRun, including passing
in the correct data to the model between timesteps and calling to the
DecisionManager to obtain decisions.

ModeRun has attributes:
- id
- description
- sosmodel
- timesteps
- scenarios
- narratives
- strategy
- status

"""
from logging import getLogger

from smif.data_layer import DataHandle
from smif.decision.decision import DecisionManager

import networkx
import itertools

class ModelRun(object):
    """Collects timesteps, scenarios , narratives and a SosModel together

    Attributes
    ----------
    name: str
        The unique name of the model run
    timestamp: :class:`datetime.datetime`
        An ISO8601 compatible timestamp of model run creation time
    description: str
        A friendly description of the model run
    sos_model: :class:`smif.model.sos_model.SosModel`
        The contained SosModel
    scenarios: dict
        For each scenario set, a mapping to a valid scenario within that set
    narratives: list
        A list of :class:`smif.parameters.Narrative` objects
    strategies: dict
    status: str
    logger: logging.Logger
    results: dict
    """

    def __init__(self):
        self.name = ""
        self.timestamp = None
        self.description = ""
        self.sos_model = None
        self._model_horizon = []

        self.scenarios = {}
        self.narratives = []
        self.strategies = None
        self.status = 'Empty'

        self.logger = getLogger(__name__)

        self.results = {}

    def as_dict(self):
        """Serialises :class:`smif.controller.modelrun.ModelRun`

        Returns a dictionary definition of a ModelRun which is
        equivalent to that required by :class:`smif.controller.modelrun.ModelRunBuilder`
        to construct a new model run

        Returns
        -------
        dict
        """
        config = {
            'name': self.name,
            'description': self.description,
            'stamp': self.timestamp,
            'timesteps': self._model_horizon,
            'sos_model': self.sos_model.name,
            'scenarios': self.scenarios,
            'narratives': self.narratives,
            'strategies': self.strategies
        }
        return config

    def validate(self):
        """Validate that this ModelRun has been set up with sufficient data
        to run
        """
        for scenario in self.scenarios:
            if scenario not in self.sos_model.scenario_models.keys():
                raise ModelRunError("ScenarioSet '{}' is selected in the ModelRun "
                                    "configuration but not found in the SosModel "
                                    "configuration".format(scenario))

    @property
    def model_horizon(self):
        """Returns the list of timesteps

        Returns
        =======
        list
            A list of timesteps, distinct and sorted in ascending order
        """
        return self._model_horizon.copy()

    @model_horizon.setter
    def model_horizon(self, value):
        self._model_horizon = sorted(list(set(value)))

    def run(self, store, warm_start_timestep=None):
        """Builds all the objects and passes them to the ModelRunner

        The idea is that this will add ModelRuns to a queue for asychronous
        processing

        """
        self.logger.debug("Running model run %s", self.name)
        if self.status == 'Built':
            if not self.model_horizon:
                raise ModelRunError("No timesteps specified for model run")
            if warm_start_timestep:
                idx = self.model_horizon.index(warm_start_timestep)
                self.model_horizon = self.model_horizon[idx:]
            self.status = 'Running'
            modelrunner = ModelRunner()
            modelrunner.solve_model(self, store)
            self.status = 'Successful'
        else:
            raise ModelRunError("Model is not yet built.")


class ModelRunner(object):
    """The ModelRunner orchestrates the simulation of a SoSModel over decision iterations and
    timesteps as provided by a DecisionManager.
    """
    def __init__(self):
        self.logger = getLogger(__name__)

    def solve_model(self, model_run, store):
        """Solve a ModelRun

        This method first calls :func:`smif.model.SosModel.before_model_run`
        with parameter data, then steps through the model horizon, calling
        :func:`smif.model.SosModel.simulate` with parameter data at each
        timestep.

        Arguments
        ---------
        model_run : :class:`smif.controller.modelrun.ModelRun`
        store : :class:`smif.data_layer.DataInterface`
        """
        # Initialise each of the sector models
        self.logger.info("Initialising each of the sector models")
        data_handle = DataHandle(
            store=store,
            modelrun_name=model_run.name,
            current_timestep=None,
            timesteps=model_run.model_horizon,
            model=model_run.sos_model
        )

        job_graph = model_run.sos_model.before_model_run(data_handle)
        connect_from = [node for node in job_graph.nodes if job_graph.out_degree(node) == 0] 

        # Initialise the decision manager (and hence decision modules)
        self.logger.debug("Initialising the decision manager")
        decision_manager = DecisionManager(store,
                                           model_run.model_horizon,
                                           model_run.name,
                                           model_run.sos_model.name)

        # Solve the model run: decision loop generates a series of bundles of independent
        # decision iterations, each with a number of timesteps to run
        self.logger.debug("Solving the models over all timesteps: %s", model_run.model_horizon)
        for bundle in decision_manager.decision_loop():
            # each iteration is independent at this point, so the following loop is a
            # candidate for running in parallel
            for iteration, timesteps in bundle.items():
                self.logger.info('Running decision iteration %s', iteration)

                # Each timestep *might* be able to be run in parallel - until we have an
                # explicit way of declaring inter-timestep dependencies
                for timestep in timesteps:
                    self.logger.info('Running timestep %s', timestep)

                    # Write decisions for current timestep to state

                    data_handle = DataHandle(
                        store=store,
                        modelrun_name=model_run.name,
                        current_timestep=timestep,
                        timesteps=model_run.model_horizon,
                        model=model_run.sos_model,
                        decision_iteration=iteration
                    )
                    decision_manager.get_decision(data_handle, timestep, iteration)

                    sub_job_graph = model_run.sos_model.simulate(data_handle)
                    connect_to = [node for node in sub_job_graph.nodes if sub_job_graph.in_degree(node) == 0]

                    job_graph = networkx.compose(job_graph, sub_job_graph)
                    for from_node, to_node in itertools.product(connect_from, connect_to):
                        job_graph.add_edge(from_node, to_node)

        return job_graph


class ModelRunBuilder(object):
    """Builds the ModelRun object from the configuration
    """
    def __init__(self):
        self.model_run = ModelRun()
        self.logger = getLogger(__name__)

    def construct(self, model_run_config):
        """Set up the whole ModelRun

        Arguments
        ---------
        model_run_config : dict
            A valid model run configuration dictionary
        """
        self.model_run.name = model_run_config['name']
        self.model_run.description = model_run_config['description']
        self.model_run.timestamp = model_run_config['stamp']
        self._add_timesteps(model_run_config['timesteps'])
        self._add_sos_model(model_run_config['sos_model'])
        self._add_scenarios(model_run_config['scenarios'])
        self._add_narratives(model_run_config['narratives'])
        self._add_strategies(model_run_config['strategies'])

        self.model_run.status = 'Built'

    def validate(self):
        """Check and/or assert that the modelrun is correctly set up
        - should raise errors if invalid
        """
        assert self.model_run is not None, "Sector model not loaded"
        self.model_run.validate()
        return True

    def finish(self):
        """Returns a configured model run ready for operation

        """
        if self.model_run.status == 'Built':
            self.validate()
            return self.model_run
        else:
            raise RuntimeError("Run construct() method before finish().")

    def _add_sos_model(self, sos_model_object):
        """

        Arguments
        ---------
        sos_model_object : smif.model.sos_model.SosModel
        """
        self.model_run.sos_model = sos_model_object

    def _add_timesteps(self, timesteps):
        """Set the timesteps of the system-of-systems model

        Arguments
        ---------
        timesteps : list
            A list of timesteps
        """
        self.logger.info("Adding timesteps to model run")
        self.model_run.model_horizon = timesteps

    def _add_scenarios(self, scenarios):
        """

        Arguments
        ---------
        scenarios : dict
            A dictionary of {scenario set: scenario name}, one for each scenario set
        """
        self.model_run.scenarios = scenarios

    def _add_narratives(self, narratives):
        """

        Arguments
        ---------
        narratives : list
            A list of smif.parameters.Narrative objects
        """
        self.model_run.narratives = narratives

    def _add_strategies(self, strategies):
        """

        Arguments
        ---------
        narratives : list
            A list of strategies
        """
        self.model_run.strategies = strategies


class ModelRunError(Exception):
    """Raise when model run requirements are not satisfied
    """
    pass
