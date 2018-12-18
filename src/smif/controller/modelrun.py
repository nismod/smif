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

import networkx as nx
from smif.controller.scheduler import JobScheduler
from smif.decision.decision import DecisionManager
from smif.exception import SmifModelRunError, SmifTimestepResolutionError
from smif.metadata import RelativeTimestep
from smif.model import ModelOperation, ScenarioModel


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
        scenarios = set(self.scenarios)
        model_scenarios = set(scenario.name for scenario in self.sos_model.scenario_models)
        missing_scenarios = scenarios - model_scenarios
        if missing_scenarios:
            raise SmifModelRunError("ScenarioSets {} are selected in the ModelRun "
                                    "configuration but not found in the SosModel "
                                    "configuration".format(missing_scenarios))

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
        self.logger.profiling_start('modelrun.run', self.name)

        if self.status == 'Built':
            if not self.model_horizon:
                raise SmifModelRunError("No timesteps specified for model run")
            if warm_start_timestep:
                idx = self.model_horizon.index(warm_start_timestep)
                self.model_horizon = self.model_horizon[idx:]
            self.status = 'Running'
            modelrunner = ModelRunner()
            modelrunner.solve_model(self, store)
            self.status = 'Successful'
        else:
            raise SmifModelRunError("Model is not yet built.")

        self.logger.profiling_stop('modelrun.run', self.name)


class ModelRunner(object):
    """The ModelRunner orchestrates the simulation of a SoSModel over decision iterations and
    timesteps as provided by a DecisionManager.
    """
    def __init__(self):
        self.logger = getLogger(__name__)

    def solve_model(self, model_run, store):
        """Solve a ModelRun

        This method steps through the model horizon, building
        a job graph and submitting this to the scheduler
        at each decision loop.

        Arguments
        ---------
        model_run : :class:`smif.controller.modelrun.ModelRun`
        store : :class:`smif.data_layer.Store`
        """
        # Solve the model run: decision loop generates a series of bundles of independent
        # decision iterations, each with a number of timesteps to run
        self.logger.debug("Solving the models over all timesteps: %s", model_run.model_horizon)

        # Initialise the decision manager (and hence decision modules)
        self.logger.debug("Initialising the decision manager")
        decision_manager = DecisionManager(store,
                                           model_run.model_horizon,
                                           model_run.name,
                                           model_run.sos_model)

        # Initialise the job scheduler
        self.logger.debug("Initialising the job scheduler")
        job_scheduler = JobScheduler()
        job_scheduler.store = store

        for bundle in decision_manager.decision_loop():
            # each iteration is independent at this point, so the following loop is a
            # candidate for running in parallel
            job_graph = self.build_job_graph(model_run, bundle)

            job_id, err = job_scheduler.add(job_graph)
            self.logger.debug("Running job %s", job_id)
            if err is not None:
                status = job_scheduler.get_status(job_id)
                self.logger.debug("Job %s %s", job_id, status['status'])
                raise err

    def build_job_graph(self, model_run, bundle):
        """ Build a job graph

        Build and return the job graph for an entire bundle, including before_model_run jobs
        when the models were not yet initialised.

        Constraints:
        - Bundle must have keys: 'decision_iterations' and 'timesteps'
        - Running a bundle runs each (decision iteration, timestep) pair specified by the
          combinations of decision iterations and timesteps
        - (decision iteration, timestep) pairs must be unique over an entire model run
        - In a single bundle, timesteps must be a consecutive subset of the model horizon
          timesteps

        The first timestep in each decision iteration of a bundle is either:
        - the first timestep in the model horizon and initialised from the model run starting
          point with scenario data and initial-timestep interventions only
        - or another timestep, picking up from where some previous (timestep, decision
          iteration) left off, which is explicitly included in the bundle.

        If a bundle's timesteps start from a timestep after the first one in the model horizon,
        the bundle must provide 'decision_links', and bundle must start from the very next
        timestep available in the model horizon.

        Jobs need to be able to identify a point to pick up from, namely the (timestep,
        decision iteration) which identifies the immediately preceding simulation state.

        E.g. request running first two timesteps::

            {
                'decision_iterations': [0, 1],
                'timesteps': [0, 1]
            }

        Run first two timesteps again, with an updated decision::

            {
                'decision_iterations': [1, 2],
                'timesteps': [0, 1]
            }

        Results meet decision requirements, so run next two timesteps, linking this bundle's
        decision iterations to previous decision iterations::

            {
                'decision_iterations': [3, 4],
                'timesteps': [2, 3],
                'decision_links': {3: 1, 4: 2}
            }


        Arguments
        ---------
        model_run: :class:`smif.controller.modelrun.ModelRun` bundle: :class:`dict`

        Returns
        -------
        :class:`networkx.Graph` A populated job graph with edges showing dependencies between
            different operations and timesteps
        """
        job_graph = nx.DiGraph()

        # Solve the model run: decision loop generates a series of bundles of independent
        # decision iterations, each with a number of timesteps to run
        for decision_iteration in bundle['decision_iterations']:
            self.logger.info('Running decision iteration %s', decision_iteration)

            for timestep_index, timestep in enumerate(bundle['timesteps']):
                self.logger.info('Running timestep %s', timestep)
                # one simulate job node per model
                job_graph.add_nodes_from(
                    self._make_simulate_job_nodes(
                        model_run.name,
                        model_run.sos_model.models,
                        decision_iteration,
                        timestep,
                        model_run.model_horizon
                    )
                )
                # edges to match within-timestep dependencies
                job_graph.add_edges_from(
                    self._make_current_simulate_job_edges(
                        model_run.name,
                        model_run.sos_model.dependencies,
                        timestep,
                        decision_iteration
                    )
                )

                # connect any between-timestep dependencies
                if timestep_index == 0:
                    # first timestep in bundle
                    try:
                        # connect to outputs from a previous bundle
                        relative = RelativeTimestep.PREVIOUS
                        previous_timestep = relative.resolve_relative_to(
                            timestep, model_run.model_horizon)
                        previous_decision_iteration = \
                            bundle['decision_links'][decision_iteration]
                        job_graph.add_edges_from(
                            self._make_between_bundle_previous_simulate_job_edges(
                                model_run.name,
                                model_run.sos_model.dependencies,
                                timestep,
                                previous_timestep,
                                decision_iteration,
                                previous_decision_iteration
                            )
                        )
                    except SmifTimestepResolutionError:
                        # no previous timestep, use scenarios to provide initial intertimestep
                        # dependenciess
                        job_graph.add_edges_from(
                            self._make_initial_previous_simulate_job_edges(
                                model_run.name,
                                model_run.sos_model.dependencies,
                                timestep,
                                decision_iteration
                            )
                        )

                else:
                    # subsequent timestep in a bundle - connect to previous timestep
                    previous_timestep = bundle['timesteps'][timestep_index - 1]
                    job_graph.add_edges_from(
                        self._make_within_bundle_previous_simulate_job_edges(
                            model_run.name,
                            model_run.sos_model.dependencies,
                            timestep,
                            previous_timestep,
                            decision_iteration
                        )
                    )

        if not model_run.initialised:
            # one before_model_run job per model
            self.logger.info("Initialising each of the sector models")
            job_graph.add_nodes_from(
                self._make_before_model_run_job_nodes(
                    model_run.name,
                    model_run.sos_model.models,
                    model_run.model_horizon
                )
            )
            # must run before any simulate jobs
            for decision_iteration in bundle['decision_iterations']:
                for timestep in bundle['timesteps']:
                    job_graph.add_edges_from(
                        self._make_before_model_run_job_edges(
                            model_run.name,
                            model_run.sos_model.models,
                            timestep,
                            decision_iteration
                        )
                    )
            model_run.initialised = True

        if not nx.is_directed_acyclic_graph(job_graph):
            raise NotImplementedError(
                "SosModel dependency graphs must not contain within-timestep cycles")

        return job_graph

    @staticmethod
    def _make_before_model_run_job_nodes(modelrun_name, models, horizon):
        return [
            (
                ModelRunner._make_job_id(
                    modelrun_name, model.name, ModelOperation.BEFORE_MODEL_RUN),
                {
                    'model': model,
                    'modelrun_name': modelrun_name,
                    'current_timestep': None,
                    'timesteps': horizon,
                    'decision_iteration': None,
                    'operation': ModelOperation.BEFORE_MODEL_RUN
                }
            )
            for model in models
        ]

    @staticmethod
    def _make_before_model_run_job_edges(modelrun_name, models, timestep, decision_iteration):
        edges = []
        for model in models:
            from_id = ModelRunner._make_job_id(
                modelrun_name, model.name, ModelOperation.BEFORE_MODEL_RUN)
            to_id = ModelRunner._make_job_id(
                modelrun_name, model.name, ModelOperation.SIMULATE, timestep,
                decision_iteration)
            edges.append((from_id, to_id))
        return edges

    @staticmethod
    def _make_simulate_job_nodes(modelrun_name, models, decision_iteration, timestep, horizon):
        return [
            (
                ModelRunner._make_job_id(
                    modelrun_name, model.name, ModelOperation.SIMULATE, timestep,
                    decision_iteration),
                {
                    'model': model,
                    'modelrun_name': modelrun_name,
                    'current_timestep': timestep,
                    'timesteps': horizon,
                    'decision_iteration': decision_iteration,
                    'operation': ModelOperation.SIMULATE
                }
            )
            for model in models
        ]

    @staticmethod
    def _make_current_simulate_job_edges(modelrun_name, dependencies, timestep,
                                         decision_iteration):
        edges = []
        for dependency in dependencies:
            if dependency.timestep != RelativeTimestep.PREVIOUS:
                from_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.source_model.name, ModelOperation.SIMULATE,
                    timestep, decision_iteration)
                to_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.sink_model.name, ModelOperation.SIMULATE,
                    timestep, decision_iteration)
                edges.append((from_id, to_id))
        return edges

    @staticmethod
    def _make_within_bundle_previous_simulate_job_edges(modelrun_name, dependencies, timestep,
                                                        previous_timestep, decision_iteration):
        edges = []
        for dependency in dependencies:
            if dependency.timestep == RelativeTimestep.PREVIOUS:
                from_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.source_model.name, ModelOperation.SIMULATE,
                    previous_timestep, decision_iteration)
                to_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.sink_model.name, ModelOperation.SIMULATE,
                    timestep, decision_iteration)
                edges.append((from_id, to_id))
        return edges

    @staticmethod
    def _make_between_bundle_previous_simulate_job_edges(modelrun_name, dependencies, timestep,
                                                         previous_timestep, decision_iteration,
                                                         previous_decision_iteration):
        edges = []
        for dependency in dependencies:
            if dependency.timestep == RelativeTimestep.PREVIOUS:
                from_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.source_model.name, ModelOperation.SIMULATE,
                    previous_timestep, previous_decision_iteration)
                to_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.sink_model.name, ModelOperation.SIMULATE,
                    timestep, decision_iteration)
                edges.append((from_id, to_id))
        return edges

    @staticmethod
    def _make_initial_previous_simulate_job_edges(modelrun_name, dependencies, timestep,
                                                  decision_iteration):
        edges = []
        for dependency in dependencies:
            if isinstance(dependency.source_model, ScenarioModel):
                from_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.source_model.name, ModelOperation.SIMULATE,
                    timestep, decision_iteration)
                to_id = ModelRunner._make_job_id(
                    modelrun_name, dependency.sink_model.name, ModelOperation.SIMULATE,
                    timestep, decision_iteration)
                edges.append((from_id, to_id))
        return edges

    @staticmethod
    def _make_job_id(modelrun_name, model_name, operation, timestep=None,
                     decision_iteration=None):
        if operation == ModelOperation.BEFORE_MODEL_RUN:
            id_ = '%s_%s_%s' % (modelrun_name, operation.value, model_name)
        else:
            id_ = '%s_%s_%s_%s_%s' % (
                modelrun_name, operation.value, timestep, decision_iteration, model_name)
        return id_


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
        self.model_run.initialised = False
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
