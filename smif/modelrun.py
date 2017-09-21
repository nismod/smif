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


class ModelRun(object):
    """
    """

    def __init__(self):

        self.name = ""
        self.timestamp = None
        self.description = ""
        self.sos_model = None
        self._model_horizon = []

        self.scenarios = {}
        self.narratives = {}
        self.strategies = None
        self.status = 'Empty'

        self.logger = getLogger(__name__)

        self.results = {}

    @property
    def model_horizon(self):
        """Returns the list of timesteps

        Returns
        =======
        list
            A list of timesteps, distinct and sorted in ascending order
        """
        return self._model_horizon

    @model_horizon.setter
    def model_horizon(self, value):
        self._model_horizon = sorted(list(set(value)))

    def run(self):
        """Builds all the objects and passes them to the ModelRunner

        The idea is that this will add ModelRuns to a queue for asychronous
        processing

        """
        self.logger.debug("Running model run %s", self.name)
        if self.status == 'Built':
            self.status = 'Running'
            modelrunner = ModelRunner()
            modelrunner.solve_model(self)
            self.status = 'Successful'
            return modelrunner.results
        else:
            raise ValueError("Model is not yet built.")


class ModelRunner(object):
    """Runs a ModelRun
    """

    def __init__(self):
        self.logger = getLogger(__name__)
        self.results = {}

    def solve_model(self, model_run):
        """Solve a ModelRun

        Arguments
        ---------
        model_run : :class:`smif.modelrun.ModelRun`
        """
        # Initialise each of the sector models
        for model in model_run.sos_model.sector_models:
            model_run.sos_model.models[model].before_model_run()

        # Solve the models over all timesteps
        for timestep in model_run.model_horizon:
            self.logger.debug('Running model for timestep %s', timestep)
            data = self._get_parameter_data(model_run)
            self.logger.debug("Passing parameter data %s into '%s'",
                              data, model_run.sos_model.name)

            self.results[timestep] = model_run.sos_model.simulate(timestep,
                                                                  data)
        return self.results

    def _get_parameter_data(self, model_run):
        """Loads overridden parameter values from narrative/policy files
        """
        data = {}

        return data


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
        self.model_run.timestamp = model_run_config['stamp']
        self._add_timesteps(model_run_config['timesteps'])
        self._add_sos_model(model_run_config['sos_model'])
        self._add_scenarios(model_run_config['scenarios'])
        self._add_narratives(model_run_config['narratives'])

        self.model_run.status = 'Built'

    def finish(self):
        """Returns a configured model run ready for operation

        """
        if self.model_run.status == 'Built':
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
        narratives : dict
            A dictionary of {narrative set: narrative name}, one for each narrative set
        """
        self.model_run.narratives = narratives
