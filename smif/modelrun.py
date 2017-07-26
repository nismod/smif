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
from smif.sos_model import SosModelBuilder


class ModelRun(object):
    """
    """

    def __init__(self):

        self._name = 0
        self.description = ""
        self.sos_model = None
        self.model_horizon = None
        self.scenarios = None
        self.narratives = None
        self.strategy = None
        self.status = 'Empty'

        self.logger = getLogger(__name__)

    @property
    def name(self):
        """Unique identifier of the ModelRun
        """
        return self._name

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
        else:
            raise ValueError("Model is not yet built.")


class ModelRunner(object):
    """Runs a ModelRun
    """

    def __init__(self):
        self.logger = getLogger(__name__)

    def solve_model(self, model_run):
        """Solve a ModelRun

        Arguments
        ---------
        model_run : :class:`smif.modelrun.ModelRun`
        """
        for timestep in model_run.model_horizon:
            self.logger.debug('Running model for timestep %s', timestep)
            model_run.sos_model.run(timestep)


class ModelRunBuilder(object):
    """Builds the ModelRun object from the configuration
    """
    def __init__(self):
        self.model_run = ModelRun()
        self.logger = getLogger(__name__)

    def construct(self, config_data):
        """Set up the whole ModelRun

        Parameters
        ----------
        config_data : dict
            A valid system-of-systems model configuration dictionary
        """
        self._add_timesteps(config_data['timesteps'])
        self._add_sos_model(config_data)

    def finish(self):
        """Returns a configured model run ready for operation

        """
        return self.model_run

    def _add_sos_model(self, config_data):
        """
        """
        builder = SosModelBuilder()
        builder.construct(config_data, self.model_run.model_horizon)
        self.model_run.sos_model = builder.finish()

    def _add_timesteps(self, timesteps):
        """Set the timesteps of the system-of-systems model

        Parameters
        ----------
        timesteps : list
            A list of timesteps
        """
        self.logger.info("Adding timesteps to model run")
        self.model_run.model_horizon = timesteps
