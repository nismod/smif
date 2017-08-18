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

from smif.convert.area import get_register as get_region_register
from smif.convert.area import RegionSet
from smif.convert.interval import get_register as get_interval_register
from smif.convert.interval import IntervalSet
from smif.model.sos_model import SosModelBuilder


class ModelRun(object):
    """
    """

    def __init__(self):

        self._name = 0
        self.description = ""
        self.sos_model = None
        self._model_horizon = []

        self.narratives = None
        self.strategies = None
        self.status = 'Empty'

        self.logger = getLogger(__name__)

        # space and time
        self.regions = get_region_register()
        self.intervals = get_interval_register()

    @property
    def name(self):
        """Unique identifier of the ModelRun
        """
        return self._name

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
            model_run.sos_model.simulate(timestep)


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

        self.load_region_sets(config_data['region_sets'])
        self.load_interval_sets(config_data['interval_sets'])

        self._add_sos_model(config_data)

        self.model_run.status = 'Built'

    def finish(self):
        """Returns a configured model run ready for operation

        """
        if self.model_run.status == 'Built':
            return self.model_run
        else:
            raise RuntimeError("Run construct() method before finish().")

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

    def load_region_sets(self, region_sets):
        """Loads the region sets into the system-of-system model

        Parameters
        ----------
        region_sets: list
            A dict, where key is the name of the region set, and the value
            the data
        """
        assert isinstance(region_sets, dict)

        region_set_definitions = region_sets.items()
        if len(region_set_definitions) == 0:
            msg = "No region sets have been defined"
            self.logger.warning(msg)
        for name, data in region_set_definitions:
            msg = "Region set data is not a list"
            assert isinstance(data, list), msg
            self.model_run.regions.register(RegionSet(name, data))

    def load_interval_sets(self, interval_sets):
        """Loads the time-interval sets into the system-of-system model

        Parameters
        ----------
        interval_sets: list
            A dict, where key is the name of the interval set, and the value
            the data
        """
        interval_set_definitions = interval_sets.items()
        if len(interval_set_definitions) == 0:
            msg = "No interval sets have been defined"
            self.logger.warning(msg)

        for name, data in interval_set_definitions:
            interval_set = IntervalSet(name, data)
            self.model_run.intervals.register(interval_set)
