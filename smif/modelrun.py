"""The Model Run collects scenarios, timesteps, narratives, and
model colleciton into a package which can be deployed and run.

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
from smif.sos_model import SosModelBuilder


class ModelRun(object):
    """
    """

    counter = 0

    @staticmethod
    def get_id():
        """Increments the class variable to create a unique id for each
        instance of ModelRun
        """
        ModelRun.counter += 1
        return ModelRun.counter

    def __init__(self):

        self._id = ModelRun.get_id()
        self.description = ""
        self.sos_model = None
        self.model_horizon = None
        self.scenario = None
        self.narratives = None
        self.strategy = None
        self.status = 'Ready'

    @property
    def id(self):
        return self._id

    def run(self):
        """Builds all the objects and passes them to the ModelRunner queue for processing

        """
        if self.status == 'Built':
            self.status = 'Running'
            modelrunner = ModelRunner()
            modelrunner.solve_model(self)
            self.status = 'Successful'
        else:
            raise ValueError("Model is not yet built.")

    def build(self, model_config):
        """Constructs the model collection

        Arguments
        ---------
        model_config : dict
        """
        builder = SosModelBuilder()
        builder.construct(model_config)
        self.sos_model = builder.finish()
        self.status = 'Built'


class ModelRunner(object):
    """Runs a ModelRun
    """

    def __init__(self):
        pass

    def solve_model(self, model_run):
        """Solve a ModelRun

        Arguments
        ---------
        model_run : :class:`smif.modelrun.ModelRun`
        """
        model_run.sos_model.run()
