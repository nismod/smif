import subprocess
from collections import defaultdict


class Scheduler(object):
    """The scheduler can run instances of smif as a subprocess
    and can provide information whether the modelrun is running,
    is done or has failed.
    """
    def __init__(self):
        self._status = defaultdict(lambda: 'unknown')
        self._process = {}
        self._output = {}
        self._err = {}
        self.lock = False

    def add(self, model_run_name, args):
        """Add a model_run to the Modelrun scheduler.

        Parameters
        ----------
        model_run_name: str
            Name of the modelrun
        args: dict
            Arguments for the command-line interface

        Exception
        ---------
        Exception
            When the modelrun was already started

        Notes
        -----
        There is no queuing mechanism implemented, each `add`
        will directly start a subprocess. This means that it
        is possible to run multiple modelruns concurrently.
        This may cause conflicts, it depends on the
        implementation whether a certain sector model / wrapper
        touches the filesystem or other shared resources.
        """
        if self._status[model_run_name] is not 'running':
            self._output[model_run_name] = ""
            self._output[model_run_name] = ""
            self._process[model_run_name] = subprocess.Popen(
                'smif -v run' + ' ' + model_run_name + ' ' + '-d' + ' ' + args['directory'],
                shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self._status[model_run_name] = 'running'
        else:
            raise Exception('Model is already running.')

    def get_status(self, model_run_name):
        """Get the status from the Modelrun scheduler.

        Parameters
        ----------
        model_run_name: str
            Name of the modelrun

        Returns
        -------
        dict: A message containing the status, command-line
        output and error that can be directly sent back over
        the http api.

        Notes
        -----
        Possible status:

        unknown:
            Model run was not started
        queing:
            Model run is waiting to be executed
        running:
            Model run is running
        done:
            Model run was completed succesfully
        failed:
            Model run completed running with an exit code
        """
        if self._status[model_run_name] == 'unknown':
            return {
                'status': 'unknown'
            }
        elif self._status[model_run_name] == 'running':
            if self.lock == False:
                self.lock = True
                for line in iter(self._process[model_run_name].stdout.readline, b''):
                    self._output[model_run_name] += line.decode()
                    self._process[model_run_name].stdout.flush()
                self.lock = False

            if self._process[model_run_name].poll() == 0:
                self._status[model_run_name] = 'done'
            elif self._process[model_run_name].poll() == 1:
                self._status[model_run_name] = 'failed'

            return {
                'status': 'running',
                'output': self._output[model_run_name]
            }
        elif self._status[model_run_name] == 'done':
            return {
                'status': 'done',
                'output': self._output[model_run_name],
            }
        elif self._status[model_run_name] == 'failed':
            return {
                'status': 'failed',
                'output': self._output[model_run_name]
            }