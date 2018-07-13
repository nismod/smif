import subprocess
from collections import defaultdict
from datetime import datetime


class Scheduler(object):
    """The scheduler can run instances of smif as a subprocess
    and can provide information whether the modelrun is running,
    is done or has failed.
    """
    def __init__(self):
        self._status = defaultdict(lambda: 'unstarted')
        self._process = {}
        self._output = defaultdict(str)
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

            smif_call = (
                'exec smif ' +
                '-'*(int(args['verbosity']) > 0) + 'v'*int(args['verbosity']) +
                ' run' + ' ' + model_run_name + ' ' +
                '-d' + ' ' + args['directory'] + ' ' +
                '-w'*args['warm_start'] + ' '*args['warm_start'] +
                '-i' + ' ' + args['output_format']
            )

            self._process[model_run_name] = subprocess.Popen(
                smif_call,
                shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self._output[model_run_name] = "\x1b[1;34mModelrun \x1b \x1b[0m" + \
                                           model_run_name + "\n"
            self._output[model_run_name] += "\x1b[1;34mTime \x1b \x1b " \
                                            "\x1b \x1b \x1b \x1b[0m" + \
                                            datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
            self._output[model_run_name] += "\x1b[1;34mPID" + " \x1b"*7 + "[0m" + \
                                            str(self._process[model_run_name].pid) + "\n"
            self._output[model_run_name] += "\x1b[1;34mCommand" + " \x1b"*3 + "[0m" + \
                                            smif_call + "\n"

            self._output[model_run_name] += "-" * 100 + "\n"
            self._status[model_run_name] = 'running'
        else:
            raise Exception('Model is already running.')

    def kill(self, model_run_name):
        """ Kill a Modelrun that is already running

        Parameters
        ----------
        model_run_name: str
            Name of the modelrun
        """
        if self._status[model_run_name] == 'running':
            self._process[model_run_name].kill()
            self._status[model_run_name] = 'stopped'

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

        unstarted:
            Model run was not started
        queing:
            Model run is waiting to be executed
        running:
            Model run is running
        stopped:
            Model run was stopped (killed) by user
        done:
            Model run was completed succesfully
        failed:
            Model run completed running with an exit code
        """
        if self._status[model_run_name] == 'running':
            if self.lock is False:
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
            'status': self._status[model_run_name],
            'output': self._output[model_run_name]
        }
