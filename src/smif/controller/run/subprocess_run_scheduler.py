"""Schedulers are used to run models.

The defaults provided allow model runs to be scheduled as subprocesses,
or individual models to be called in series.

Future implementations may interface with common schedulers to queue
up models to run in parallel and/or distributed.

Calls smif run ... on the selected model in a sub process, where ...
are the options set in the app for info/debug messages, warm start
and output format.
"""
import subprocess
from collections import defaultdict
from datetime import datetime


class SubProcessRunScheduler(object):
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
        if self._status[model_run_name] != 'running':
            self._output[model_run_name] = ''
            self._status[model_run_name] = 'queing'

            smif_call = (
                'smif run ' +
                '-'*(int(args['verbosity']) > 0) + 'v'*int(args['verbosity']) + ' ' +
                model_run_name + ' ' +
                '-d' + ' ' + args['directory'] + ' ' +
                '-w'*args['warm_start'] + ' '*args['warm_start'] +
                '-i' + ' ' + args['output_format']
            )

            self._process[model_run_name] = subprocess.Popen(
                smif_call,
                shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            format_args = {
                'model_run_name': model_run_name,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pid': str(self._process[model_run_name].pid),
                'smif_call': smif_call,
                'colour': "\x1b[1;34m",
                'reset': "\x1b[0m",
                'space': " \x1b"
            }
            format_str = """\
                {colour}Modelrun{reset} {model_run_name}
                {colour}Time{reset}     {datetime}
                {colour}PID{reset}      {pid}
                {colour}Command{reset}  {smif_call}
                """
            format_str.replace(" ", "{space}")
            output = format_str.format(**format_args)
            output += "-" * 100 + "\n"
            self._output[model_run_name] = output
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

    def get_scheduler_type(self):
        return "default"

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
