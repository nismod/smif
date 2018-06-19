""" Implements methods for loading, building and executing model runs
and a Scheduler that runs instances of smif as a subprocess
"""
from subprocess import PIPE, Popen


class Scheduler(object):

    def __init__(self):
        self._process = {}
        self._output = {}
        self._err = {}

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
        """
        if model_run_name not in self._process or self._process[model_run_name].poll() is None:
            self._process[model_run_name] = Popen(
                ['smif', 'run', model_run_name, '-d', args['directory']],
                stdout=PIPE, stderr=PIPE
            )
        else:
            raise Exception('Modelrun was already added.')

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
        running:
            Model run is running
        done:
            Model run was completed succesfully
        failed:
            Model run completed running with an exit code
        """
        if model_run_name not in self._process:
            return {
                'status': 'unknown'
            }
        elif self._process[model_run_name].poll() is None:
            return {
                'status': 'running',
            }
        else:
            if model_run_name not in self._output:
                self._output[model_run_name], self._err[model_run_name] = \
                    self._process[model_run_name].communicate()

            if self._process[model_run_name].returncode == 0:
                return {
                    'status': 'done',
                    'output': self._output[model_run_name].decode('utf-8'),
                    'err': self._err[model_run_name].decode('utf-8')
                }
            else:
                return {
                    'status': 'failed',
                    'output': self._output[model_run_name].decode('utf-8'),
                    'err': self._err[model_run_name].decode('utf-8')
                }
