"""Schedulers are used to run models.

The defaults provided allow model runs to be scheduled as subprocesses,
or individual models to be called in series.

Future implementations may interface with common schedulers to queue
up models to run in parallel and/or distributed.
"""
import itertools
import logging
import subprocess
from collections import defaultdict
from datetime import datetime

import networkx
from smif.model import ModelOperation


class ModelRunScheduler(object):
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
                'smif ' +
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


class JobScheduler(object):
    """Run JobGraphs produced by a :class:`~smif.controller.modelrun.ModelRun`
    """
    def __init__(self):
        self._status = defaultdict(lambda: 'unstarted')
        self._id_counter = itertools.count()
        self.logger = logging.getLogger(__name__)

    def add(self, job_graph):
        """Add a JobGraph to the JobScheduler and run directly

        Arguments
        ---------
        job_graph: :class:`networkx.graph`
        """
        job_graph_id = self._next_id()
        try:
            self._run(job_graph, job_graph_id)
        except Exception as ex:
            self._status[job_graph_id] = 'failed'
            return job_graph_id, ex

        return job_graph_id, None

    def kill(self, job_graph_id):
        """Kill a job_graph that is already running - not implemented

        Parameters
        ----------
        job_graph_id: int
        """
        raise NotImplementedError

    def get_status(self, job_graph_id):
        """Get job graph status

        Parameters
        ----------
        job_graph_id: int

        Returns
        -------
        dict: A message containing the status

        Notes
        -----
        Possible statuses:

        unstarted:
            Job graph has not yet started
        running:
            Job graph is running
        done:
            Job graph was completed succesfully
        failed:
            Job graph completed running with an exit code
        """
        return {'status': self._status[job_graph_id]}

    def _run(self, job_graph, job_graph_id):
        """Run a job graph
        - sort the jobs into a single list
        - unpack model, data_handle and operation from each node
        """
        self._status[job_graph_id] = 'running'

        for job_node_id, job in self._get_run_order(job_graph):
            self.logger.info("Job %s", job_node_id)
            model = job['model']
            data_handle = job['data_handle']
            operation = job['operation']
            if operation is ModelOperation.BEFORE_MODEL_RUN:
                # before_model_run may not be implemented by all jobs
                try:
                    model.before_model_run(data_handle)
                except AttributeError as ex:
                    self.logger.warning(ex)

            elif operation is ModelOperation.SIMULATE:
                model.simulate(data_handle)

            else:
                raise ValueError("Unrecognised operation: {}".format(operation))

        self._status[job_graph_id] = 'done'

    def _next_id(self):
        return next(self._id_counter)

    @staticmethod
    def _get_run_order(graph):
        """Returns a list of jobs in a runnable order.

        Returns
        -------
        list
            A list of job nodes
        """
        try:
            # topological sort gives a single list from directed graph,
            # ignoring opportunities to run independent models in parallel
            run_order = networkx.topological_sort(graph)

            # list of Models (typically ScenarioModel and SectorModel)
            ordered_jobs = [(run, graph.nodes[run]) for run in run_order]
        except networkx.NetworkXUnfeasible:
            raise NotImplementedError("Job graphs must not contain cycles")

        return ordered_jobs
