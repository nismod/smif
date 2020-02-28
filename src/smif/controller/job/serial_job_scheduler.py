"""Job Schedulers are used to run job graphs.

Runs a job graph by calling execute_model_step for each operation in order
"""
import itertools
import logging
import traceback
from collections import defaultdict

import networkx
from smif.controller.execute_step import (execute_model_before_step,
                                          execute_model_step)
from smif.model import ModelOperation


class SerialJobScheduler(object):
    """Run JobGraphs produced by a :class:`~smif.controller.modelrun.ModelRun`
    """
    def __init__(self, store=None):
        self._status = defaultdict(lambda: 'unstarted')
        self._id_counter = itertools.count()
        self.logger = logging.getLogger(__name__)
        self.store = store

    def add(self, job_graph, dry_run=False):
        """Add a JobGraph to the SerialJobScheduler and run directly

        Arguments
        ---------
        job_graph: :class:`networkx.graph`
        dry_run: boolean, optional
            If True, print job steps without running
        """
        job_graph_id = self._next_id()
        try:
            self._run(job_graph, job_graph_id, dry_run)
        except Exception as ex:
            self._status[job_graph_id] = 'failed'
            traceback.print_exc()
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

    def _run(self, job_graph, job_graph_id, dry_run=False):
        """Run a job graph
        - sort the jobs into a single list
        - unpack model, data_handle and operation from each node
        """
        try:
            self.logger.profiling_start(
                'SerialJobScheduler._run()', 'graph_' + str(job_graph_id))
        except AttributeError:
            self.logger.info('START SerialJobScheduler._run():graph_%s', job_graph_id)

        self._status[job_graph_id] = 'running'

        for job_node_id, job in self._get_run_order(job_graph):
            self._run_job(job_node_id, job, dry_run)

        self._status[job_graph_id] = 'done'
        try:
            self.logger.profiling_stop(
                'SerialJobScheduler._run()', 'graph_' + str(job_graph_id))
        except AttributeError:
            self.logger.info(
                'STOP SerialJobScheduler._run():graph_%s', job_graph_id)

    def _run_job(self, job_node_id, job, dry_run=False):
        self.logger.info("Job %s", job_node_id)  # Call root logger to satisfy CLI test
        try:
            self.logger.profiling_start('SerialJobScheduler._run()', 'job_' + job_node_id)
        except AttributeError:
            self.logger.info('START SerialJobScheduler._run():job_%s', job_node_id)

        if job['operation'] == ModelOperation.SIMULATE:
            execute_model_step(
                job['modelrun_name'],
                job['model'].name,
                job['current_timestep'],
                job['decision_iteration'],
                self.store,
                dry_run
            )
        elif job['operation'] == ModelOperation.BEFORE_MODEL_RUN:
            execute_model_before_step(
                job['modelrun_name'],
                job['model'].name,
                self.store,
                dry_run
            )
        else:
            raise ValueError("Model operation not recognised", job)

        try:
            self.logger.profiling_stop('SerialJobScheduler._run()', 'job_' + job_node_id)
        except AttributeError:
            self.logger.info('STOP SerialJobScheduler._run():job_%s', job_node_id)

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
