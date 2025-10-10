"""Test SubProcessRunScheduler and SerialJobScheduler"""

from copy import copy
from unittest.mock import Mock, patch

import networkx
from pytest import fixture, raises

import smif
from smif.controller.job import SerialJobScheduler
from smif.model import ModelOperation, SectorModel


class EmptySectorModel(SectorModel):
    def simulate(self, data):
        return data


class TestSerialJobScheduler:
    @fixture
    def job_graph(self):
        G = networkx.DiGraph()
        a_model = EmptySectorModel("a")

        G.add_node(
            "a",
            model=a_model,
            operation=ModelOperation.BEFORE_MODEL_RUN,
            modelrun_name="test",
            current_timestep=1,
            timesteps=[1],
            decision_iteration=0,
        )
        b_model = EmptySectorModel("b")
        G.add_node(
            "b",
            model=b_model,
            operation=ModelOperation.SIMULATE,
            modelrun_name="test",
            current_timestep=1,
            timesteps=[1],
            decision_iteration=0,
        )
        G.add_edge("a", "b")
        return G

    @fixture
    def scheduler(self, empty_store):
        empty_store.write_model_run(
            {
                "name": "test",
                "narratives": {},
                "scenarios": {},
                "sos_model": "test_sos_model",
                "timesteps": [],
            }
        )
        empty_store.write_sos_model(
            {
                "name": "test_sos_model",
                "scenario_dependencies": [],
                "model_dependencies": [],
            }
        )
        model = {
            "description": "",
            "inputs": [],
            "outputs": [],
            "parameters": [],
            "path": smif.model.__file__,
            "classname": "Model",
        }
        for name in ["a", "b", "c"]:
            m = copy(model)
            m["name"] = name
            empty_store.write_model(m)

        scheduler = SerialJobScheduler(empty_store)
        return scheduler

    def test_add(self, job_graph, scheduler):
        job_id, err = scheduler.add(job_graph)

        print(err)
        assert err is None
        assert scheduler.get_status(job_id)["status"] == "done"

    def test_default_status(self):
        scheduler = SerialJobScheduler()
        assert scheduler.get_status(0)["status"] == "unstarted"

    def test_add_cyclic(self, job_graph, scheduler):
        job_graph.add_edge("b", "a")
        job_id, err = scheduler.add(job_graph)

        assert isinstance(err, NotImplementedError)
        assert scheduler.get_status(job_id)["status"] == "failed"

    def test_kill_fails(self, job_graph, scheduler):
        job_id, err = scheduler.add(job_graph)

        assert err is None
        with raises(NotImplementedError):
            scheduler.kill(job_id)

    def test_unknown_operation(self, job_graph, scheduler):
        model = EmptySectorModel("c")

        job_graph.add_node(
            "c",
            model=model,
            operation="unknown_operation",
            modelrun_name="test",
            current_timestep=1,
            timesteps=[1],
            decision_iteration=0,
        )
        job_id, err = scheduler.add(job_graph)

        assert isinstance(err, ValueError)
        assert scheduler.get_status(job_id)["status"] == "failed"
