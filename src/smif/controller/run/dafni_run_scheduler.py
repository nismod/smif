"""Schedulers are used to run models.

The defaults provided allow model runs to be scheduled as subprocesses, or individual models to
be called in series.

Future implementations may interface with common schedulers to queue up models to run in
parallel and/or distributed.

The DAFNIRunScheduler currently only works while connected to the RAL vpn.

Posts all of the model run information to the DAFNI API so that a DAFNI worker can start the
model run.
"""
import json
import os
import time
from collections import defaultdict

import requests

from minio import Minio
from ruamel.yaml import YAML  # type: ignore

if "BACKEND_NISMOD_MINIO_SECRETS_FILE" in os.environ:
    MINIO_CREDENTIALS_FILE = os.environ['BACKEND_NISMOD_MINIO_SECRETS_FILE']
else:
    MINIO_CREDENTIALS_FILE = ""
if "BACKEND_SECRET_KEY" in os.environ:
    SECRET_KEY = os.environ['BACKEND_SECRET_KEY']
else:
    SECRET_KEY = ""
if "BACKEND_ACCESS_KEY" in os.environ:
    ACCESS_KEY = os.environ['BACKEND_ACCESS_KEY']
else:
    ACCESS_KEY = ""
if "BACKEND_JOBSUBMISSION_API" in os.environ:
    JOBSUBMISSION_API_URL = os.environ['BACKEND_JOBSUBMISSION_API']
else:
    JOBSUBMISSION_API_URL = ""
if "BACKEND_MINIO_IP" in os.environ:
    MINIO_IP = os.environ['BACKEND_MINIO_IP']
else:
    MINIO_IP = ""

URL_AUTH = JOBSUBMISSION_API_URL + "auth/obtain_token/"
URL_JOBS = JOBSUBMISSION_API_URL + "nismod-model/jobs"


class DAFNIRunScheduler(object):
    """The scheduler can run instances of smif as a subprocess
    and can provide information whether the modelrun is running,
    is done or has failed.
    """
    def __init__(self, username, password):
        self._status = defaultdict(lambda: 'unstarted')
        self._process = {}
        self._output = defaultdict(str)
        self._err = {}
        self.jobId = 0
        self.lock = False
        self.username = username
        self.password = password
        response = requests.post(
            URL_AUTH,
            json={
                "username": self.username,
                "password": self.password
            },
            allow_redirects=False
        )
        response.raise_for_status()
        token = response.json()['token']
        self.auth_header = json.loads('{ "Authorization": "JWT ' + token + '"}')
        response = requests.get(URL_JOBS, headers=self.auth_header)
        response.raise_for_status()

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
        DAFNI's queuing mechanism starts model runs in separate
        container. This means that it is possible to run multiple
        modelruns concurrently. This will not cause conflicts.
        """
        if self._status[model_run_name] != 'running':
            self._output[model_run_name] = ''
            self._status[model_run_name] = 'queing'

            yaml_files = self.get_yamls(model_run_name, args)
            model_run_id = model_run_name.replace("_", "-")

            minio_credentials = self.get_dict_from_json(MINIO_CREDENTIALS_FILE)
            minio_client = Minio(
                MINIO_IP,
                access_key=minio_credentials['accessKey'],
                secret_key=minio_credentials['secretKey'],
                secure=False
            )
            bucket_list = minio_client.list_buckets()

            for bucket in bucket_list:
                if bucket.name == model_run_id:
                    for obj in minio_client.list_objects(model_run_id, recursive=True):
                        minio_client.remove_object(model_run_id, obj.object_name)
                    minio_client.remove_bucket(model_run_id)

            minio_client.make_bucket(model_run_id)
            for yml in yaml_files:
                local_path = args['directory'] + yml
                with open(local_path, 'rb') as yml_data:
                    yml_stat = os.stat(local_path)
                    minio_client.put_object(
                        model_run_id, yml[1:], yml_data, yml_stat.st_size)

            response = requests.get(URL_JOBS, headers=self.auth_header)
            response.raise_for_status()

            for job in response.json():
                if job['job']['job_name'] == model_run_id:
                    response = requests.delete(
                        URL_JOBS + "/" + str(job['job']['id']), headers=self.auth_header)
                    response.raise_for_status()

            response = requests.post(
                URL_JOBS,
                json={
                    "job_name": model_run_id,
                    "model_name": model_run_name,
                    "minio_config_id": model_run_id
                },
                headers=self.auth_header
            )
            response.raise_for_status()

    def get_scheduler_type(self):
        return "dafni"

    def get_yamls(self, model_run_name, args):
        yaml_files = []
        yaml_files.append("/config/model_runs/" + model_run_name + ".yml")
        f = open(args['directory'] + yaml_files[0], "r")
        doc = YAML(typ='safe').load(f.read())

        yaml_files.append("/config/sos_models/" + doc['sos_model'] + ".yml")

        sos_f = open(args['directory'] + yaml_files[1])
        sos_doc = YAML(typ='safe').load(sos_f.read())

        for sector_model in sos_doc['sector_models']:
            yaml_files.append("/config/sector_models/" + sector_model + ".yml")

        for scenario in sos_doc['scenarios']:
            yaml_files.append("/config/scenarios/" + scenario + ".yml")

        return yaml_files

    def get_dict_from_json(self, file_path):
        '''
        Given a JSON file, will return the dictionary of values within that file.
        Is developed to wait for a file to exist (for the case of reading vault secrets)
        to avoid the inherent race condition.
        '''
        count = 0
        while True:
            try:
                with open(file_path) as file:
                    data = json.load(file)
                    return data
            except FileNotFoundError:
                count += 1
                if count > 3:
                    raise FileNotFoundError
                else:
                    print('{} does not exist yet. Waiting 5 seconds and trying again.\
                        Have tried {} times'.format(file_path, count))
                    time.sleep(5)

    def kill(self, model_run_name):
        if self._status[model_run_name] == 'running':
            self._status[model_run_name] = 'stopped'

        minio_credentials = self.get_dict_from_json(MINIO_CREDENTIALS_FILE)
        minio_client = Minio(
            MINIO_IP,
            access_key=minio_credentials['accessKey'],
            secret_key=minio_credentials['secretKey'],
            secure=False
        )

        model_run_id = model_run_name.replace("_", "-")
        yaml_files_minio = minio_client.list_objects(model_run_id, recursive=True)
        for d in yaml_files_minio:
            minio_client.remove_object(model_run_id, d.object_name)

        minio_client.remove_bucket(model_run_id)

        response = requests.get(URL_JOBS, headers=self.auth_header)
        response.raise_for_status()

        for job in response.json():
            if job['job']['job_name'] == model_run_id:
                requests.delete(
                    URL_JOBS + "/" + str(job['job']['id']), headers=self.auth_header)

    def get_status(self, model_run_name):
        response = requests.get(URL_JOBS, headers=self.auth_header)
        response.raise_for_status()
        model_run_id = model_run_name.replace("_", "-")
        if len(response.json()) > 0:
            for j in response.json():
                if j['job']['job_name'] == model_run_id:
                    job = j['job']
                    status = job['status']
                    jobStatus = ["unstarted", "unstarted", "running", "done", "failed"]
                    self._status[model_run_name] = jobStatus[status]
                    break
        else:
            self._status[model_run_name] = "unstarted"
        return {
            'status': self._status[model_run_name],
            'output': self._output[model_run_name]
        }
