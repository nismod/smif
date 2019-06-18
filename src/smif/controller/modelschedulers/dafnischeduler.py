"""Schedulers are used to run models.

The defaults provided allow model runs to be scheduled as subprocesses,
or individual models to be called in series.

Future implementations may interface with common schedulers to queue
up models to run in parallel and/or distributed.

The Dafni Scheduler currently only works while connected to
the RAL vpn.
"""
import subprocess
from ruamel.yaml import YAML  # type: ignore
import time
import json
import os
import requests
from collections import defaultdict
from datetime import datetime
from smif.data_layer import Store
from minio import Minio
from minio.error import ResponseError

if "BACKEND_NISMOD_MINIO_SECRETS_FILE" in os.environ:
    MINIO_CREDENTIALS_FILE=os.environ['BACKEND_NISMOD_MINIO_SECRETS_FILE']
else: 
    MINIO_CREDENTIALS_FILE=""
if "BACKEND_SECRET_KEY" in os.environ:
    SECRET_KEY=os.environ['BACKEND_SECRET_KEY']
else: 
    SECRET_KEY=""
if "BACKEND_ACCESS_KEY" in os.environ:
    ACCESS_KEY=os.environ['BACKEND_ACCESS_KEY']
else: 
    ACCESS_KEY=""

JOBSUBMISSION_API_URL="https://pilots-jobsubmissionapi-review-nismod-api-3udmh2.staging.dafni.rl.ac.uk/"
URL_AUTH=JOBSUBMISSION_API_URL + "auth/obtain_token/"
URL_JOBS=JOBSUBMISSION_API_URL + "nismod-model/jobs"

class DafniScheduler(object):
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
            json = json.loads('{ \
                "username": "' + self.username + '", \
                "password": "' + self.password + '" \
            }')
        )
        response.raise_for_status()
        print(response.text)
        print(response.json())
        self.token = response.json()['token']

        auth_header = json.loads('{ "Authorization": "JWT ' + self.token + '"}')
        response = requests.get(URL_JOBS, headers=auth_header)
        print(response.text)
        
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

            yaml_files = self.get_yamls(model_run_name, args)
            model_run_id = model_run_name.replace("_", "-")

            #minio_credentials = self.get_dict_from_json(MINIO_CREDENTIALS_FILE)
            minio_client = Minio(
                "130.246.6.245:9000",
                ACCESS_KEY,
                SECRET_KEY,
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
                try:
                    local_path = args['directory'] + yml
                    with open(local_path, 'rb') as yml_data:
                        yml_stat = os.stat(local_path)
                        minio_client.put_object(model_run_id, yml[1:], yml_data, yml_stat.st_size)
                except ResponseError as err:
                    print(err)

            auth_header = json.loads('{ "Authorization": "JWT ' + self.token + '"}')
            response = requests.get(URL_JOBS, headers=auth_header)

            for job in response.json():  
                if job['job']['job_name'] == model_run_id:
                    response = requests.delete(URL_JOBS + "/" + str(job['job']['id']), headers=auth_header)
                    print(response)

            job_string = json.loads('{ \
                "job_name": "' + model_run_id + '", \
                "model_name": "' + model_run_name + '", \
                "minio_config_id": "' + model_run_id + '" \
            }')
            response = requests.post(URL_JOBS, headers=auth_header, json=job_string)
            print(response.text)

        print("ADD")

    def get_scheduler_type(self):
        return "dafni"

    def get_yamls(self, model_run_name, args):
        yaml_files = []
        yaml_files.append("/config/model_runs/" + model_run_name + ".yml")
        f = open(args['directory'] + yaml_files[0], "r")
        doc = YAML().load(f.read(), YAML().SafeLoader)

        yaml_files.append("/config/sos_models/" + doc['sos_model'] + ".yml")
        
        sos_f = open(args['directory'] + yaml_files[1])
        sos_doc = YAML().load(sos_f.read(), YAML().SafeLoader)

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

        minio_client = Minio(
            "130.246.6.245:9000",
            ACCESS_KEY,
            SECRET_KEY,
            # access_key=minio_credentials['accessKey'],
            # secret_key=minio_credentials['secretKey'],
            secure=False
        )

        model_run_id = model_run_name.replace("_", "-")
        yaml_files_minio = minio_client.list_objects(model_run_id, recursive=True)
        for d in yaml_files_minio:
            print(d.object_name)
            minio_client.remove_object(model_run_id, d.object_name)

        minio_client.remove_bucket(model_run_id)

        auth_header = json.loads('{ "Authorization": "JWT ' + self.token + '"}')

        auth_header = json.loads('{ "Authorization": "JWT ' + self.token + '"}')
        response = requests.get(URL_JOBS, headers=auth_header)

        for job in response.json():  
            if job['job']['job_name'] == model_run_id:
                requests.delete(URL_JOBS + "/" + str(job['job']['id']), headers=auth_header)

        print("KILL")

    def get_status(self, model_run_name):
        print("GET_STATUS")
        auth_header = json.loads('{ "Authorization": "JWT ' + self.token + '"}')
        response = requests.get(URL_JOBS, headers=auth_header)
        print(response.text)
        model_run_id = model_run_name.replace("_", "-")
        for j in response.json():  
            if j['job']['job_name'] == model_run_id:
                job = j['job']
                status = job['status']
                jobStatus = ["unstarted", "unstarted", "running", "done", "failed"]
                self._status[model_run_name] = jobStatus[status]
                break
        return {
            'status': self._status[model_run_name],
            'output': self._output[model_run_name]
        }

