"""Database store implementations
"""
import json

# import psycopg2 to handel database transactions
import psycopg2
import psycopg2.extras
from smif.data_layer.abstract_config_store import ConfigStore
from smif.data_layer.abstract_data_store import DataStore
from smif.data_layer.abstract_metadata_store import MetadataStore


def initiate_db_connection(host, user_name, database_name, port, password):
    """Establish a database connection

    Returns
    -------
    database_connection
        An established connection to the database

    """

    # attempt to create the database connection
    database_connection = psycopg2.connect("host=%s dbname=%s user=%s password=%s port=%s" %
                                           (host, database_name, user_name, password, port))

    return database_connection


class DbConfigStore(ConfigStore):
    """Database backend for config store
    """

    def __init__(self, host, user, dbname, port, password):
        """Initiate. Setup database connection. Set up common values.
        """
        # establish database connection
        self.database_connection = initiate_db_connection(host, user, dbname, port, password)

        # list of expected port types - these should be specified somewhere, or not presumed at all
        self.port_types = ['inputs', 'outputs', 'parameters']

    # region Model runs
    def read_model_runs(self):
        """Read all model runs

        Returns
        -------
        list
            A list of dicts containing model runs
        """
        # establish a cursor for the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # query database to get model runs
        cursor.execute('SELECT * FROM model_runs;')

        # get result of query
        model_runs = cursor.fetchall()

        return model_runs

    def read_model_run(self, model_run_name):
        """Read a single model run

        Returns
        -------
        dict
            A dictionary containing a model run definition
        """
        # establish a cursor for the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # sql to get model run information
        cursor.execute('SELECT * FROM model_runs WHERE name=%s;', [model_run_name])

        # get result of query
        model_run = cursor.fetchall()

        # check only one model run has been returned
        if len(model_run) > 1:
            # more than one model run returned, a database data error has occurred
            return
        elif len(model_run) == 0:
            # no model run exists with given name
            return

        return model_run[0]

    def write_model_run(self, model_run):
        """Read all systems of systems models

        Argument
        --------
        model_run: dict
            A dictionary containing a model run definition
        """
        # establish a cursor for the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check model run name is unique
        cursor.execute('SELECT id FROM model_runs WHERE name = %s;', [model_run['name']])

        # get result of query
        model_runs = cursor.fetchall()

        # if one or more model runs are returned, return an error
        if len(model_runs) > 0:
            # model run with same name already exists
            return

        # check given sos model already exists
        cursor.execute('SELECT id FROM sos_models WHERE name=%s;', [model_run['sos_model']])

        # get result of query
        sos_models = cursor.fetchall()

        # check that only one sos model exists with the name given, otherwise return an error
        if len(sos_models) > 1:
            # more than one sos model with the given name - a data error
            return
        elif len(sos_models) == 0:
            # no sos model exists with the name given
            return

        # write model run to database
        cursor.execute('INSERT INTO model_runs (name, sos_model, sos_model_id) VALUES (%s,%s,%s);', [model_run['name'], model_run['sos_model'], sos_models[0]['id']])

        # write data to database
        self.database_connection.commit()

        return

    def update_model_run(self, model_run_name, model_run):
        """Update an existing model run definition

        Argument
        --------
        model_run_name: string
            The name of the model run to update
        model_run: dict
            A dictionary containing a model run definition with those arguments to be updated
        """
        # establish a cursor for the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check given model run is in the database
        cursor.execute('SELECT id FROM model_runs WHERE name=%s;', [model_run_name])

        # get result of query
        model_runs = cursor.fetchall()

        # if anything but one model run returned, return an error
        if len(model_runs) == 1:

            # if the sos model key has been passed in the definition
            if 'sos_model' in model_run.keys():

                # check the passed model exists
                cursor.execute('SELECT id FROM sos_models WHERE name=%s;', [model_run['sos_model']])

                # get result of query
                sos_model_id = cursor.fetchall()

                # if only one model run id returned, make change
                if len(sos_model_id) == 1:
                    # update table with new sos model
                    cursor.execute('UPDATE model_runs SET sos_model=%s, sos_model_id=%s WHERE name=%s;', [model_run['sos_model'], sos_model_id[0]['id'],model_run_name])

                    # commit update to database
                    self.database_connection.commit()

                else:
                    # return as error as given sos model does not exist
                    return

        else:
            # return an error to the user
            return

        return

    def delete_model_run(self, model_run_name):
        """Delete a model run

        Argument
        --------
        model_run_name: string
            The name of the model run
        """
        # establish a cursor for the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check given model run is in database
        cursor.execute('SELECT id FROM model_runs WHERE name=%s;', [model_run_name])

        # get result of query
        model_runs = cursor.fetchall()

        # if only one model run returned, delete it
        if len(model_runs) == 1:
            # delete model run
            cursor.execute('DELETE FROM model_runs WHERE name=%s;', [model_run_name])

            # commit the delete action to the database
            self.database_connection.commit()

        else:
            # return an error as no model run, or multiple, returned
            return

        return

    # endregion

    # region System-of-systems models
    def read_sos_models(self):
        """Read all systems of systems models

        Returns
        -------
        list
            A list of dicts containing sos model definitions
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # get sos model details
        cursor.execute('SELECT name FROM sos_models;')

        # get returned data
        sos_models = cursor.fetchall()

        # is some models are returned
        if sos_models is not None:

            # create list to store models
            sos_models_list = []

            # loop through existing models
            for model_name in sos_models:

                # get the model details
                sos_models_list.append(self.read_sos_model(model_name[0]))

        return sos_models_list

    def read_sos_model(self, sos_model_name):
        """Read a single system of systems model

        Argument
        --------
        sos_model_name: string
            The name of the systems of systems model to read

        Returns
        -------
        dict
            The systems of systems model definition
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # get sos model details
        cursor.execute('SELECT * FROM sos_models WHERE name=%s;', [sos_model_name])

        # get returned data
        sos_model = cursor.fetchone()

        if sos_model is not None:
            # get simulation models
            cursor.execute('SELECT * FROM sos_model_simulation_models WHERE sos_model_name=%s;', [sos_model_name])

            # add all returned models to the sector models key in the sos_model definition
            sos_model['sector_models'] = []
            for model in cursor.fetchall():
                sos_model['sector_models'].append(model)

            # get scenario sets
            cursor.execute('SELECT * FROM sos_model_scenarios WHERE sos_model_name=%s;', [sos_model_name])

            # add all returned models to the sector models key in the sos_model definition
            sos_model['scenario_sets'] = []
            for scenario in cursor.fetchall():
                sos_model['scenario_sets'].append(scenario)

            # get sos model dependencies
            cursor.execute('SELECT * FROM sos_model_dependencies WHERE sos_model_name=%s;', [sos_model_name])

            # add all returned dependencies to the dependencies key in the sos_model definition
            sos_model['dependencies'] = []
            for dependency in cursor.fetchall():
                sos_model['dependencies'].append(dependency)

        return sos_model

    def write_sos_model(self, sos_model):
        """Write a systems of systems model

        Argument
        --------
        sos_model: dict
            The definition for a systems os systems model

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check name does not already exist
        cursor.execute('SELECT name FROM sos_models WHERE name=%s;', [sos_model['name']])

        # get returned data from query
        sos_models = cursor.fetchall()

        # if one of more names are returned
        if len(sos_models) > 0:
            # return an error - sos model already exists, use a different name
            return

        # write sos model, return id
        cursor.execute('INSERT INTO sos_models (name, description) VALUES (%s, %s) RETURNING id;', [sos_model['name'], sos_model['description']])

        # write to database
        self.database_connection.commit()

        # get returned model id
        sos_model_id = cursor.fetchone()

        # write dependencies
        # loop through passed dependencies
        for dependency in sos_model['dependencies']:
            # check dependent models already exist in database - source model
            cursor.execute('SELECT name FROM simulation_models WHERE name=%s;', [dependency['source_model']])

            # get returned data from query
            sos_models = cursor.fetchall()

            # if no names are returned
            if len(sos_models) == 0:
                # return an error - simulation model does not exist
                return

            # check dependent models already exist in database - sink model
            cursor.execute('SELECT name FROM simulation_models WHERE name=%s;', [dependency['sink_model']])

            # get returned data from query
            sos_models = cursor.fetchall()

            # if no names are returned
            if len(sos_models) == 0:
                # return an error - simulation model does not exist
                return

            # sql to write dependency to db
            cursor.execute('INSERT INTO sos_model_dependencies (sos_model_name, source_model, source_output, sink_model, sink_input, lag) VALUES (%s,%s,%s,%s,%s,%s);', [sos_model['name'], dependency['source_model'], dependency['source_model_output'], dependency['sink_model'], dependency['sink_model_input'], dependency['lag']])

            # write to database
            self.database_connection.commit()

        # write sos_model_sim_models
        sim_model_counter = 1
        for sector_model in sos_model['sector_models']:

            # check simulation model already in database
            cursor.execute('SELECT id FROM simulation_models WHERE name=%s;', [sector_model])

            # get returned data from query
            simulation_models = cursor.fetchall()

            # if no names are returned
            if len(simulation_models) == 0:
                # return an error - simulation model does not exist
                return

            # write link between sos model and simulation models
            cursor.execute('INSERT INTO sos_model_simulation_models (sos_model_name, simulation_model_name, simulation_model_id, sos_sim_model_id) VALUES (%s,%s,%s,%s);', [sos_model['name'], sector_model, simulation_models[0]['id'], sim_model_counter])

            # iterate counter
            sim_model_counter += 1

            # write to database
            self.database_connection.commit()

        # write sos_model_scenarios
        for scenario in sos_model['scenario_sets']:
            # check scenario already exists
            cursor.execute('SELECT name FROM scenarios WHERE name=%s;', [scenario])

            # get returned data from query
            scenario_names = cursor.fetchall()

            # if no names are returned
            if len(scenario_names) == 0:
                # return an error - scenario does not exist
                return

            # add data into database
            cursor.execute('INSERT INTO sos_model_scenarios (sos_model_name, scenario_name) VALUES (%s,%s);', [sos_model['name'], scenario])

        # write to database
        self.database_connection.commit()

        return

    def update_sos_model(self, sos_model_name, sos_model):
        """Update a systems of systems model

        Argument
        --------
        sos_model_name: string
            The name of the systems of systems model to update
        sos_model: dict
            The definition of a systems of systems model with only the data to be updated in

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check sos model exists
        cursor.execute('SELECT id FROM sos_models WHERE name=%s;', [sos_model_name])

        # get result of query
        sos_models = cursor.fetchall()

        # if no model with the name exists, return
        if len(sos_models) == 0:
            # no model with the name given
            return

        # update dependencies if passed
        if 'dependencies' in sos_model.keys():

            # loop through dependencies
            for dependency in sos_model['dependencies']:
                # update details

                if 'source_model' in dependency.keys():
                    # check dependent models already exist in database - source model
                    cursor.execute('SELECT name FROM simulation_models WHERE name=%s;', [dependency['source_model']])

                    # get returned data from query
                    sos_models = cursor.fetchall()

                    # if no names are returned
                    if len(sos_models) == 0:
                        # return an error - simulation model does not exist
                        return

                    # update
                    cursor.execute('UPDATE dependencies SET source_model=%s WHERE name=%s;' [dependency['source_model'], sos_model_name])

                    # write to database
                    self.database_connection.commit()

                if 'sink_model' in dependency.keys():
                    # check dependent models already exist in database - sink model
                    cursor.execute('SELECT name FROM simulation_models WHERE name=%s;', [dependency['sink_model']])

                    # get returned data from query
                    sos_models = cursor.fetchall()

                    # if no names are returned
                    if len(sos_models) == 0:
                        # return an error - simulation model does not exist
                        return

                    # update
                    cursor.execute('UPDATE dependencies SET sink_model=%s WHERE name=%s;', [dependency['source_model'], sos_model_name])

                    # write to database
                    self.database_connection.commit()

                if 'source_model_output' in dependency.keys():
                    # update
                    cursor.execute('UPDATE dependencies SET source_model_output=%s WHERE name=%s;', [dependency['source_model_output'], sos_model['name']])

                    # write to database
                    self.database_connection.commit()

                if 'sink_model_input' in dependency.keys():
                    # update
                    cursor.execute('UPDATE dependencies SET sink_model_input=%s WHERE name=%s;', [dependency['sink_model_input'], sos_model_name])

                    # write to database
                    self.database_connection.commit()

                if 'lag' in dependency.keys():
                    # update
                    cursor.execute('UPDATE dependencies SET lag=%s WHERE name=%s;', [dependency['lag'], sos_model_name])

                    # write to database
                    self.database_connection.commit()

        # need to update methods as does not all for multiple as different rows
        # update sos_model_simulation_models if passed
        if 'sector_models' in sos_model.keys():

            # loop through the sector models
            for sector_model in sos_model['sector_models']:

                # check simulation model already in database
                cursor.execute('SELECT name FROM simulation_models WHERE name=%s;', [sector_model])

                # get returned data from query
                sos_models = cursor.fetchall()

                # if no names are returned
                if len(sos_models) == 0:
                    # return an error - simulation model does not exist
                    return

            # if all simulation models in database, attempt to update the sos model sim model relation

            # get what is already in the database
            cursor.execute('SELECT *  FROM sos_model_simulation_models WHERE sos_model_name=%s;', [sos_model_name])

            # get result from query
            sos_model_sim_models = cursor.fetchall()

            for sector_model in sos_model['sector_models']:
                # already checked if sim model in database
                # start by seeing if in relation already - loop through existing data
                for sim_model in sos_model_sim_models:
                    if sim_model['simulation_model_name'] == sector_model:
                        # return an error as already in relation so don't need to add again
                        return

                # if here, not in relation so add sim model to sos model sim model relation
                # get max id count in relation
                cursor.execute('SELECT max(sos_sim_model_id) FROM sos_model_simulation_models WHERE sos_model_name=%s;', [sos_model_name])

                # get result of query
                sos_sim_model_id = cursor.fetchone()

                # write sim model to sos model sim models relation
                cursor.execute('INSERT INTO sos_model_simulation_models (sos_model_name, simulation_model_name, sos_sim_model_id) VALUES (%s,%s,%s);', [sos_model_name, sector_model, sos_sim_model_id[0]])

                # write update to database
                self.database_connection.commit()

        # need to update methods as does not allow multiple as different rows
        # update sos_model_scenarios if passed
        if 'scenario_sets' in sos_model.keys():

            # loop through passed scenarios
            for scenario in sos_model['scenario_sets']:
                # check scenario already exists
                cursor.execute('SELECT name FROM scenarios WHERE name=%s;', [scenario])

                # get returned data from query
                scenario_names = cursor.fetchall()

                # if no names are returned
                if len(scenario_names) == 0:
                    # return an error - scenario does not exist
                    return

                # add data into database
                cursor.execute('UPDATE sos_model_scenarios SET scenario_name=%s WHERE sos_model_name=%s;', [scenario, sos_model_name])

            # write to database
            self.database_connection.commit()

        # update sos model description if passed
        if 'description' in sos_model.keys():

            # sql to update description
            cursor.execute('UPDATE sos_models SET description=%s WHERE name=%s;', [sos_model['description'], sos_model_name])

            # write to database
            self.database_connection.commit()

        return

    def delete_sos_model(self, sos_model_name):
        """Delete a systems of systems model

        Argument
        --------
        sos_model_name: string
            The name of the systems of systems model to delete

        """

        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check sos model exists
        cursor.execute('SELECT id FROM sos_models WHERE name=%s;', [sos_model_name])

        # get result of query
        sos_models = cursor.fetchall()

        # if none returned, exit with error
        if len(sos_models) == 0:
            # no model with the given name found
            return

        # delete from sos_model_simulation_models
        cursor.execute('DELETE FROM sos_model_simulation_models WHERE sos_model_name = %s;', [sos_model_name])

        # run query to delete from db
        self.database_connection.commit()

        # delete from sos_model_scenarios
        cursor.execute('DELETE FROM sos_model_scenarios WHERE sos_model_name = %s;', [sos_model_name])

        # run query to delete from db
        self.database_connection.commit()

        # delete from sos_model_dependencies
        cursor.execute('DELETE FROM sos_model_dependencies WHERE sos_model_name = %s;', [sos_model_name])

        # run query to delete from db
        self.database_connection.commit()

        # delete from model_run
        cursor.execute('DELETE FROM model_runs WHERE sos_model = %s;', [sos_model_name])

        # run query to delete from db
        self.database_connection.commit()

        # delete from sos_models
        cursor.execute('DELETE FROM sos_models WHERE name = %s;', [sos_model_name])

        # run query to delete from db
        self.database_connection.commit()

        return
    # endregion

    # region Models
    def read_models(self):
        """Read all simulation models

        Returns
        -------
        list
            A list of dictionaries for the models returned
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # run sql call
        cursor.execute('SELECT * FROM simulation_models')

        # get returned data
        simulation_models = cursor.fetchall()

        # loop through the returned simulation models to get their details
        # create dictionary to store simulation models
        simulation_model_list = {}
        # loop through known models
        for simulation_model in simulation_models:

            # get details of the models from the read call and add to list
            simulation_model_list[simulation_model['name']] = self.read_model(simulation_model['name'])

        # return data to user
        return simulation_model_list

    def read_model(self, model_name):
        """Read a simulation model

        Argument
        --------
        model_name: string
            The name of the model to read

        Returns
        -------
        dict
            A dictionary of the model definition returned
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # run sql call
        cursor.execute('SELECT * FROM simulation_models WHERE name=%s', [model_name])

        # get returned data
        simulation_model = cursor.fetchall()

        # check if a simulation model has been found and only proceed if so
        if len(simulation_model) == 1:
            # get port types for model - inputs, outputs, parameters
            # run sql call
            for port_type in self.port_types:

                # get specification details by joining sim model port and specification relation using the specification id (unique to each specification)
                cursor.execute('SELECT s.* FROM simulation_model_port smp JOIN specifications s ON smp.specification_id=s.id WHERE smp.model_name=%s AND port_type=%s;', [model_name, port_type])

                # get returned data
                port_data = cursor.fetchall()

                # add port data to model dictionary
                simulation_model[0][port_type] = port_data

        # return data to user
        return simulation_model

    def write_model(self, model):
        """Write a simulation model to the database

        Argument
        --------
        model: dictionary
            A model definition

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # write model
        # write the model to the database
        cursor.execute('INSERT INTO simulation_models (name, description, interventions, wrapper_location) VALUES (%s,%s,%s,%s) RETURNING id;', [model['name'], model['description'], json.dumps(model['interventions']), model['wrapper_location']])

        # commit changes to database
        self.database_connection.commit()

        # get returned data
        simulation_model_id = cursor.fetchone()

        # write the specification to the database (metadata on the inputs/outputs/parameters)

        # loop through port types
        for port_type in self.port_types:
            # if port type is a key in given model definition, add all specifications to database
            if port_type in model.keys():
                # loop through the specifications for the port type
                for spec in model[port_type]:
                    # should probably check it is already not in the database first

                    # add specification to database
                    cursor.execute('INSERT INTO specifications (name, description, dimensions, unit, suggested_range, absolute_range) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id;', [spec['name'], spec['description'], spec['dimensions'], spec['unit'], spec['suggested_range'], spec['absolute_range']])

                    # commit changes to database
                    self.database_connection.commit()

                    # get returned data
                    specification_id = cursor.fetchone()

                    # write to port
                    # write model and specification to port table, including the specification id
                    cursor.execute('INSERT INTO simulation_model_port (model_name, model_id, specification_name, specification_id, port_type) VALUES (%s, %s, %s, %s, %s);', [model['name'], simulation_model_id[0], spec['name'], specification_id[0], port_type])

                    # commit changes to database
                    self.database_connection.commit()

        # return data to user
        return

    def update_model(self, model_name, model):
        """Update a simulation model

        Argument
        --------
        model_name: string
            The name of the model to update
        model:
             A model definition with only the fields to be updated

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # update the model description is passed
        if 'description' in model.keys():
            cursor.execute('UPDATE simulation_models SET description=%s WHERE name=%s;', [model['description'], model_name])

        # update the intervention list is passed
        if 'interventions' in model.keys():
            cursor.execute('UPDATE simulation_models SET interventions=%s WHERE name=%s;', [json.dumps(model['interventions']), model_name])

        # update the wrapper location if passed
        if 'wrapper_location' in model.keys():
            cursor.execute('UPDATE simulation_models SET wrapper_location=%s WHERE name=%s;', [model['wrapper_location'], model_name])

        # commit changes to database
        self.database_connection.commit()

        # update any of the port types if passed
        # need to figure out how to update an inputs/output/parameter and the specification
        for port_type in self.port_types:

            # if the port type has been passed to be updated
            if port_type in model.keys():

                # loop through each specification for the port type
                for spec in model[port_type]:

                    # get the id of the specification to update - based on name, model and port
                    cursor.execute('SELECT specification_id FROM simulation_model_port WHERE port_type=%s and specification_name=%s and model_name=%s;', [port_type, spec['name'], model_name])

                    # get result of query
                    spec_id = cursor.fetchall()

                    # check for possible errors related to the existence of the specification
                    if len(spec_id) > 1:
                        # return an error, more than one specification id returned
                        return
                    elif len(spec_id) == 0:
                        # return an error - no matching specification found to be updated
                        return

                    # check for each key in the specification and update if present
                    if 'name' in spec.keys():
                        continue
                    if 'description' in spec.keys():
                        # run update sql
                        cursor.execute('UPDATE specifications SET description = %s WHERE id=%s', [spec['description'], spec_id])
                    if 'dimensions' in spec.keys():
                        # run update sql
                        cursor.execute('UPDATE specifications SET dimensions = %s WHERE id=%s',                                       [spec['dimensions'], spec_id])
                    if 'unit' in spec.keys():
                        # run update sql
                        cursor.execute('UPDATE specifications SET unit = %s WHERE id=%s', [spec['unit'], spec_id])
                    if 'suggested_range' in spec.keys():
                        # run update sql
                        cursor.execute('UPDATE specifications SET suggested_range = %s WHERE id=%s',[spec['suggested_range'], spec_id])
                    if 'absolute_range' in spec.keys():
                        # run update sql
                        cursor.execute('UPDATE specifications SET absolute_range = %s WHERE id=%s',[spec['absolute_range'], spec_id])

                    # commit changes to database
                    self.database_connection.commit()

        return

    def delete_model(self, model_name):
        """Delete a simulation model

        Argument
        --------
        model_name: string
            The name of the model to be deleted

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # delete specifications if unique to model
        # get a list of the specifications linked to the model name
        cursor.execute('SELECT * FROM simulation_model_port WHERE model_name=%s;', [model_name])

        # loop through returned model specifications
        for specification in cursor.fetchall():

            # check if the specification is used by any other models
            cursor.execute('SELECT COUNT(*) FROM simulation_model_port WHERE specification_id=%s;', [specification['id']])

            # get the count from the query
            specification_count = cursor.fetchone()

            # if the count is only 1, safe to delete specification, otherwise leave it
            if specification_count == 1:
                cursor.execute('DELETE FROM specifications WHERE id=%s;', [specification['id']])

                # commit changes to database
                self.database_connection.commit()

        # run sql call to delete from simulation model port
        cursor.execute('DELETE FROM simulation_model_port WHERE model_name=%s;', [model_name])

        # commit changes to database
        self.database_connection.commit()

        # run sql call to delete from simulation models
        cursor.execute('DELETE FROM simulation_models WHERE name=%s;', [model_name])

        # commit changes to database
        self.database_connection.commit()

        # get the number of rows deleted and return
        affected_rows = cursor.rowcount

        return
    # endregion

    # region Scenarios
    def read_scenarios(self):
        """Read list of scenarios

         Returns
        -------
        list
            A list of scenarios as dictionary's
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # run sql call
        cursor.execute('SELECT * FROM scenarios')

        # get returned data
        scenarios = cursor.fetchall()

        # return data to user
        return scenarios

    def read_scenario(self, scenario_name):
        """Read a scenario

        Argument
        --------
        scenario_name: string
            The name of the scenario to read

        Returns
        -------
        dict
            A scenario definition
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # run sql call
        cursor.execute('SELECT * FROM scenarios WHERE name=%s', [scenario_name])

        # get returned data
        scenario = cursor.fetchone()

        # return data to user
        return scenario

    def write_scenario(self, scenario):
        """Write a scenario

        Argument
        --------
        scenario: dict
            A scenario definition

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # run sql call
        sql = 'INSERT INTO scenarios (name, description) VALUES (%s,%s) RETURNING id;'
        cursor.execute(sql, [scenario['name'], scenario['description']])

        # commit changes to database
        self.database_connection.commit()

        # get id of new scenario - checks it has been written in
        scenario_id = cursor.fetchone()

        return

    def update_scenario(self, scenario_name, scenario):
        """Update a scenario

        Argument
        --------
        scenario_name: string
            The name of the scenario to update
        scenario: dict
            The project configuration

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # run sql call
        cursor.execute('UPDATE scenarios SET description = %s WHERE name=%s', [scenario['description'], scenario_name])

        # commit changes to database
        self.database_connection.commit()

        return

    def delete_scenario(self, scenario_name):
        """Delete a scenario

        Argument
        --------
        scenario_name: string
            The name of the scenario to delete
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # check passed scenario exists
        cursor.execute('SELECT id FROM scenarios WHERE name=%s;', [scenario_name])

        # get returned data - scenario id
        scenario_id = cursor.fetchall()

        if len(scenario_id) > 0:

            # need to check for and delete any scenario variants that are associated with this scenario
            cursor.execute('DELETE FROM scenario_variants WHERE scenario_name=%s;', [scenario_name])

            # commit changes to database
            self.database_connection.commit()

            # run sql call
            cursor.execute('DELETE FROM scenarios WHERE name=%s', [scenario_name])

            # commit changes to database
            self.database_connection.commit()

        return
    # endregion

    # region Scenario Variants
    def read_scenario_variants(self, scenario_name):
        """Read scenario variants

        Argument
        --------
        scenario_name: string
            The name of the scenario name to return variants for

        Returns
        -------
        dict
            A list scenario variants as dictionaries
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # run sql call
        cursor.execute('SELECT sv.*, v.description, v.data FROM scenario_variants sv JOIN variants v ON sv.variant_name = v.name WHERE sv.scenario_name=%s', [scenario_name])

        # get returned data
        scenario_variants = cursor.fetchall()

        # return data to user
        return scenario_variants

    def read_scenario_variant(self, scenario_name, variant_name):
        """Read scenario variants

        Argument
        --------
        scenario_name: string
            The name of the scenario
        variant_name: string
            The name of scenario variant to return

        Returns
        -------
        dict
            A scenario variant definition
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # run sql call
        cursor.execute('SELECT sv.*, v.description, v.data FROM scenario_variants sv JOIN variants v ON sv.variant_name = v.name WHERE sv.scenario_name=%s AND sv.variant_name=%s', [scenario_name, variant_name])

        # get returned data
        scenario_variant = cursor.fetchall()

        # return data to user
        return scenario_variant

    def write_scenario_variant(self, scenario_name, variant):
        """Write scenario variant

        Argument
        --------
        scenario_name: string
            The name of the scenario the variant is associated with
        variant: dict
            The variant definition

        Returns
        -------
        integer
            The id of the added scenario variant
        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # here is a dependency on the given scenario name already existing
        # - this can be enforced in the database by a foreign key, but should probably check here
        # - would also enable access to the scenario id
        cursor.execute('SELECT id FROM scenarios WHERE name=%s;', [scenario_name])

        # get returned data
        scenario_id = cursor.fetchall()

        # if a scenario_id has been found
        if len(scenario_id) == 1:

            # run sql to add data to variants to database
            cursor.execute('INSERT INTO variants (name, description, data) VALUES (%s,%s,%s) RETURNING id;', [variant['variant_name'], variant['description'], json.dumps(variant['data'])])

            # commit changes to database
            self.database_connection.commit()

            # get returned data
            variant_id = cursor.fetchone()

            # run sql call
            sql = 'INSERT INTO scenario_variants (scenario_name, variant_name, scenario_id) VALUES (%s,%s,%s) RETURNING id;'
            cursor.execute(sql, [variant['scenario_name'], variant['variant_name'], scenario_id[0]['id']])

            # commit changes to database
            self.database_connection.commit()

            # get returned data
            scenario_variant_id = cursor.fetchone()

            return scenario_variant_id
        else:
            return

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        """Update scenario variant

        Argument
        --------
        scenario_name: string
            The name of the scenario the variant is associated with
        variant_name: string
            The name of variant to be updated
        variant: dict
            The variant definition containing the data to be updated

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if 'description' in variant.keys() and 'data' in variant.keys():

            # run sql call
            cursor.execute('UPDATE variants SET description=%s AND data=%s WHERE name = %s RETURNING id;', [variant['description'], json.dumps(variant['data']), variant_name])

        elif 'description' in variant.keys() and 'data' not in variant.keys():

            # run sql call
            cursor.execute('UPDATE variants SET description=%s WHERE name = %s RETURNING id;', [variant['description'], variant_name])

        elif 'description' not in variant.keys() and 'data' in variant.keys():

            # run sql call
            cursor.execute('UPDATE variants SET data=%s WHERE name = %s RETURNING id;', [json.dumps(variant['data']), variant_name])

        # commit changes to database
        self.database_connection.commit()

        # get the number of rows deleted and return
        affected_rows = cursor.rowcount

        return affected_rows

    def delete_scenario_variant(self, scenario_name, variant_name):
        """"Delete scenario variant

        Argument
        --------
        scenario_name: string
            The name of the scenario the variant to be deleted is associated with
        variant_name: string
            The name of the variant to be deleted

        Returns
        -------
        integer
            The number of rows deleted

        """
        # establish a cursor to read the database
        cursor = self.database_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # run sql call
        sql = 'DELETE FROM scenario_variants WHERE scenario_name=%s AND variant_name=%s'
        cursor.execute(sql, [scenario_name, variant_name])

        # commit changes to database
        self.database_connection.commit()

        # run sql call
        cursor.execute('DELETE FROM variants WHERE name=%s', [scenario_name, variant_name])

        # commit changes to database
        self.database_connection.commit()

        # get the number of rows deleted and return
        affected_rows = cursor.rowcount

        return affected_rows

    # endregion

    # region Narratives
    def read_narrative(self, sos_model_name, narrative_name):
        raise NotImplementedError()
    # endregion

    # region Strategies
    def read_strategies(self, model_run_name):
        raise NotImplementedError()

    def write_strategies(self, model_run_name, strategies):
        raise NotImplementedError()
    # endregion


class DbMetadataStore(MetadataStore):
    """Database backend for metadata store
    """
    # region Units
    def read_unit_definitions(self):
        raise NotImplementedError()
    # endregion

    # region Dimensions
    def read_dimensions(self):
        raise NotImplementedError()

    def read_dimension(self, dimension_name):
        raise NotImplementedError()

    def write_dimension(self, dimension):
        raise NotImplementedError()

    def update_dimension(self, dimension_name, dimension):
        raise NotImplementedError()

    def delete_dimension(self, dimension_name):
        raise NotImplementedError()
    # endregion


class DbDataStore(DataStore):
    """Database backend for data store
    """
    # region Scenario Variant Data
    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        raise NotImplementedError()

    def write_scenario_variant_data(self, scenario_name, variant_name,
                                    data_array, timestep=None):
        raise NotImplementedError()
    # endregion

    # region Narrative Data
    def read_narrative_variant_data(self, narrative_name, variant_name, variable,
                                    timestep=None):
        raise NotImplementedError()

    def write_narrative_variant_data(self, narrative_name, variant_name, data_array,
                                     timestep=None):
        raise NotImplementedError()

    def read_model_parameter_default(self, model_name, parameter_name):
        raise NotImplementedError()

    def write_model_parameter_default(self, model_name, parameter_name, data):
        raise NotImplementedError()
    # endregion

    # region Interventions
    def read_interventions(self, sector_model_name):
        raise NotImplementedError()

    def read_initial_conditions(self, sector_model_name):
        raise NotImplementedError()
    # endregion

    # region State
    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        raise NotImplementedError()

    def write_state(self, state, modelrun_name, timestep, decision_iteration=None):
        raise NotImplementedError()
    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_spec, destination_spec):
        raise NotImplementedError

    def write_coefficients(self, source_spec, destination_spec, data):
        raise NotImplementedError()
    # endregion

    # region Results
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     decision_iteration=None):
        raise NotImplementedError()

    def write_results(self, data_array, modelrun_name, model_name, timestep=None,
                      decision_iteration=None):
        raise NotImplementedError()

    def prepare_warm_start(self, modelrun_id):
        raise NotImplementedError()
    # endregion
