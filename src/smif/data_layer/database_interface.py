"""Database store implementations
"""
# import psycopg2 to handel database transactions
import psycopg2
import json
from psycopg2.extras import DictCursor

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
    database_connection = psycopg2.connect("host=%s dbname=%s user=%s password=%s port=%s" % (host, database_name, user_name, password, port))

    return database_connection


class DbConfigStore(ConfigStore):
    """Database backend for config store
    """

    def __init__(self, host, user, dbname, port, password):
        """Initiate. Setup database connection.
        """
        # establish database connection
        self.database_connection = initiate_db_connection(host, user, dbname, port, password)

    # region Model runs
    def read_model_runs(self):
        raise NotImplementedError()

    def read_model_run(self, model_run_name):
        raise NotImplementedError()

    def write_model_run(self, model_run):
        raise NotImplementedError()

    def update_model_run(self, model_run_name, model_run):
        raise NotImplementedError()

    def delete_model_run(self, model_run_name):
        raise NotImplementedError()
    # endregion

    # region System-of-systems models
    def read_sos_models(self):
        raise NotImplementedError()

    def read_sos_model(self, sos_model_name):
        raise NotImplementedError()

    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    def update_sos_model(self, sos_model_name, sos_model):
        raise NotImplementedError()

    def delete_sos_model(self, sos_model_name):
        raise NotImplementedError()
    # endregion

    # region Models
    def read_models(self):
        raise NotImplementedError()

    def read_model(self, model_name):
        raise NotImplementedError()

    def write_model(self, model):
        raise NotImplementedError()

    def update_model(self, model_name, model):
        raise NotImplementedError()

    def delete_model(self, model_name):
        raise NotImplementedError()
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
        cursor.execute('INSERT INTO scenarios (name, description) VALUES (%s,%s) RETURNING id;', [scenario['name'], scenario['description']])

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
        cursor.execute('SELECT sv.*, v.description, v.data FROM scenario_variants sv JOIN variants v ON sv.variant_name = v.variant_name WHERE sv.scenario_name=%s', [scenario_name])

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
        cursor.execute('SELECT sv.*, v.description, v.data FROM scenario_variants sv JOIN variants v ON sv.variant_name = v.variant_name WHERE sv.scenario_name=%s AND sv.variant_name=%s', [scenario_name, variant_name])

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
            cursor.execute('INSERT INTO variants (variant_name, description, data) VALUES (%s,%s,%s) RETURNING id;', [variant['variant_name'], variant['description'], json.dumps(variant['data'])])

            # commit changes to database
            self.database_connection.commit()

            # get returned data
            variant_id = cursor.fetchone()

            # run sql call
            cursor.execute('INSERT INTO scenario_variants (scenario_name, variant_name, scenario_id) VALUES (%s,%s,%s) RETURNING id;', [variant['scenario_name'], variant['variant_name'], scenario_id[0]['id']])

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
            cursor.execute('UPDATE variants SET description=%s AND data=%s WHERE variant_name = %s RETURNING id;', [variant['description'], json.dumps(variant['data']), variant_name])

        elif 'description' in variant.keys() and 'data' not in variant.keys():

            # run sql call
            cursor.execute('UPDATE variants SET description=%s WHERE variant_name = %s RETURNING id;', [variant['description'], variant_name])

        elif 'description' not in variant.keys() and 'data' in variant.keys():

            # run sql call
            cursor.execute('UPDATE variants SET data=%s WHERE variant_name = %s RETURNING id;', [json.dumps(variant['data']), variant_name])

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
        cursor.execute('DELETE FROM scenario_variants WHERE scenario_name=%s AND variant_name=%s', [scenario_name, variant_name])

        # commit changes to database
        self.database_connection.commit()

        # run sql call
        cursor.execute('DELETE FROM variants WHERE variant_name=%s', [scenario_name, variant_name])

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
