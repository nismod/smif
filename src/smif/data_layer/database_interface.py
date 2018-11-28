"""Database store implementations
"""
# import psycopg2 to handel database transactions
import psycopg2
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
        scenario_id = cursor.fethall()

        if scenario_id is not None:

            # need to check for and delete any scenario variants that are associated with this modelurn
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
        raise NotImplementedError()

    def read_scenario_variant(self, scenario_name, variant_name):
        raise NotImplementedError()

    def write_scenario_variant(self, scenario_name, variant):
        raise NotImplementedError()

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        raise NotImplementedError()

    def delete_scenario_variant(self, scenario_name, variant_name):
        raise NotImplementedError()
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
