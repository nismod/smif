"""Test CSV data store
"""
import csv
import os
from tempfile import TemporaryDirectory

import numpy as np
from pytest import fixture, mark, raises
from smif.data_layer.data_array import DataArray
from smif.data_layer.file.file_data_store import CSVDataStore
from smif.exception import SmifDataMismatchError, SmifDataNotFoundError
from smif.metadata import Spec


@fixture
def config_handler(setup_folder_structure):
    handler = CSVDataStore(str(setup_folder_structure))
    return handler


@fixture
def get_scenario_data():
    """Return sample scenario_data
    """
    return [
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'timestep': 2017
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'timestep': 2017
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'timestep': 2017
        },
        {
            'population_count': 210,
            'county': 'oxford',
            'season': 'fall_month',
            'timestep': 2017
        },
    ]


@fixture
def get_faulty_scenario_data():
    """Return faulty sample scenario_data
    """
    return [
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'year': 2017
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'year': 2017
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'year': 2017
        },
        {
            'population_count': 210,
            'county': 'oxford',
            'season': 'fall_month',
            'year': 2017
        },
    ]


@fixture(scope='function')
def get_remapped_scenario_data():
    """Return sample scenario_data
    """
    data = [
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'timestep': 2015
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'timestep': 2015
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'timestep': 2015
        },
        {
            'population_count': 210,
            'county': 'oxford',
            'season': 'fall_month',
            'timestep': 2015
        },
        {
            'population_count': 100,
            'county': 'oxford',
            'season': 'cold_month',
            'timestep': 2016
        },
        {
            'population_count': 150,
            'county': 'oxford',
            'season': 'spring_month',
            'timestep': 2016
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'hot_month',
            'timestep': 2016
        },
        {
            'population_count': 200,
            'county': 'oxford',
            'season': 'fall_month',
            'timestep': 2016
        }
    ]
    return data


@fixture
def scenario_spec():
    return Spec(
        name='population_count',
        unit='people',
        dtype='int',
        dims=['county', 'season'],
        coords={
            'county': ['oxford'],
            'season': ['cold_month', 'spring_month', 'hot_month', 'fall_month']
        }
    )


class TestReadState:
    def test_read_state(self, config_handler):
        handler = config_handler

        modelrun_name = 'modelrun'
        timestep = 2010
        decision_iteration = 0
        dir_ = os.path.join(handler.results_folder, modelrun_name)
        path = os.path.join(dir_, 'state_2010_decision_0.csv')
        os.makedirs(dir_, exist_ok=True)
        with open(path, 'w') as state_fh:
            state_fh.write("build_year,name\n2010,power_station")

        actual = handler.read_state(modelrun_name, timestep, decision_iteration)
        expected = [{'build_year': 2010, 'name': 'power_station'}]
        assert actual == expected


class TestScenarios:
    """Scenario data should be readable, metadata is currently editable. May move to make it
    possible to import/edit/write data.
    """
    def test_scenario_data_raises(self, setup_folder_structure, config_handler,
                                  get_faulty_scenario_data, scenario_spec):
        """If a scenario file has incorrect keys, raise a friendly error identifying
        missing keys
        """
        basefolder = setup_folder_structure
        scenario_data = get_faulty_scenario_data

        keys = scenario_data[0].keys()
        filepath = os.path.join(str(basefolder), 'data', 'scenarios', 'population_high.csv')
        with open(filepath, 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        key = 'population_high.csv'
        with raises(SmifDataMismatchError):
            config_handler.read_scenario_variant_data(key, scenario_spec, 2017)

    def test_scenario_data_validates(self, setup_folder_structure, config_handler,
                                     get_remapped_scenario_data, scenario_spec):
        """Store performs validation of scenario data against raw interval and region data.

        As such `len(region_names) * len(interval_names)` is not a valid size
        of scenario data under cases where resolution definitions contain
        remapping/resampling info (i.e. multiple hours in a year/regions mapped
        to one name).

        The set of unique region or interval names can be used instead.
        """
        basefolder = setup_folder_structure
        scenario_data = get_remapped_scenario_data
        spec = scenario_spec

        keys = scenario_data[0].keys()
        filepath = os.path.join(str(basefolder), 'data', 'scenarios', 'population_high.csv')
        with open(filepath, 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        expected_data = np.array([[100, 150, 200, 210]], dtype=float)
        expected = DataArray(spec, expected_data)

        actual = config_handler.read_scenario_variant_data('population_high.csv', spec, 2015)
        assert actual == expected


class TestNarrativeVariantData:
    """Narratives, parameters and interventions should be readable, metadata is editable. May
    move to clarify the distinctions here, and extend to specify strategies and contraints.
    """
    def test_narrative_data(self, setup_folder_structure, config_handler, get_narrative):
        """ Test to dump a narrative (yml) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        filepath = os.path.join(
            str(basefolder), 'data', 'narratives', 'central_planning.csv')
        with open(filepath, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['homogeneity_coefficient'])
            writer.writeheader()
            writer.writerow({'homogeneity_coefficient': 8})

        spec = Spec.from_dict({
            'name': 'homogeneity_coefficient',
            'description': "How homegenous the centralisation process is",
            'absolute_range': [0, 1],
            'expected_range': [0, 1],
            'unit': 'percentage',
            'dtype': 'float'
        })

        actual = config_handler.read_narrative_variant_data('central_planning.csv', spec)
        assert actual == DataArray(spec, np.array(8, dtype=float))

    def test_narrative_data_missing(self, config_handler):
        """Should raise a SmifDataNotFoundError if narrative has no data
        """
        with raises(SmifDataNotFoundError):
            config_handler.read_narrative_variant_data('does not exist', None)

    def test_default_data_mismatch(self, config_handler, get_sector_model_parameter_defaults):
        parameter_name = 'smart_meter_savings'
        spec = get_sector_model_parameter_defaults[parameter_name].spec
        path = os.path.join(config_handler.data_folders['parameters'], 'default.csv')
        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[parameter_name])
            writer.writeheader()
            for i in range(4):
                writer.writerow({parameter_name: i})

        with raises(SmifDataMismatchError):
            config_handler.read_model_parameter_default('default.csv', spec)


class TestResults:
    """Results from intermediate stages of running ModelRuns should be writeable and readable.
    """
    def test_read_results(self, setup_folder_structure, config_handler):
        """Results from .csv in a folder structure which encodes metadata
        in filenames and directory structure.

        On the pattern of:
            <modelrun_name>/
            <model_name>/
            decision_<id>/
                output_<output_name>_
                timestep_<timestep>.csv
        """
        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'
        decision_iteration = 1
        output = 'electricity_demand'
        timestep = 2020
        output_spec = Spec(
            name='electricity_demand',
            unit='MWh',
            dtype='float',
            dims=['region', 'interval'],
            coords={
                'region': ['oxford'],
                'interval': [1]
            }
        )

        expected_data = np.array([[2.0]])
        expected = DataArray(output_spec, expected_data)
        csv_contents = "region,interval,electricity_demand\noxford,1,2.0\n"

        path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model,
            "decision_{}".format(decision_iteration),
            "output_{}_timestep_{}".format(
                output,
                timestep
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + '.csv', 'w') as fh:
            fh.write(csv_contents)
        actual = config_handler.read_results(
            modelrun, model, output_spec, timestep, decision_iteration)
        assert actual == expected


@mark.skip(reason="Move to test available_results implementation")
class TestWarmStart:
    """If re-running a ModelRun with warm-start specified explicitly, results should be checked
    for existence and left in place.
    """
    def test_prepare_warm_start(self, setup_folder_structure):
        """ Confirm that the warm start copies previous model results
        and reports the correct next timestep
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = CSVDataStore(str(basefolder))

        # Create results for a 'previous' modelrun
        previous_results_path = os.path.join(
            str(setup_folder_structure),
            "results", modelrun, model,
            "decision_none"
        )
        os.makedirs(previous_results_path, exist_ok=True)

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2020.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,4.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2025.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,6.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2030.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,8.0\n")

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep == 2030

        # Confirm that previous results (excluding the last timestep) exist
        current_results_path = os.path.join(
            str(setup_folder_structure),
            "results", modelrun, model,
            "decision_none"
        )

        warm_start_results = os.listdir(current_results_path)

        assert 'output_electricity_demand_timestep_2020.csv' in warm_start_results
        assert 'output_electricity_demand_timestep_2025.csv' in warm_start_results
        assert 'output_electricity_demand_timestep_2030.csv' not in warm_start_results

    def test_prepare_warm_start_other_local_storage(self, setup_folder_structure):
        """ Confirm that the warm start does not work when previous
        results were saved using a different local storage type
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = CSVDataStore(str(basefolder))

        # Create results for a 'previous' modelrun
        previous_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(previous_results_path, exist_ok=True)

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2020.parquet")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,4.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2025.parquet")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,6.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2030.parquet")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,8.0\n")

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep is None

    def test_prepare_warm_start_no_previous_results(self, setup_folder_structure):
        """ Confirm that the warm start does not work when no previous
        results were saved
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = CSVDataStore(str(basefolder))

        # Create results for a 'previous' modelrun
        previous_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(previous_results_path, exist_ok=True)

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep is None

        # Confirm that no results were copied
        current_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(current_results_path, exist_ok=True)
        assert len(os.listdir(current_results_path)) == 0

    def test_prepare_warm_start_no_previous_modelrun(self, setup_folder_structure):
        """ Confirm that the warm start does not work when no previous
        modelrun occured
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = CSVDataStore(str(basefolder))

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep is None

        # Confirm that no results were copied
        current_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(current_results_path, exist_ok=True)
        assert len(os.listdir(current_results_path)) == 0


class TestCoefficients:
    """Dimension conversion coefficients should be cached to disk and read if available.
    """
    @fixture
    def from_spec(self):
        return Spec(name='from_test_coef', dtype='int')

    @fixture
    def to_spec(self):
        return Spec(name='to_test_coef', dtype='int')

    def test_read_write(self, from_spec, to_spec, config_handler):
        data = np.eye(1000)
        handler = config_handler
        handler.write_coefficients(from_spec, to_spec, data)
        actual = handler.read_coefficients(from_spec, to_spec)
        np.testing.assert_equal(actual, data)

    def test_read_raises(self, from_spec, to_spec, config_handler):
        handler = config_handler
        missing_spec = Spec(name='missing_coef', dtype='int')
        with raises(SmifDataNotFoundError):
            handler.read_coefficients(missing_spec, to_spec)

    def test_dfi_raises_if_folder_missing(self):
        """Ensure we can write files, even if project directory starts empty
        """
        with TemporaryDirectory() as tmpdirname:
            # start with empty project (no data/coefficients subdirectory)
            with raises(SmifDataNotFoundError):
                CSVDataStore(tmpdirname)
