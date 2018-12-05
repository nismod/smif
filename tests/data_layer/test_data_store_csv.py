"""Test CSV data store
"""
import csv
import os
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
from pytest import fixture, mark, raises
from smif.data_layer.data_array import DataArray
from smif.data_layer.datafile_interface import CSVDataStore
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
    spec = Spec(
        name='people',
        unit='people',
        dtype='int',
        dims=['county', 'season'],
        coords={
            'county': ['oxford'],
            'season': ['cold_month', 'spring_month', 'hot_month', 'fall_month']
        }
    )
    return data, spec


@mark.xfail
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

    def test_get_state_filename_all(self, config_handler):

        handler = config_handler

        modelrun_name = 'a modelrun'
        timestep = 2010
        decision_iteration = 0

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)

        expected = os.path.join(
                handler.results_folder, modelrun_name,
                'state_2010_decision_0.csv')

        assert actual == expected

    def test_get_state_filename_none_iteration(self, config_handler):
        handler = config_handler
        modelrun_name = 'a modelrun'
        timestep = 2010
        decision_iteration = None

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)
        expected = os.path.join(handler.results_folder, modelrun_name, 'state_2010.csv')

        assert actual == expected

    def test_get_state_filename_both_none(self, config_handler):
        handler = config_handler
        modelrun_name = 'a modelrun'
        timestep = None
        decision_iteration = None

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)
        expected = os.path.join(
            handler.results_folder, modelrun_name, 'state_0000.csv')

        assert actual == expected

    def test_get_state_filename_timestep_none(self, config_handler):
        handler = config_handler

        modelrun_name = 'a modelrun'
        timestep = None
        decision_iteration = 0

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)
        expected = os.path.join(
            handler.results_folder, modelrun_name, 'state_0000_decision_0.csv')

        assert actual == expected


@mark.xfail
class TestScenarios:
    """Scenario data should be readable, metadata is currently editable. May move to make it
    possible to import/edit/write data.
    """
    def test_scenario_data(self, setup_folder_structure, config_handler, get_scenario_data):
        """ Test to dump a scenario (CSV) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        scenario_data = get_scenario_data

        keys = scenario_data[0].keys()
        filepath = os.path.join(str(basefolder), 'data', 'scenarios', 'population_high.csv')
        with open(filepath, 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        variant = config_handler.read_scenario_variant('population', 'High Population (ONS)')
        variant['data'] = {'population_count': filepath}
        config_handler.update_scenario_variant('population', 'High Population (ONS)', variant)

        data = np.array([[100, 150, 200, 210]])
        actual = config_handler.read_scenario_variant_data(
            'population',
            'High Population (ONS)',
            'population_count',
            timestep=2017)

        spec = Spec.from_dict({
            'name': "population_count",
            'description': "The count of population",
            'unit': 'people',
            'dtype': 'int',
            'coords': {'county': ['oxford'],
                       'season': ['cold_month', 'spring_month', 'hot_month', 'fall_month']},
            'dims': ['county', 'season']})

        expected = DataArray(spec, data)

        assert actual == expected

    def test_scenario_data_raises(self, setup_folder_structure, config_handler,
                                  get_faulty_scenario_data):
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

        variant = config_handler.read_scenario_variant('population', 'High Population (ONS)')
        variant['data'] = {'population_count': filepath}
        config_handler.update_scenario_variant('population', 'High Population (ONS)', variant)

        with raises(SmifDataMismatchError):
            config_handler.read_scenario_variant_data(
                'population',
                'High Population (ONS)',
                'population_count',
                timestep=2017)

    def test_scenario_data_validates(self, setup_folder_structure, config_handler,
                                     get_remapped_scenario_data):
        """Store performs validation of scenario data against raw interval and region data.

        As such `len(region_names) * len(interval_names)` is not a valid size
        of scenario data under cases where resolution definitions contain
        remapping/resampling info (i.e. multiple hours in a year/regions mapped
        to one name).

        The set of unique region or interval names can be used instead.
        """
        basefolder = setup_folder_structure
        scenario_data, spec = get_remapped_scenario_data

        keys = scenario_data[0].keys()
        filepath = os.path.join(str(basefolder), 'data', 'scenarios', 'population_high.csv')
        with open(filepath, 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        variant = config_handler.read_scenario_variant('population', 'High Population (ONS)')
        variant['data'] = {'population_count': filepath}
        config_handler.update_scenario_variant('population', 'High Population (ONS)', variant)

        expected_data = np.array([[100, 150, 200, 210]], dtype=float)
        actual = config_handler.read_scenario_variant_data(
            'population',
            'High Population (ONS)',
            'population_count',
            timestep=2015)

        expected = DataArray(spec, expected_data)

        assert actual == expected


@fixture(scope='function')
def setup_narratives(config_handler, get_sos_model):
    config_handler.write_sos_model(get_sos_model)


# need to test with spec and new methods
@mark.usefixtures('setup_narratives')
@mark.xfail
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

        test_variant = None
        test_narrative = None
        sos_model = config_handler.read_sos_model('energy')
        for narrative in sos_model['narratives']:
            if narrative['name'] == 'governance':
                test_narrative = narrative
                for variant in narrative['variants']:
                    if variant['name'] == 'Central Planning':
                        test_variant = variant
                        break
                break

        test_variant['data'] = {'homogeneity_coefficient': filepath}
        test_narrative['variants'] = [test_variant]
        sos_model['narratives'] = [test_narrative]
        config_handler.update_sos_model('energy', sos_model)

        actual = config_handler.read_narrative_variant_data(
            'energy', 'governance', 'Central Planning', 'homogeneity_coefficient')

        spec = Spec.from_dict({
            'name': 'homogeneity_coefficient',
            'description': "How homegenous the centralisation process is",
            'absolute_range': [0, 1],
            'expected_range': [0, 1],
            'unit': 'percentage',
            'dtype': 'float'
        })

        assert actual == DataArray(spec, np.array(8, dtype=float))

    def test_narrative_data_missing(self, config_handler):
        """Should raise a SmifDataNotFoundError if narrative has no data
        """
        with raises(SmifDataNotFoundError):
            config_handler.read_narrative_variant_data(
                'energy', 'governance', 'Central Planning', 'does not exist')

    def test_default_data_mismatch(self, config_handler, get_sector_model_parameter_defaults):
        sector_model_name = 'energy_demand'
        parameter_name = 'smart_meter_savings'
        data = get_sector_model_parameter_defaults[parameter_name]
        data.data = np.array([1, 2, 3])
        config_handler.write_sector_model_parameter_default(
            sector_model_name, parameter_name, data)

        with raises(SmifDataMismatchError) as ex:
            config_handler.read_model_parameter_default(
                sector_model_name, parameter_name)

        msg = "Reading default parameter values for energy_demand:smart_meter_savings"
        assert msg in str(ex)


# need to test with spec replacing spatial/temporal resolution
@mark.xfail
class TestResults:
    """Results from intermediate stages of running ModelRuns should be writeable and readable.
    """
    def test_read_results(self, setup_folder_structure, get_handler_csv,
                          get_handler_binary):
        """Results from .csv in a folder structure which encodes metadata
        in filenames and directory structure.

        With no decision/iteration specifiers:
            results/
            <modelrun_name>/
            <model_name>/
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv
        Else:
            results/
            <modelrun_name>/
            <model_name>/
            decision_<id>/
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv
        """
        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'
        output = 'electricity_demand'
        timestep = 2020
        spatial_resolution = 'lad'
        temporal_resolution = 'annual'

        # 1. case with no decision
        expected = np.array([[[1.0]]])
        csv_contents = "region,interval,value\noxford,1,1.0\n"
        binary_contents = pa.serialize(expected).to_buffer()

        path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model,
            "decision_none",
            "output_{}_timestep_{}".format(
                output,
                timestep
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + '.csv', 'w') as fh:
            fh.write(csv_contents)
        actual = get_handler_csv.read_results(modelrun, model, output,
                                              spatial_resolution,
                                              temporal_resolution, timestep)
        assert actual == expected

        with pa.OSFile(path + '.dat', 'wb') as f:
            f.write(binary_contents)
        actual = get_handler_binary.read_results(modelrun, model, output,
                                                 spatial_resolution,
                                                 temporal_resolution, timestep)
        assert actual == expected

        # 2. case with decision
        decision_iteration = 1
        expected = np.array([[[2.0]]])
        csv_contents = "region,interval,value\noxford,1,2.0\n"
        binary_contents = pa.serialize(expected).to_buffer()

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
        actual = get_handler_csv.read_results(modelrun, model, output,
                                              spatial_resolution,
                                              temporal_resolution, timestep,
                                              None, decision_iteration)
        assert actual == expected

        with pa.OSFile(path + '.dat', 'wb') as f:
            f.write(binary_contents)
        actual = get_handler_binary.read_results(modelrun, model, output,
                                                 spatial_resolution,
                                                 temporal_resolution, timestep,
                                                 None, decision_iteration)
        assert actual == expected


# TODO - move test (and implementations) up to test_store
@mark.xfail
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


@mark.xfail
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
