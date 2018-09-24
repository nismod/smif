# -*- coding: utf-8 -*-
"""Validate the correct format and presence of the config data
for the system-of-systems model
"""
from smif.exception import ValidationError

VALIDATION_ERRORS = []


def validate_sos_model_config(data):
    """Check expected values for data loaded from master config file
    """
    if not isinstance(data, dict):
        msg = "Main config file should contain setup data, instead found: {}"
        err = ValidationError(msg.format(data))
        VALIDATION_ERRORS.append(err)
        return

    # check dependencies
    if "dependencies" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'dependencies' specified in main config file.")
        )

    # check timesteps
    if "timesteps" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'timesteps' file specified in main config file."))
    else:
        validate_path_to_timesteps(data["timesteps"])

    # check sector models
    if "sector_models" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'sector_models' specified in main config file."))
    else:
        validate_sector_models_initial_config(data["sector_models"])

    # check scenario data
    if "scenario_data" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'scenario_data' specified in main config file."))
    else:
        validate_scenario_data_config(data["scenario_data"])

    # check planning
    if "planning" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'planning' mode specified in main config file."))
    else:
        validate_planning_config(data["planning"])

    # check region_sets
    if "region_sets" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'region_sets' specified in main config file."))
    else:
        validate_region_sets_config(data["region_sets"])

    # check interval_sets
    if "interval_sets" not in data:
        VALIDATION_ERRORS.append(
            ValidationError("No 'interval_sets' specified in main config file."))
    else:
        validate_interval_sets_config(data["interval_sets"])


def validate_path_to_timesteps(timesteps):
    """Check timesteps is a path to timesteps file
    """
    if not isinstance(timesteps, str):
        VALIDATION_ERRORS.append(
            ValidationError(
                "Expected 'timesteps' in main config to specify " +
                "a timesteps file, instead got {}.".format(timesteps)))


def validate_timesteps(timesteps, file_path):
    """Check timesteps is a list of integers
    """
    if not isinstance(timesteps, list):
        msg = "Loading {}: expected a list of timesteps.".format(file_path)
        VALIDATION_ERRORS.append(ValidationError(msg))
    else:
        msg = "Loading {}: timesteps should be integer years, instead got {}"
        for timestep in timesteps:
            if not isinstance(timestep, int):
                VALIDATION_ERRORS.append(msg.format(file_path, timestep))


def validate_time_intervals(intervals, file_path):
    """Check time intervals
    """
    if not isinstance(intervals, list):
        msg = "Loading {}: expected a list of time intervals.".format(file_path)
        VALIDATION_ERRORS.append(ValidationError(msg))
    else:
        for interval in intervals:
            validate_time_interval(interval)


def validate_time_interval(interval):
    """Check a single time interval
    """
    if not isinstance(interval, dict):
        msg = "Expected a time interval, instead got {}.".format(interval)
        VALIDATION_ERRORS.append(ValidationError(msg))
        return

    required_keys = ["id", "start", "end"]
    for key in required_keys:
        if key not in interval:
            fmt = "Expected a value for '{}' in each " + \
                "time interval, only received {}"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key, interval)))


def validate_sector_models_initial_config(sector_models):
    """Check list of sector models initial configuration
    """
    if not isinstance(sector_models, list):
        fmt = "Expected 'sector_models' in main config to " + \
              "specify a list of sector models to run, instead got {}."
        VALIDATION_ERRORS.append(ValidationError(fmt.format(sector_models)))
    else:
        if len(sector_models) == 0:
            VALIDATION_ERRORS.append(
                ValidationError("No 'sector_models' specified in main config file."))

        # check each sector model
        for sector_model_config in sector_models:
            validate_sector_model_initial_config(sector_model_config)


def validate_sector_model_initial_config(sector_model_config):
    """Check a single sector model initial configuration
    """
    if not isinstance(sector_model_config, dict):
        fmt = "Expected a sector model config block, instead got {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(sector_model_config)))
        return

    required_keys = ["name", "config_dir", "path", "classname"]
    for key in required_keys:
        if key not in sector_model_config:
            fmt = "Expected a value for '{}' in each " + \
                  "sector model in main config file, only received {}"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key, sector_model_config)))


def validate_dependency_spec(input_spec, model_name):
    """Check the input specification for a single sector model
    """
    if not isinstance(input_spec, list):
        fmt = "Expected a list of parameter definitions in '{}' model " + \
              "input specification, instead got {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(model_name, input_spec)))
        return

    for dep in input_spec:
        validate_dependency(dep)


def validate_dependency(dep):
    """Check a dependency specification
    """
    if not isinstance(dep, dict):
        fmt = "Expected a dependency specification, instead got {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(dep)))
        return

    required_keys = ["name", "spatial_resolution", "temporal_resolution", "units"]
    for key in required_keys:
        if key not in dep:
            fmt = "Expected a value for '{}' in each model dependency, only received {}"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key, dep)))


def validate_scenario_data_config(scenario_data):
    """Check scenario data
    """
    if not isinstance(scenario_data, list):
        fmt = "Expected a list of scenario datasets in main model config, " + \
              "instead got {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(scenario_data)))
        return

    for scenario in scenario_data:
        validate_scenario(scenario)


def validate_scenario(scenario):
    """Check a single scenario specification
    """
    if not isinstance(scenario, dict):
        fmt = "Expected a scenario specification, instead got {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(scenario)))
        return

    required_keys = ["parameter", "spatial_resolution", "temporal_resolution", "units", "file"]
    for key in required_keys:
        if key not in scenario:
            fmt = "Expected a value for '{}' in each scenario, only received {}"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key, scenario)))


def validate_scenario_data(data, file_path):
    """Check a list of scenario observations
    """
    if not isinstance(data, list):
        fmt = "Expected a list of scenario data in {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(file_path)))
        return

    for datum in data:
        validate_scenario_datum(datum, file_path)


def validate_scenario_datum(datum, file_path):
    """Check a single scenario datum
    """
    if not isinstance(datum, dict):
        fmt = "Expected a scenario data point, instead got {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(datum)))
        return

    required_keys = ["region", "interval", "year", "value"]
    for key in required_keys:
        if key not in datum:
            fmt = "Expected a value for '{}' in each data point in a scenario, " + \
                  "only received {}"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key, datum)))


def validate_initial_conditions(data, file_path):
    """Check a list of initial condition observations
    """
    if not isinstance(data, list):
        fmt = "Expected a list of initial conditions in {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(file_path)))
        return

    for datum in data:
        validate_initial_condition(datum, file_path)


def validate_initial_condition(datum, file_path):
    """Check a single initial condition datum
    """
    if not isinstance(datum, dict):
        fmt = "Expected a initial condition data point, instead got {} from {}"
        VALIDATION_ERRORS.append(ValidationError(fmt.format(datum, file_path)))
        return

    required_keys = ["name", "build_date"]
    for key in required_keys:
        if key not in datum:
            fmt = "Expected a value for '{}' in each data point in a initial condition, " + \
                  "only received {} from {}"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key, datum, file_path)))


def validate_planning_config(planning):
    """Check planning options
    """
    required_keys = ["pre_specified", "rule_based", "optimisation"]
    for key in required_keys:
        if key not in planning:
            fmt = "No '{}' settings specified under 'planning' " + \
                  "in main config file."
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key)))

    # check each planning type
    for key, planning_type in planning.items():
        if "use" not in planning_type:
            fmt = "No 'use' settings specified for '{}' 'planning'"
            VALIDATION_ERRORS.append(ValidationError(fmt.format(key)))
            continue
        if planning_type["use"]:
            if "files" not in planning_type or \
               not isinstance(planning_type["files"], list) or \
               len(planning_type["files"]) == 0:

                fmt = "No 'files' provided for the '{}' " + \
                      "planning type in main config file."
                VALIDATION_ERRORS.append(ValidationError(fmt.format(key)))


def validate_region_sets_config(region_sets):
    """Check regions sets
    """
    required_keys = ["name", "file"]
    for key in required_keys:
        for region_set in region_sets:
            if key not in region_set:
                fmt = "Expected a value for '{}' in each " + \
                    "region set in main config file, only received {}"
                VALIDATION_ERRORS.append(ValidationError(fmt.format(key, region_set)))


def validate_interval_sets_config(interval_sets):
    """Check interval sets
    """
    required_keys = ["name", "file"]
    for key in required_keys:
        for interval_set in interval_sets:
            if key not in interval_set:
                fmt = "Expected a value for '{}' in each " + \
                    "interval set in main config file, only received {}"
                VALIDATION_ERRORS.append(ValidationError(fmt.format(key, interval_set)))


def validate_interventions(data, path):
    """Validate the loaded data as required for model interventions
    """
    # check required keys
    required_keys = ["name", "location", "capital_cost", "operational_lifetime",
                     "economic_lifetime"]

    # except for some keys which are allowed simple values,
    # expect each attribute to be of the form {value: x, units: y}
    simple_keys = ["name", "sector", "location"]

    for intervention in data:
        for key in required_keys:
            if key not in intervention:
                fmt = "Loading interventions from {}, required " + \
                      "a value for '{}' in each intervention, but only " + \
                      "received {}"
                VALIDATION_ERRORS.append(ValidationError(fmt.format(path, key, intervention)))

        for key, value in intervention.items():
            if key not in simple_keys and (
                    not isinstance(value, dict)
                    or "value" not in value
                    or "units" not in value):
                fmt = "Loading interventions from {3}, {0}.{1} was {2} but " + \
                      "should have specified units, " + \
                      "e.g. {{'value': {2}, 'units': 'm'}}"

                msg = fmt.format(intervention["name"], key, value, path)
                VALIDATION_ERRORS.append(ValidationError(msg))
