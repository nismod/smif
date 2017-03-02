# -*- coding: utf-8 -*-
"""Validate config data for the system-of-systems model
"""


class ValidationError(Exception):
    """Custom exception to use for parsing validation.
    """
    pass


def validate_sos_model_config(data):
    """Check expected values for data loaded from master config file
    """
    # check timesteps
    if "timesteps" not in data:
        raise ValidationError("No 'timesteps' file specified in main config file.")
    validate_path_to_timesteps(data["timesteps"])

    # check sector models
    if "sector_models" not in data:
        raise ValidationError("No 'sector_models' specified in main config file.")
    validate_sector_models_initial_config(data["sector_models"])

    # check planning
    if "planning" not in data:
        raise ValidationError("No 'planning' mode specified in main config file.")
    validate_planning_config(data["planning"])


def validate_path_to_timesteps(timesteps):
    """Check timesteps is a path to timesteps file
    """
    if not isinstance(timesteps, str):
        raise ValidationError("Expected 'timesteps' in main config to specify " +
                              "a timesteps file, instead got {}.".format(timesteps))


def validate_timesteps(timesteps, file_path):
    """Check timesteps is a path to timesteps file
    """
    if not isinstance(timesteps, list):
        raise ValidationError("Loading {}: expected a list of timesteps.".format(file_path))


def validate_sector_models_initial_config(sector_models):
    """Check list of sector models initial configuration
    """
    if not isinstance(sector_models, list):
        fmt = "Expected 'sector_models' in main config to " + \
              "specify a list of sector models to run, instead got {}."
        raise ValidationError(fmt.format(sector_models))

    if len(sector_models) == 0:
        raise ValidationError("No 'sector_models' specified in main config file.")

    # check each sector model
    for sector_model_config in sector_models:
        validate_sector_model_initial_config(sector_model_config)


def validate_sector_model_initial_config(sector_model_config):
    """Check a single sector model initial configuration
    """
    required_keys = ["name", "config_dir", "path", "classname"]
    for key in required_keys:
        if key not in sector_model_config:
            fmt = "Expected a value for '{}' in each " + \
                  "sector model in main config file, only received {}"
            raise ValidationError(fmt.format(key, sector_model_config))


def validate_planning_config(planning):
    """Check planning options
    """
    required_keys = ["pre_specified", "rule_based", "optimisation"]
    for key in required_keys:
        if key not in planning:
            fmt = "No '{}' settings specified under 'planning' " + \
                  "in main config file."
            raise ValidationError(fmt.format(key))

    # check each planning type
    for key, planning_type in planning.items():
        if "use" not in planning_type:
            raise ValidationError("")
        if planning_type["use"]:
            if "files" not in planning_type or \
               not isinstance(planning_type["files"], list) or \
               len(planning_type["files"]) == 0:

                fmt = "No 'files' provided for the '{}' " + \
                      "planning type in main config file."
                raise ValidationError(fmt.format(key))


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
                fmt = "Loading interventions from {}, required" + \
                        "a value for '{}' in each intervention, but only " + \
                        "received {}"
                raise ValidationError(fmt.format(path, key, intervention))

        for key, value in intervention.items():
            if key not in simple_keys and (
                    not isinstance(value, dict)
                    or "value" not in value
                    or "units" not in value):
                fmt = "Loading interventions from {3}, {0}.{1} was {2} but " + \
                      "should have specified units, " + \
                      "e.g. {{'value': {2}, 'units': 'm'}}"

                msg = fmt.format(intervention["name"], key, value, path)
                raise ValidationError(msg)
