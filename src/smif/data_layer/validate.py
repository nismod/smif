# -*- coding: utf-8 -*-
"""Validate the correct format and presence of the config data
for the system-of-systems model
"""
import itertools

from smif.exception import (SmifDataError, SmifDataInputError,
                            SmifValidationError)

VALIDATION_ERRORS = []


def validate_sos_model_format(sos_model):
    errors = []

    if not isinstance(sos_model, dict):
        msg = "Main config file should contain setup data, instead found: {}"
        err = SmifValidationError(msg.format(sos_model))
        errors.append(err)
        return sos_model

    default_keys = {
        'name': '',
        'description': '',
        'sector_models': [],
        'scenarios': [],
        'narratives': [],
        'model_dependencies': [],
        'scenario_dependencies': []
    }

    # Add default values to missing keys
    for key, value in default_keys.items():
        if key not in sos_model:
            sos_model[key] = value

    # Report keys that should not be in the config
    for key, value in sos_model.items():
        if key not in default_keys:
            errors.append(
                SmifValidationError(
                    'Invalid key `%s` in sos_model configuration `%s`.'
                    % (key, sos_model['name']))
            )

    # Throw collection of errors
    if errors:
        raise SmifDataError(errors)

    return sos_model


def validate_sos_model_config(sos_model, sector_models, scenarios):
    """Check expected values for data loaded from master config file
    """
    errors = []

    if not isinstance(sos_model, dict):
        msg = "Main config file should contain setup data, instead found: {}"
        err = SmifValidationError(msg.format(sos_model))
        errors.append(err)
        return

    # check description
    errors.extend(_validate_description(sos_model))

    # check sector models
    errors.extend(_validate_sos_model_models(sos_model, sector_models))

    # check scenarios
    errors.extend(_validate_sos_model_scenarios(sos_model, scenarios))

    # check narratives
    errors.extend(_validate_sos_model_narratives(sos_model, sector_models))

    # check dependencies
    errors.extend(_validate_sos_model_deps(sos_model, sector_models, scenarios))

    if errors:
        raise SmifDataError(errors)


def _validate_sos_model_models(sos_model, sector_models):
    errors = []
    if not sos_model['sector_models']:
        errors.append(
            SmifDataInputError(
                'sector_models',
                'At least one sector model must be selected.',
                'A system-of-systems model requires to have at least one system ' +
                'enabled to build a valid configuration.'))

    for sector_model in sos_model['sector_models']:
        if sector_model not in [sector_model['name'] for sector_model in sector_models]:
            errors.append(
                SmifDataInputError(
                    'sector_models',
                    '%s must have a valid sector_model configuration.' % (sector_model),
                    'Smif refers to the sector_model-configurations to find ' +
                    'details about a selected sector_model.'))
    return errors


def _validate_sos_model_scenarios(sos_model, scenarios):
    errors = []
    for scenario in sos_model['scenarios']:
        if scenario not in [scenario['name'] for scenario in scenarios]:
            errors.append(
                SmifDataInputError(
                    'scenarios',
                    '%s must have a valid scenario configuration.' % (scenario),
                    'Smif refers to the scenario-configurations to find ' +
                    'details about a selected scenario.'))
    return errors


def _validate_sos_model_narratives(sos_model, sector_models):
    errors = []
    for narrative in sos_model['narratives']:
        # Check provides are valid
        for model_name in narrative['provides']:

            # A narrative can only provides for enabled models
            if model_name not in sos_model['sector_models']:
                errors.append(
                    SmifDataInputError(
                        'narratives',
                        ('Narrative `%s` provides data for model `%s` that is not enabled ' +
                         'in this system-of-systems model.') % (narrative['name'], model_name),
                        'A narrative can only provide for enabled models.'))
            else:
                # A narrative can only provides parameters that exist in the model
                try:
                    sector_model = _pick_sector_model(model_name, sector_models)
                except KeyError:
                    msg = 'Narrative `{}` provides data for model `{}` that is not found.'
                    errors.append(
                        SmifDataInputError(
                            'models',
                            msg.format(narrative['name'], model_name),
                            'A narrative can only provide for existing models.'))
                    sector_model = {'parameters': []}

                parameters = [
                    parameter['name'] for parameter in sector_model['parameters']
                ]
                for provide in narrative['provides'][model_name]:
                    msg = 'Narrative `{}` provides data for non-existing model parameter `{}`'
                    if provide not in parameters:
                        errors.append(
                            SmifDataInputError(
                                'narratives',
                                msg.format(narrative['name'], provide),
                                'A narrative can only provide existing model parameters.'
                            )
                        )

        # Check if all variants are valid
        for variant in narrative['variants']:
            should_provide = list(itertools.chain(*narrative['provides'].values()))
            variant_provides = list(variant['data'].keys())
            if sorted(variant_provides) != sorted(should_provide):
                msg = 'Narrative `{}`, variant `{}` provides incorrect data.'
                errors.append(
                    SmifDataInputError(
                        'narratives',
                        msg.format(narrative['name'], variant['name']),
                        'A variant can only provide data for parameters that are specified ' +
                        'by the narrative.'))
    return errors


def _pick_sector_model(name, models):
    for model in models:
        if model['name'] == name:
            return model
    raise KeyError("Model '{}' not found in models".format(name))


def _validate_sos_model_deps(sos_model, sector_models, scenarios):
    errors = []
    errors.extend(_validate_dependencies(
        sos_model, 'model_dependencies',
        sector_models, 'sector_models',
        sector_models, 'sector_models'
    ))

    errors.extend(_validate_dependencies(
        sos_model, 'scenario_dependencies',
        scenarios, 'scenarios',
        sector_models, 'sector_models'
    ))
    return errors


def _validate_description(configuration):
    errors = []

    if len(configuration['description']) > 255:
        errors.append(
            SmifDataInputError(
                'description',
                'Description must not contain more than 255 characters.',
                'A description should briefly outline a `%s` configuration.'
                % (configuration['name'])))

    return errors


def _validate_dependencies(configuration, conf_key, source, source_key, sink, sink_key):
    errors = []
    for idx, dependency in enumerate(configuration[conf_key]):
        errors.extend(_validate_dependency_cycle(
            idx, dependency, conf_key))
        errors.extend(_validate_dependency_in_sos_model(
            idx, dependency, configuration, conf_key, source_key, sink_key))
        errors.extend(_validate_dependency(
            idx, dependency, conf_key, source, source_key, sink, sink_key))
    return errors


def _validate_dependency_cycle(idx, dependency, conf_key):
    errors = []
    # Circular dependencies are not allowed
    is_current = 'timestep' not in dependency or dependency['timestep'] == 'CURRENT'
    if dependency['source'] == dependency['sink'] and is_current:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Circular dependencies are not allowed.' % (idx + 1),
                'Smif does not support self-dependencies unless the dependency is on ' +
                'output from a previous timestep.'))
    return errors


def _validate_dependency_in_sos_model(idx, dependency, configuration, conf_key, source_key,
                                      sink_key):
    errors = []
    # Source / Sink must be enabled in sos_model config
    if dependency['source'] not in configuration[source_key]:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Source `%s` is not enabled.' %
                (idx + 1, dependency['source']),
                'Each dependency source must be enabled in the sos-model'))

    if dependency['sink'] not in configuration[sink_key]:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Sink `%s` is not enabled.' %
                (idx + 1, dependency['sink']),
                'Each dependency sink must be enabled in the sos-model'))

    # Sink can only have a single dependency
    dep_sinks = [
        (dependency['sink'], dependency['sink_input'])
        for dependency in configuration[conf_key]
    ]
    if dep_sinks.count((dependency['sink'], dependency['sink_input'])) > 1:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Sink input `%s` is driven by multiple sources.'
                % (idx + 1, dependency['sink_input']),
                'A model input can only be driven by a single model output.'))
    return errors


def _validate_dependency(idx, dependency, conf_key, source, source_key, sink,
                         sink_key):
    errors = []
    # Source and sink model configurations must exist
    source_model = [model for model in source if model['name'] == dependency['source']]
    sink_model = [model for model in sink if model['name'] == dependency['sink']]
    if not source_model:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Source `%s` does not exist.' %
                (idx + 1, dependency['source']),
                'Each dependency source must have a `%s` configuration.' %
                (source_key)))
    if not sink_model:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Sink  `%s` does not exist.' %
                (idx + 1, dependency['sink']),
                'Each dependency sink must have a `%s` configuration.' %
                (sink_key)))
    if not sink_model or not source_model:
        # not worth doing further checks if source/sink does not exist
        return errors

    # Source_output and sink_input must exist
    if source_key == 'sector_models':
        source_model_outputs = [
            output for output in source_model[0]['outputs']
            if output['name'] == dependency['source_output']
        ]
    if source_key == 'scenarios':
        source_model_outputs = [
            output for output in source_model[0]['provides']
            if output['name'] == dependency['source_output']
        ]
    sink_model_inputs = [
        input_ for input_ in sink_model[0]['inputs']
        if input_['name'] == dependency['sink_input']
    ]

    if not source_model_outputs:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Source output `%s` does not exist.' %
                (idx + 1, dependency['source_output']),
                'Each dependency source output must exist in the `%s` configuration.' %
                (source_key)))
    if not sink_model_inputs:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Sink input `%s` does not exist.' %
                (idx + 1, dependency['sink_input']),
                'Each dependency sink input must exist in the `%s` configuration.' %
                (sink_key)))
    if not source_model_outputs or not sink_model_inputs:
        # not worth doing further checks if source_output/sink_input does not exist
        return errors

    # Source_output and sink_input must have matching specs
    source_model_output = source_model_outputs[0]
    sink_model_input = sink_model_inputs[0]
    if sorted(source_model_output['dims']) != sorted(sink_model_input['dims']):
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Source `%s` has different dimensions than sink ' % (
                    idx + 1,
                    source_model_output['name']
                ) +
                '`%s` (%s != %s).' % (
                    sink_model_input['name'],
                    source_model_output['dims'],
                    sink_model_input['dims']
                ),
                'Dependencies must have matching dimensions.'))
    if source_model_output['dtype'] != sink_model_input['dtype']:
        errors.append(
            SmifDataInputError(
                conf_key,
                '(Dependency %s) Source `%s` has a different dtype than sink ' % (
                    idx + 1,
                    source_model_output['name'],
                ) +
                '`%s` (%s != %s).' % (
                    sink_model_input['name'],
                    source_model_output['dtype'],
                    sink_model_input['dtype']),
                'Dependencies must have matching data types.'))

    return errors


def validate_path_to_timesteps(timesteps):
    """Check timesteps is a path to timesteps file
    """
    if not isinstance(timesteps, str):
        VALIDATION_ERRORS.append(
            SmifValidationError(
                "Expected 'timesteps' in main config to specify " +
                "a timesteps file, instead got {}.".format(timesteps)))


def validate_timesteps(timesteps, file_path):
    """Check timesteps is a list of integers
    """
    if not isinstance(timesteps, list):
        msg = "Loading {}: expected a list of timesteps.".format(file_path)
        VALIDATION_ERRORS.append(SmifValidationError(msg))
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
        VALIDATION_ERRORS.append(SmifValidationError(msg))
    else:
        for interval in intervals:
            validate_time_interval(interval)


def validate_time_interval(interval):
    """Check a single time interval
    """
    if not isinstance(interval, dict):
        msg = "Expected a time interval, instead got {}.".format(interval)
        VALIDATION_ERRORS.append(SmifValidationError(msg))
        return

    required_keys = ["id", "start", "end"]
    for key in required_keys:
        if key not in interval:
            fmt = "Expected a value for '{}' in each " + \
                "time interval, only received {}"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, interval)))


def validate_sector_models_initial_config(sector_models):
    """Check list of sector models initial configuration
    """
    if not isinstance(sector_models, list):
        fmt = "Expected 'sector_models' in main config to " + \
              "specify a list of sector models to run, instead got {}."
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(sector_models)))
    else:
        if len(sector_models) == 0:
            VALIDATION_ERRORS.append(
                SmifValidationError("No 'sector_models' specified in main config file."))

        # check each sector model
        for sector_model_config in sector_models:
            validate_sector_model_initial_config(sector_model_config)


def validate_sector_model_initial_config(sector_model_config):
    """Check a single sector model initial configuration
    """
    if not isinstance(sector_model_config, dict):
        fmt = "Expected a sector model config block, instead got {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(sector_model_config)))
        return

    required_keys = ["name", "config_dir", "path", "classname"]
    for key in required_keys:
        if key not in sector_model_config:
            fmt = "Expected a value for '{}' in each " + \
                  "sector model in main config file, only received {}"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, sector_model_config)))


def validate_dependency_spec(input_spec, model_name):
    """Check the input specification for a single sector model
    """
    if not isinstance(input_spec, list):
        fmt = "Expected a list of parameter definitions in '{}' model " + \
              "input specification, instead got {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(model_name, input_spec)))
        return

    for dep in input_spec:
        validate_dependency(dep)


def validate_dependency(dep):
    """Check a dependency specification
    """
    if not isinstance(dep, dict):
        fmt = "Expected a dependency specification, instead got {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(dep)))
        return

    required_keys = ["name", "spatial_resolution", "temporal_resolution", "units"]
    for key in required_keys:
        if key not in dep:
            fmt = "Expected a value for '{}' in each model dependency, only received {}"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, dep)))


def validate_scenario_data_config(scenario_data):
    """Check scenario data
    """
    if not isinstance(scenario_data, list):
        fmt = "Expected a list of scenario datasets in main model config, " + \
              "instead got {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(scenario_data)))
        return

    for scenario in scenario_data:
        validate_scenario(scenario)


def validate_scenario(scenario):
    """Check a single scenario specification
    """
    if not isinstance(scenario, dict):
        fmt = "Expected a scenario specification, instead got {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(scenario)))
        return

    required_keys = ["parameter", "spatial_resolution", "temporal_resolution", "units", "file"]
    for key in required_keys:
        if key not in scenario:
            fmt = "Expected a value for '{}' in each scenario, only received {}"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, scenario)))


def validate_scenario_data(data, file_path):
    """Check a list of scenario observations
    """
    if not isinstance(data, list):
        fmt = "Expected a list of scenario data in {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(file_path)))
        return

    for datum in data:
        validate_scenario_datum(datum, file_path)


def validate_scenario_datum(datum, file_path):
    """Check a single scenario datum
    """
    if not isinstance(datum, dict):
        fmt = "Expected a scenario data point, instead got {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(datum)))
        return

    required_keys = ["region", "interval", "year", "value"]
    for key in required_keys:
        if key not in datum:
            fmt = "Expected a value for '{}' in each data point in a scenario, " + \
                  "only received {}"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, datum)))


def validate_initial_conditions(data, file_path):
    """Check a list of initial condition observations
    """
    if not isinstance(data, list):
        fmt = "Expected a list of initial conditions in {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(file_path)))
        return

    for datum in data:
        validate_initial_condition(datum, file_path)


def validate_initial_condition(datum, file_path):
    """Check a single initial condition datum
    """
    if not isinstance(datum, dict):
        fmt = "Expected a initial condition data point, instead got {} from {}"
        VALIDATION_ERRORS.append(SmifValidationError(fmt.format(datum, file_path)))
        return

    required_keys = ["name", "build_date"]
    for key in required_keys:
        if key not in datum:
            fmt = "Expected a value for '{}' in each data point in a initial condition, " + \
                  "only received {} from {}"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, datum, file_path)))


def validate_planning_config(planning):
    """Check planning options
    """
    required_keys = ["pre_specified", "rule_based", "optimisation"]
    for key in required_keys:
        if key not in planning:
            fmt = "No '{}' settings specified under 'planning' " + \
                  "in main config file."
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key)))

    # check each planning type
    for key, planning_type in planning.items():
        if "use" not in planning_type:
            fmt = "No 'use' settings specified for '{}' 'planning'"
            VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key)))
            continue
        if planning_type["use"]:
            if "files" not in planning_type or \
               not isinstance(planning_type["files"], list) or \
               len(planning_type["files"]) == 0:

                fmt = "No 'files' provided for the '{}' " + \
                      "planning type in main config file."
                VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key)))


def validate_region_sets_config(region_sets):
    """Check regions sets
    """
    required_keys = ["name", "file"]
    for key in required_keys:
        for region_set in region_sets:
            if key not in region_set:
                fmt = "Expected a value for '{}' in each " + \
                    "region set in main config file, only received {}"
                VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, region_set)))


def validate_interval_sets_config(interval_sets):
    """Check interval sets
    """
    required_keys = ["name", "file"]
    for key in required_keys:
        for interval_set in interval_sets:
            if key not in interval_set:
                fmt = "Expected a value for '{}' in each " + \
                    "interval set in main config file, only received {}"
                VALIDATION_ERRORS.append(SmifValidationError(fmt.format(key, interval_set)))


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
                VALIDATION_ERRORS.append(
                    SmifValidationError(fmt.format(path, key, intervention)))

        for key, value in intervention.items():
            if key not in simple_keys and (
                    not isinstance(value, dict)
                    or "value" not in value
                    or "units" not in value):
                fmt = "Loading interventions from {3}, {0}.{1} was {2} but " + \
                      "should have specified units, " + \
                      "e.g. {{'value': {2}, 'units': 'm'}}"

                msg = fmt.format(intervention["name"], key, value, path)
                VALIDATION_ERRORS.append(SmifValidationError(msg))
