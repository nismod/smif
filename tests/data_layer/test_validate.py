"""Test config validation
"""
from smif.data_layer.validate import validate_sos_model_config
from smif.exception import SmifDataError


class TestValidateSosModel:
    """Check that validation raises validation errors when one part of the
    configuration is incorrect
    """
    def test_correct(self, get_sos_model, get_sector_model, energy_supply_sector_model,
                     sample_scenarios):
        """Expect no error on the default configuration
        """
        validate_sos_model_config(
            get_sos_model,
            [get_sector_model, energy_supply_sector_model],
            sample_scenarios)

    def test_description_too_long(self, get_sos_model, get_sector_model,
                                  energy_supply_sector_model, sample_scenarios):
        """Expect exception when description is too long
        """
        get_sos_model['description'] = 255 * 'a'
        validate_sos_model_config(
            get_sos_model,
            [get_sector_model, energy_supply_sector_model],
            sample_scenarios)

        try:
            get_sos_model['description'] = 256 * 'a'
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'description'
            assert 'characters' in ex.args[0][0].error

    def test_sector_models_none_configured(self, get_sos_model, get_sector_model,
                                           energy_supply_sector_model, sample_scenarios):
        """Expect exception when no sector_models are configured
        """
        try:
            get_sos_model['sector_models'] = []
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'sector_models'
            assert 'one sector model must be selected' in ex.args[0][0].error

    def test_sector_model_missing_reference(self, get_sos_model, sample_scenarios):
        """Expect error when references sector model
        configuration does not exist
        """
        try:
            validate_sos_model_config(
                get_sos_model,
                [],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'sector_models'
            assert 'valid sector_model configuration' in ex.args[0][0].error

    def test_scenario_missing_reference(self, get_sos_model, get_sector_model,
                                        energy_supply_sector_model, sample_scenarios):
        """Expect error when references scenario
        configuration does not exist
        """
        try:
            get_sos_model['scenarios'] = ['scenario_a']
            sample_scenarios = []
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'scenarios'
            assert 'valid scenario configuration' in ex.args[0][0].error

    def test_dependencies_circular(self, get_sos_model, get_sector_model,
                                   energy_supply_sector_model, sample_scenarios):
        """Expect error when circular dependency is in
        configuration
        """
        get_sos_model['sector_models'] = [
            'energy_demand'
        ]
        get_sos_model['model_dependencies'] = [
            {
                "source": "energy_demand",
                "source_output": "output_a",
                "sink": "energy_demand",
                "sink_input": "input_a"
            }
        ]
        get_sector_model['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        get_sector_model['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Circular dependencies' in ex.args[0][0].error

    def test_dependencies_source_or_sink_not_enabled(self, get_sos_model, get_sector_model,
                                                     energy_supply_sector_model,
                                                     sample_scenarios):
        """Expect error when source or sink is not enabled
        in the sector_model configuration
        """
        get_sos_model['model_dependencies'] = [
            {
                "source": "energy_demand",
                "source_output": "output_a",
                "sink": "energy_supply",
                "sink_input": "input_a"
            }
        ]
        get_sos_model['sector_models'] = [
            'energy_supply'
        ]
        get_sector_model['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        energy_supply_sector_model['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][2].component == 'model_dependencies'
            assert '`energy_demand` is not enabled' in ex.args[0][2].error

    def test_dependencies_source_output_or_sink_input_not_exist(
            self, get_sos_model, get_sector_model, energy_supply_sector_model,
            sample_scenarios):
        """Expect error when source output or sink input do not
        exist
        """
        get_sos_model['model_dependencies'] = [
            {
                "source": "energy_demand",
                "source_output": "output_a",
                "sink": "energy_supply",
                "sink_input": "input_a"
            }
        ]
        get_sector_model['outputs'] = []
        energy_supply_sector_model['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Source output `output_a` does not exist' in ex.args[0][0].error

        get_sector_model['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        energy_supply_sector_model['inputs'] = []

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Sink input `input_a` does not exist' in ex.args[0][0].error

    def test_dependencies_has_matching_specs(self, get_sos_model, get_sector_model,
                                             energy_supply_sector_model, sample_scenarios):
        """Expect error when source output or sink input have
        different specs
        """
        get_sos_model['model_dependencies'] = [
            {
                "source": "energy_demand",
                "source_output": "output_a",
                "sink": "energy_supply",
                "sink_input": "input_a"
            }
        ]
        get_sector_model['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        energy_supply_sector_model['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_b"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'model_dependencies'
            msg = '`output_a` has different dimensions than sink `input_a`'
            assert msg in ex.args[0][0].error

        get_sector_model['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        energy_supply_sector_model['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_b",
                "unit": "unit_a"
            }
        ]

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert ex.args[0]
            assert ex.args[0][0].component == 'model_dependencies'
            msg = '`output_a` has a different dtype than sink `input_a`'
            assert msg in ex.args[0][0].error

    def test_dependencies_sink_input_can_only_be_driven_once(
            self, get_sos_model, get_sector_model, energy_supply_sector_model,
            sample_scenarios):
        """Expect error when sink input is driven by multiple
        sources
        """
        get_sos_model['model_dependencies'] = [
            {
                "source": "energy_demand",
                "source_output": "output_a",
                "sink": "energy_supply",
                "sink_input": "input_a"
            },
            {
                "source": "energy_demand",
                "source_output": "output_b",
                "sink": "energy_supply",
                "sink_input": "input_a"
            }
        ]
        get_sector_model['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            },
            {
                "name": "output_b",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        energy_supply_sector_model['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]

        try:
            validate_sos_model_config(
                get_sos_model,
                [get_sector_model, energy_supply_sector_model],
                sample_scenarios)
            assert False
        except SmifDataError as ex:
            assert len(ex.args[0]) == 2
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Sink input `input_a` is driven by multiple sources' in ex.args[0][0].error
            assert ex.args[0][1].component == 'model_dependencies'
            assert 'Sink input `input_a` is driven by multiple sources' in ex.args[0][0].error
