"""Test config validation
"""
from pytest import fixture, raises
from smif.data_layer.validate import validate_sos_model_config
from smif.exception import SmifDataError, SmifDataInputError


@fixture(scope='function')
def get_sos_models_config():
    """Return minimum sos_model config
    """
    return [
        {
            "name": "sos_model",
            "description": "",
            "sector_models": [
                'sector_model_a'
            ],
            "model_dependencies": [],
            "scenario_dependencies": [],
            "scenarios": [],
        }
    ]

@fixture(scope='function')
def get_sector_models_config():
    """Return minimum sector_model config
    """
    return [
        {
            "name": "sector_model_a",
            "description": "",
            "classname": "",
            "initial_conditions": [],
            "inputs": [],
            "interventions": [],
            "outputs": [],
            "parameters": [],
            "path": "",
            "active": True
        },
        {
            "name": "sector_model_b",
            "description": "",
            "classname": "",
            "initial_conditions": [],
            "inputs": [],
            "interventions": [],
            "outputs": [],
            "parameters": [],
            "path": "",
            "active": True
        },
    ]

@fixture(scope='function')
def get_scenarios_config():
    """Return minimum scenarios config
    """
    return [
    {
        "name": "scenario_1",
        "description": "",
        "provides": [
            {
                "name": "provides_1",
                "description": "",
                "dims": [
                    "country"
                ],
                "dtype": "float",
                "unit": "ml"
            }
        ],
        "variants": [
            {
                "name": "variant_1",
                "description": "",
                "data": {
                    "provides_1": "variant_1.csv"
                },
            }
        ],
        "active": True
    }
]

class TestValidateSosModel:
    """Check that validation raises validation errors when one part of the
    configuration is incorrect
    """
    def test_correct(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect no error on the default configuration
        """
        validate_sos_model_config(
            get_sos_models_config[0], 
            get_sector_models_config,
            get_scenarios_config)

    def test_description_too_long(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect exception when description is too long
        """
        get_sos_models_config[0]['description'] = 255 * 'a'
        validate_sos_model_config(
            get_sos_models_config[0], 
            get_sector_models_config,
            get_scenarios_config)

        try:
            get_sos_models_config[0]['description'] = 256 * 'a'
            validate_sos_model_config(
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'description'
            assert 'characters' in ex.args[0][0].error

    def test_sector_models_none_configured(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect exception when no sector_models are configured
        """
        try:
            get_sos_models_config[0]['sector_models'] = []
            validate_sos_model_config(
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'sector_models'
            assert 'one sector model must be selected' in ex.args[0][0].error

    def test_sector_model_missing_reference(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when references sector model 
        configuration does not exist
        """
        try:
            get_sector_models_config = []
            validate_sos_model_config(
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'sector_models'
            assert 'valid sector_model configuration' in ex.args[0][0].error

    def test_scenario_missing_reference(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when references scenario 
        configuration does not exist
        """
        try:
            get_sos_models_config[0]['scenarios'] = ['scenario_a']
            get_scenarios_config = []
            validate_sos_model_config(
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'scenarios'
            assert 'valid scenario configuration' in ex.args[0][0].error

    def test_dependencies_circular(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when circular dependency is in 
        configuration
        """
        get_sos_models_config[0]['sector_models'] = [
            'sector_model_a'
        ]
        get_sos_models_config[0]['model_dependencies'] = [
            {
                "source": "sector_model_a",
                "source_output": "output_a",
                "sink": "sector_model_a",
                "sink_input": "input_a"
            }
        ]
        get_sector_models_config[0]['inputs'] = [
            {
                "name": "input_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        get_sector_models_config[0]['outputs'] = [
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
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Circular dependencies' in ex.args[0][0].error
    
    def test_dependencies_source_or_sink_not_enabled(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when source or sink is not enabled
        in the sector_model configuration
        """
        get_sos_models_config[0]['model_dependencies'] = [
            {
                "source": "sector_model_a",
                "source_output": "output_a",
                "sink": "sector_model_b",
                "sink_input": "input_a"
            }
        ]
        get_sos_models_config[0]['sector_models'] = [
            'sector_model_b'
        ]
        get_sector_models_config[0]['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        get_sector_models_config[1]['inputs'] = [
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
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'model_dependencies'
            assert '`sector_model_a` is not enabled' in ex.args[0][0].error

    def test_dependencies_source_output_or_sink_input_not_exist(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when source output or sink input do not
        exist
        """
        get_sos_models_config[0]['model_dependencies'] = [
            {
                "source": "sector_model_a",
                "source_output": "output_a",
                "sink": "sector_model_b",
                "sink_input": "input_a"
            }
        ]
        get_sos_models_config[0]['sector_models'] = [
            "sector_model_a",
            "sector_model_b"
        ]
        get_sector_models_config[0]['outputs'] = []
        get_sector_models_config[1]['inputs'] = [
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
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Source output `output_a` does not exist' in ex.args[0][0].error

        get_sector_models_config[0]['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        get_sector_models_config[1]['inputs'] = []

        try:
            validate_sos_model_config(
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Sink input `input_a` does not exist' in ex.args[0][0].error

    def test_dependencies_has_matching_specs(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when source output or sink input have
        different specs
        """
        get_sos_models_config[0]['model_dependencies'] = [
            {
                "source": "sector_model_a",
                "source_output": "output_a",
                "sink": "sector_model_b",
                "sink_input": "input_a"
            }
        ]
        get_sos_models_config[0]['sector_models'] = [
            "sector_model_a",
            "sector_model_b"
        ]
        get_sector_models_config[0]['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        get_sector_models_config[1]['inputs'] = [
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
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'model_dependencies'
            assert '`output_a` has different dimensions than sink `input_a`' in ex.args[0][0].error

        get_sector_models_config[0]['outputs'] = [
            {
                "name": "output_a",
                "dims": [
                    "dim_a"
                ],
                "dtype": "dtype_a",
                "unit": "unit_a"
            }
        ]
        get_sector_models_config[1]['inputs'] = [
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
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 1
            assert ex.args[0][0].component == 'model_dependencies'
            assert '`output_a` has a different dtype than sink `input_a`' in ex.args[0][0].error

    def test_dependencies_sink_input_can_only_be_driven_once(
        self, get_sos_models_config, get_sector_models_config, 
        get_scenarios_config):
        """Expect error when sink input is driven by multiple
        sources
        """
        get_sos_models_config[0]['model_dependencies'] = [
            {
                "source": "sector_model_a",
                "source_output": "output_a",
                "sink": "sector_model_b",
                "sink_input": "input_a"
            },
            {
                "source": "sector_model_a",
                "source_output": "output_b",
                "sink": "sector_model_b",
                "sink_input": "input_a"
            }
        ]
        get_sos_models_config[0]['sector_models'] = [
            "sector_model_a",
            "sector_model_b"
        ]
        get_sector_models_config[0]['outputs'] = [
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
        get_sector_models_config[1]['inputs'] = [
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
                get_sos_models_config[0], 
                get_sector_models_config,
                get_scenarios_config)
            assert False
        except(SmifDataError) as ex:
            assert len(ex.args[0]) == 2
            assert ex.args[0][0].component == 'model_dependencies'
            assert 'Sink input `input_a` is driven by multiple sources' in ex.args[0][0].error
            assert ex.args[0][1].component == 'model_dependencies'
            assert 'Sink input `input_a` is driven by multiple sources' in ex.args[0][0].error