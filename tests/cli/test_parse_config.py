"""Test config load and parse
"""
import os
from pytest import raises
from smif.cli.parse_config import ConfigParser

class TestConfigParser(object):

    def _config_fixtures_dir(self):
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "fixtures",
            "config")

    def test_load_simple_config(self):
        path = os.path.join(self._config_fixtures_dir(), "simple.yaml")
        conf = ConfigParser(path)
        assert conf.data["name"] == "test"

    def test_simple_validate_valid(self):
        conf = ConfigParser()
        conf.data = {"name": "test"}
        conf.validate({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "required": ["name"]
        })

    def test_simple_validate_invalid(self):
        conf = ConfigParser()
        conf.data = {"name": "test"}

        msg = "'nonexistent_key' is a required property"
        with raises(ValueError) as ex:
            conf.validate({
                "type": "object",
                "properties": {
                    "nonexistent_key": {
                        "type": "string"
                    }
                },
                "required": ["nonexistent_key"]
            })
        assert msg in str(ex.value)

    def test_empty_invalid(self):
        conf = ConfigParser()

        msg = "Config data not loaded"
        with raises(AttributeError) as ex:
            conf.validate({})
        assert msg in str(ex.value)

    def test_empty_modelrun_invalid(self):
        conf = ConfigParser()

        msg = "Config data not loaded"
        with raises(AttributeError) as ex:
            conf.validate_as_modelrun_config()
        assert msg in str(ex.value)

    def test_modelrun_config_validate(self):
        path = os.path.join(self._config_fixtures_dir(), "modelrun_config.yaml")
        conf = ConfigParser(path)

        conf.validate_as_modelrun_config()

    def test_missing_timestep(self):
        path = os.path.join(self._config_fixtures_dir(),
                            "modelrun_config_missing_timestep.yaml")
        conf = ConfigParser(path)

        msg = "'timesteps' is a required property"
        with raises(ValueError) as ex:
            conf.validate_as_modelrun_config()
        assert msg in str(ex.value)

    def test_used_planning_needs_files(self):
        path = os.path.join(self._config_fixtures_dir(),
                            "modelrun_config_used_planning_needs_files.yaml")
        conf = ConfigParser(path)

        msg = "A planning type needs files if it is going to be used."
        with raises(ValueError) as ex:
            conf.validate_as_modelrun_config()
        assert msg in str(ex.value)

    def test_assets_checks_for_units(self):
        conf = ConfigParser()
        conf.data = [
            {
                'type': 'asset',
                'capacity': 3,
                'operational_lifetime': {
                    'value': 150,
                    'units': "years"
                },
                'economic_lifetime': {
                    'value': 50,
                    'units': "years"
                },
                'capital_cost': {
                    'value': 50,
                    'units': "million £/km"
                }
            }
        ]

        msg = "asset.capacity was 3 but should have specified units, e.g. " + \
              "{'value': 3, 'units': 'm'}"
        with raises(ValueError) as ex:
            conf.validate_as_assets()
        assert msg in str(ex.value)
