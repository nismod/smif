"""Test config load and parse
"""
import os
from pytest import raises
from smif.cli.parse_config import ConfigParser

class TestConfigParser(object):

    def __init__(self):
        self.config_fixtures_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "fixtures",
            "config")

    def test_load_simple_config(self):
        path = os.path.join(self.config_fixtures_dir, "simple.yaml")
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
        with raises(ValueError, message=msg):
            conf.validate({
                "type": "object",
                "properties": {
                    "nonexistent_key": {
                        "type": "string"
                    }
                },
                "required": ["nonexistent_key"]
            })

    def test_modelrun_config_validate(self):
        path = os.path.join(self.config_fixtures_dir, "modelrun_config.yaml")
        conf = ConfigParser(path)

        conf.validate_as_modelrun_config()

    def test_missing_timestep(self):
        path = os.path.join(self.config_fixtures_dir,
                            "modelrun_config_missing_timestep.yaml")
        conf = ConfigParser(path)

        with raises(ValueError):
            conf.validate_as_modelrun_config()

    def test_used_planning_needs_files(self):
        path = os.path.join(self.config_fixtures_dir,
                            "modelrun_config_used_planning_needs_files.yaml")
        conf = ConfigParser(path)

        with raises(ValueError):
            conf.validate_as_modelrun_config()
