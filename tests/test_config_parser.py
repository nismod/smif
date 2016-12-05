"""Test config load and parse
"""
import os
from pytest import raises
import smif.parse_config

def test_load_simple_config():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)
    assert conf.data["name"] == "test"

def test_simple_validate_valid():
    conf = smif.parse_config.ConfigParser()
    conf.data = {"name": "test"}
    conf.validate({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"]
        })

def test_simple_validate_invalid():
    conf = smif.parse_config.ConfigParser()
    conf.data = {"name": "test"}

    msg = "'nonexistent_key' is a required property"
    with raises(ValueError, message=msg):
        conf.validate({
            "type": "object",
            "properties": {
                "nonexistent_key": { "type": "string" }
            },
            "required": ["nonexistent_key"]
        })

def test_modelrun_config_validate():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "modelrun_config.yaml")
    conf = smif.parse_config.ConfigParser(path)

    conf.validate_as_modelrun_config()

def test_modelrun_config_validate_invalid_missing_timestep():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "modelrun_config_missing_timestep.yaml")
    conf = smif.parse_config.ConfigParser(path)

    with raises(ValueError):
        conf.validate_as_modelrun_config()

def test_modelrun_config_validate_invalid_used_planning_needs_files():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "modelrun_config_used_planning_needs_files.yaml")
    conf = smif.parse_config.ConfigParser(path)

    with raises(ValueError):
        conf.validate_as_modelrun_config()
