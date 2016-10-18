"""Test config load and parse
"""
import os
from pytest import raises
import jsonschema
import smif.parse_config

def test_load_simple_config():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)
    assert conf.data()["name"] == "test"

def test_simple_validate_valid():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)
    conf.validate({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"]
        })

def test_simple_validate_invalid():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)

    msg = "'nonexistent_key' is a required property"
    with raises(jsonschema.exceptions.ValidationError, message=msg):
        conf.validate({
            "type": "object",
            "properties": {
                "nonexistent_key": { "type": "string" }
            },
            "required": ["nonexistent_key"]
        })
