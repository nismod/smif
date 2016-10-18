"""Test config load and parse
"""
import os
from pytest import raises
import smif.parse_config

def test_load_simple_config():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)
    assert conf.data()["name"] == "test"

def test_simple_validate_valid():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)
    conf.validate({"name": "string"})


def test_simple_validate_invalid():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)

    msg = "Expected a value in the config file for nonexistent_key"
    with raises(ValueError, message=msg):
        conf.validate({"nonexistent_key": "string"})
