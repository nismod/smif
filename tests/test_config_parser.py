"""Test config load and parse
"""
import os
import smif.parse_config

def test_load_simple_config():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "config", "simple.yaml")
    conf = smif.parse_config.ConfigParser(path)
    assert conf.data()["name"] == "test"
