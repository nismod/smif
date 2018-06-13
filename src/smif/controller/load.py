import logging

from smif.convert.area import RegionSet
from smif.convert.area import get_register as get_region_register
from smif.convert.interval import IntervalSet
from smif.convert.interval import get_register as get_interval_register
from smif.convert.register import Register
from smif.convert.unit import get_register as get_unit_register
from smif.data_layer import DatafileInterface

LOGGER = logging.getLogger(__name__)

REGIONS = get_region_register()
INTERVALS = get_interval_register()
UNITS = get_unit_register()


def load_region_sets(handler):
    """Loads the region sets into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`

    """
    region_definitions = handler.read_region_definitions()
    for region_def in region_definitions:
        region_def_name = region_def['name']
        LOGGER.info("Reading in region definition %s", region_def_name)
        region_data = handler.read_region_definition_data(region_def_name)
        region_set = RegionSet(region_def_name, region_data)
        REGIONS.register(region_set)


def load_interval_sets(handler):
    """Loads the time-interval sets into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`

    """
    interval_definitions = handler.read_interval_definitions()
    for interval_def in interval_definitions:
        interval_name = interval_def['name']
        LOGGER.info("Reading in interval definition %s", interval_name)
        interval_data = handler.read_interval_definition_data(interval_name)
        interval_set = IntervalSet(interval_name, interval_data)
        INTERVALS.register(interval_set)


def load_units(handler):
    """Loads the units into the project registries

    Parameters
    ----------
    handler: :class:`smif.data_layer.DataInterface`
    """
    unit_file = handler.read_units_file_name()
    if unit_file is not None:
        LOGGER.info("Loading units in from %s", unit_file)
        UNITS.register(unit_file)


def load_resolution_sets(directory):
    """Loads the region, interval units resolution sets

    Arguments
    ---------
    directory: str
        Path to the project directory
    """
    handler = DatafileInterface(directory)
    Register.data_interface = handler
    load_region_sets(handler)
    load_interval_sets(handler)
    load_units(handler)
