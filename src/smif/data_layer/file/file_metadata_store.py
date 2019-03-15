"""File-backed metadata store
"""
import copy
import json
import os
from functools import lru_cache
from logging import getLogger
from typing import Dict, List, Union

import pandas  # type: ignore
from pandas import compat as pandas_compat
from pandas.core import common as pandas_common  # type: ignore
from ruamel.yaml import YAML  # type: ignore
from smif.data_layer.abstract_metadata_store import MetadataStore
from smif.exception import SmifDataNotFoundError, SmifDataReadError

# Import fiona if available (optional dependency)
try:
    import fiona  # type: ignore
except ImportError:
    pass


class FileMetadataStore(MetadataStore):
    """File-based metadata store (supports YAML, CSV, or GDAL-compatible files)
    """
    def __init__(self, base_folder):
        super().__init__()
        self.logger = getLogger(__name__)

        base_folder = str(base_folder)
        self.units_path = os.path.join(base_folder, 'data', 'user-defined-units.txt')
        self.data_folder = os.path.join(base_folder, 'data', 'dimensions')
        self.config_folder = os.path.join(base_folder, 'config', 'dimensions')

    # region Units
    def read_unit_definitions(self) -> List[str]:
        try:
            with open(self.units_path, 'r') as units_fh:
                return [line.strip() for line in units_fh]
        except FileNotFoundError:
            self.logger.warning('Units file not found, expected at %s', str(self.units_path))
            return []

    def write_unit_definitions(self, definitions: List[str]):
        with open(self.units_path, 'w') as units_fh:
            units_fh.writelines(definitions)
    # endregion

    # region Dimensions
    def read_dimensions(self, skip_coords=False) -> List[dict]:
        dim_names = _read_filenames_in_dir(self.config_folder, '.yml')
        return [self.read_dimension(name, skip_coords) for name in dim_names]

    def read_dimension(self, dimension_name: str, skip_coords=False):
        dim = _read_yaml_file(self.config_folder, dimension_name)
        if skip_coords:
            del dim['elements']
        else:
            dim['elements'] = self._read_dimension_file(dim['elements'])
        return dim

    def write_dimension(self, dimension: Dict):
        # write elements to csv file (by default, can handle any nested data)
        elements_filename = "{}.csv".format(dimension['name'])
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and add to config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename
        _write_yaml_file(self.config_folder, dimension['name'], dimension_with_ref)

    def update_dimension(self, dimension_name: str, dimension: Dict):
        # look up elements filename and write elements

        old_dim = _read_yaml_file(self.config_folder, dimension_name)
        elements_filename = old_dim['elements']
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and update config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename

        _write_yaml_file(self.config_folder, dimension_name, dimension_with_ref)

    def delete_dimension(self, dimension_name: str):
        # read to find filename

        old_dim = _read_yaml_file(self.config_folder, dimension_name)
        elements_filename = old_dim['elements']
        # remove elements data
        os.remove(os.path.join(self.data_folder, elements_filename))
        # remove description
        os.remove(os.path.join(self.config_folder, "{}.yml".format(dimension_name)))

    @lru_cache(maxsize=32)
    def _read_dimension_file(self, filename: str) -> List[Dict]:
        filepath = os.path.join(self.data_folder, filename)
        filebasename, ext = os.path.splitext(filename)
        if ext == '.csv':
            dataframe = pandas.read_csv(filepath)
            data = _df_to_records(dataframe)
            if 'interval' in data[0]:
                data = self._unstringify_interval(data)
        elif ext in ('.geojson', '.shp'):
            data = self._read_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', "
            msg += "'.geojson', '.shp') when reading {}"
            raise SmifDataReadError(msg.format(ext, filepath))
        return data

    def _write_dimension_file(self, filename: str, data: List[Dict]):
        # lru_cache may now be invalid, so clear it
        self._read_dimension_file.cache_clear()
        path = os.path.join(self.data_folder, filename)
        filebasename, ext = os.path.splitext(filename)
        if ext == '.csv':
            if 'interval' in data[0]:
                data = self._stringify_interval(data)
            pandas.DataFrame.from_records(data).to_csv(path, index=False)
        elif ext in ('.geojson', '.shp'):
            raise NotImplementedError("Writing spatial dimensions not yet supported")
            # self._write_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', "
            msg += "'.geojson', '.shp') when writing {}"
            raise SmifDataReadError(msg.format(ext, path))
        return data

    def _stringify_interval(self, data: List[Dict]) -> List[Dict]:
        output = []
        for item in data:
            output_item = copy.copy(item)
            try:
                output_item['interval'] = json.dumps(item['interval'])
            except KeyError:
                self.logger.warning("Expected interval in element %s", item)
            output.append(output_item)
        return output

    def _unstringify_interval(self, data: List[Dict]) -> List[Dict]:
        output = []
        for item in data:
            output_item = copy.copy(item)
            try:
                output_item['interval'] = json.loads(item['interval'])
            except KeyError:
                self.logger.warning("Expected interval in element %s", item)
            output.append(output_item)
        return output
    # endregion

    @staticmethod
    def _read_spatial_file(filepath) -> List[Dict]:
        try:
            with fiona.drivers():
                with fiona.open(filepath) as src:
                    data = []
                    for feature in src:
                        element = {
                            'name': feature['properties']['name'],
                            'feature': feature
                        }
                        data.append(element)
            return data
        except NameError as ex:
            msg = "Could not read spatial dimension definition '%s' " % (filepath)
            msg += "Please install fiona to read geographic data files. Try running: \n"
            msg += "    pip install smif[spatial]\n"
            msg += "or:\n"
            msg += "    conda install fiona shapely rtree\n"
            raise SmifDataReadError(msg) from ex
        except IOError as ex:
            msg = "Could not read spatial dimension definition '%s' " % (filepath)
            msg += "Please verify that the path is correct and "
            msg += "that the file is present on this location."
            raise SmifDataNotFoundError(msg) from ex


def _read_yaml_file(directory, name):
    """Parse yaml config file into plain data (lists, dicts and simple values)

    Parameters
    ----------
    directory : str
    name : str
        file basename (without yml extension)
    """
    path = os.path.join(directory, "{}.yml".format(name))
    with open(path, 'r') as file_handle:
        return YAML().load(file_handle)


def _write_yaml_file(directory, name, data):
    """Write plain data to a file as yaml

    Parameters
    ----------
    directory : str
    name : str
        file basename (without yml extension)
    data
        Data to write (should be lists, dicts and simple values)
    """
    path = os.path.join(directory, "{}.yml".format(name))
    with open(path, 'w') as file_handle:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.allow_unicode = True
        return yaml.dump(data, file_handle)


def _read_filenames_in_dir(path, extension):
    """Returns the name of the Yaml files in a certain directory

    Arguments
    ---------
    path: str
        Path to directory
    extension: str
        Extension of files (such as: '.yml' or '.csv')

    Returns
    -------
    list
        The list of files in `path` with extension
    """
    files = []
    for filename in os.listdir(path):
        if filename.endswith(extension):
            basename, _ = os.path.splitext(filename)
            files.append(basename)
    return files


def _df_to_records(dataframe: pandas.DataFrame) -> List[Dict]:
    """Fix pandas conversion to list[dict] with python scalar values

    Ported here from future release of pandas 0.24.0

    See:
    - PR: https://github.com/pandas-dev/pandas/pull/23921
    - Issue: https://github.com/pandas-dev/pandas/issues/23753

    Note that this skips the pandas_common,maybe_box_datetimelike implementation, which may be
    desired but relies on more pandas internals so is not copied over (yet).
    """
    into_c = pandas_common.standardize_mapping(dict)
    return [
        into_c(
            (k, v)
            for k, v in pandas_compat.iteritems(row._asdict())
        )
        for row in dataframe.itertuples(index=False)
    ]
