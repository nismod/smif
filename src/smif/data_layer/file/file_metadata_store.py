"""File-backed metadata store
"""
import copy
import os
from functools import lru_cache
from logging import getLogger

import pandas
from pandas import compat as pandas_compat
from pandas.core import common as pandas_common
from ruamel.yaml import YAML
from smif.data_layer.abstract_metadata_store import MetadataStore
from smif.exception import SmifDataNotFoundError, SmifDataReadError

# Import fiona if available (optional dependency)
try:
    import fiona
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
    def read_unit_definitions(self):
        try:
            with open(self.units_path, 'r') as units_fh:
                return [line.strip() for line in units_fh]
        except FileNotFoundError:
            self.logger.warn('Units file not found, expected at %s', str(self.units_path))
            return []

    def write_unit_definitions(self, units):
        with open(self.units_path, 'w') as units_fh:
            units_fh.writelines(units)
    # endregion

    # region Dimensions
    def read_dimensions(self, skip_coords=False):
        dim_names = _read_filenames_in_dir(self.config_folder, '.yml')
        return [self.read_dimension(name, skip_coords) for name in dim_names]

    def read_dimension(self, dimension_name, skip_coords=False):
        dim = _read_yaml_file(self.config_folder, dimension_name)
        if skip_coords:
            del dim['elements']
        else:
            dim['elements'] = self._read_dimension_file(dim['elements'])
        return dim

    def write_dimension(self, dimension):
        # write elements to yml file (by default, can handle any nested data)
        elements_filename = "{}.yml".format(dimension['name'])
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and add to config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename
        _write_yaml_file(self.config_folder, dimension['name'], dimension_with_ref)

    def update_dimension(self, dimension_name, dimension):
        # look up elements filename and write elements

        old_dim = _read_yaml_file(self.config_folder, dimension_name)
        elements_filename = old_dim['elements']
        elements = dimension['elements']
        self._write_dimension_file(elements_filename, elements)

        # refer to elements by filename and update config
        dimension_with_ref = copy.copy(dimension)
        dimension_with_ref['elements'] = elements_filename

        _write_yaml_file(self.config_folder, dimension_name, dimension_with_ref)

    def delete_dimension(self, dimension_name):
        # read to find filename

        old_dim = _read_yaml_file(self.config_folder, dimension_name)
        elements_filename = old_dim['elements']
        # remove elements data
        os.remove(os.path.join(self.data_folder, elements_filename))
        # remove description
        os.remove(os.path.join(self.config_folder, "{}.yml".format(dimension_name)))

    @lru_cache(maxsize=32)
    def _read_dimension_file(self, filename):
        filepath = os.path.join(self.data_folder, filename)
        filebasename, ext = os.path.splitext(filename)
        if ext in ('.yml', '.yaml'):
            data = _read_yaml_file(self.data_folder, filebasename)
        elif ext == '.csv':
            dataframe = pandas.read_csv(filepath)
            data = _df_to_records(dataframe)
        elif ext in ('.geojson', '.shp'):
            data = self._read_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', '.yml', '.yaml', "
            msg += "'.geojson', '.shp') when reading {}"
            raise SmifDataReadError(msg.format(ext, filepath))
        return data

    def _write_dimension_file(self, filename, data):
        # lru_cache may now be invalid, so clear it
        self._read_dimension_file.cache_clear()
        path = os.path.join(self.data_folder, filename)
        filebasename, ext = os.path.splitext(filename)
        if ext in ('.yml', '.yaml'):
            _write_yaml_file(self.data_folder, filebasename, data)
        elif ext == '.csv':
            pandas.DataFrame.from_records(data).to_csv(path, index=False)
        elif ext in ('.geojson', '.shp'):
            raise NotImplementedError("Writing spatial dimensions not yet supported")
            # self._write_spatial_file(filepath)
        else:
            msg = "Extension '{}' not recognised, expected one of ('.csv', '.yml', '.yaml', "
            msg += "'.geojson', '.shp') when writing {}"
            raise SmifDataReadError(msg.format(ext, path))
        return data
    # endregion

    @staticmethod
    def _read_spatial_file(filepath):
        try:
            with fiona.drivers():
                with fiona.open(filepath) as src:
                    data = []
                    for f in src:
                        element = {
                            'name': f['properties']['name'],
                            'feature': f
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


def _df_to_records(df):
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
        for row in df.itertuples(index=False)
    ]
