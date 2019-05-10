from smif import cli

from argparse import ArgumentParser

from smif.data_layer.file import (CSVDataStore, FileMetadataStore,
                                  ParquetDataStore, YamlConfigStore)

arguments = ['prepare-convert', 'energy_central']

parser = cli.parse_arguments()
args = parser.parse_args(arguments)

print('Now calling function {}'.format(args.func.__name__))

args.func(args)
