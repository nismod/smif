"""Tests the ModelOutputs class
"""
from smif.outputs import ModelOutputs


class TestOutputFile:

    def test_outputs_file(self, water_outputs_contents):
        contents = water_outputs_contents
        mo = ModelOutputs(contents)

        expected = ['unshfl13']
        actual = mo.outputs.names
        assert actual == expected

        assert 'storage_blobby' in mo.metrics.names
        assert 'storage_state' in mo.metrics.names

    def test_parse_results_iterable(self, water_outputs_contents):
        contents = water_outputs_contents
        mo = ModelOutputs(contents)
        # Iterable by filename which allows ordered searching through file to
        # extract results
        expected = {
            'model/results.txt': {
                'storage_state': (26, 44),
                'storage_blobby': (33, 55)
            }
        }
        actual = mo.metrics.file_locations
        assert actual == expected

        expected = {
            'model/results.txt': {
                'unshfl13': (33, 44)
            }
        }
        actual = mo.outputs.file_locations
        assert actual == expected

    def test_parse_results_file(self, setup_results_file,
                                water_outputs_contents):

        base_folder = setup_results_file
        contents = water_outputs_contents

        mo = ModelOutputs(contents)
        mo.load_results_from_files(str(base_folder))

        assert mo.metrics['storage_state']['value'] == '200288'
        assert mo.metrics['storage_blobby']['value'] == '9080'
