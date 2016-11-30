"""Tests the ModelOutputs class
"""
from smif.outputs import ModelOutputs


class TestOutputFile:

    def test_outputs_file(self, water_outputs_contents):
        contents = water_outputs_contents
        mo = ModelOutputs(contents)

        expected = ['unshfl13']
        actual = mo.outputs
        assert actual == expected

        expected = ['storage_blobby', 'storage_state']
        actual = mo.metrics
        assert actual == expected

    def test_parse_results_iterable(self, water_outputs_contents):
        contents = water_outputs_contents
        mo = ModelOutputs(contents)
        # Iterable by filename which allows ordered searching through file to
        # extract results
        expected = {'model/results.txt': {'storage_state': (26, 44),
                                          'storage_blobby': (33, 55)
                                          }
                    }
        actual = mo._metrics.extract_iterable
        assert actual == expected

        expected = {'model/results.txt': {'unshfl13': (33, 44)
                                          }
                    }
        actual = mo._outputs.extract_iterable
        assert actual == expected

    def test_parse_results_file(self, setup_results_file,
                                water_outputs_contents):

        base_folder = setup_results_file
        contents = water_outputs_contents
        mo = ModelOutputs(contents)
        actual = mo._metrics.get_results(str(base_folder))
        expected = {'storage_state': '200288', 'storage_blobby': '9080'}

        assert actual == expected
