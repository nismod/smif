from pytest import raises
from smif.data_layer.model_loader import ModelLoader


def test_path_not_found():
    """Should error if module file is missing at path
    """
    loader = ModelLoader()
    with raises(FileNotFoundError) as ex:
        loader.load({
            'name': 'test',
            'path': '/path/to/model.py',
            'classname': 'WaterSupplySectorModel'
        })
    msg = "Cannot find '/path/to/model.py' for the 'test' model"
    assert msg in str(ex)
