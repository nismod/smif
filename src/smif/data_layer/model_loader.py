"""ModelLoader reads python modules as specified at runtime, loading and instantiating
objects.
"""
import importlib
import logging
import os


class ModelLoader(object):
    """Load Model from config

    Examples
    --------
    Call :py:meth:`ModelLoader.load` to create, load and return a
    :class:`Model` object.

    >>> loader = ModelLoader()
    >>> sector_model = loader.load(sector_model_config)
    >>> conversion_model = loader.load(conversion_model_config)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load(self, config):
        """Loads the model class specified by the config, returns an instance of that class
        using the Model.from_dict method.

        Arguments
        ---------
        config : dict
            The model configuration data. Must include:
                - name (name for smif internal use)
                - path (absolute path to python module file)
                - classname (name of Model implementation class)
                - anything required by the Model.from_dict classmethod

        Returns
        -------
        :class:`~smif.model.Model`
        """
        klass = self.load_model_class(config['name'], config['path'], config['classname'])
        return klass.from_dict(config)

    def load_model_class(self, model_name, model_path, classname):
        """Dynamically load model class

        Arguments
        ---------
        model_name : str
            The name used internally to identify the SectorModel
        model_path : str
            The path to the python module which contains the SectorModel
            implementation
        classname : str
            The name of the class of the SectorModel implementation

        Returns
        -------
        class
            The SectorModel implementation
        """
        if not os.path.exists(model_path):
            msg = "Cannot find '{}' for the '{}' model".format(model_path, model_name)
            raise FileNotFoundError(msg)

        msg = "Importing model %s as class %s from module at %s"
        self.logger.info(msg, model_name, classname, model_path)

        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        klass = module.__dict__[classname]
        return klass
