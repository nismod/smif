# -*- coding: utf-8 -*-
"""Data access modules for loading system-of-systems model configuration
"""

# import classes for access like ::
#         from smif.data_layer import DataHandle`
from smif.data_layer.data_handle import DataHandle
from smif.data_layer.store import Store

# Define what should be imported as * ::
#         from smif.data_layer import *
__all__ = ['DataHandle', 'Store']
