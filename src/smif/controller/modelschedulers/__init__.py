"""
Provides all the model run schedulers.
"""

# import classes for access like ::
#         from smif.controller.modelschedulers import ModelRunScheduler
from smif.controller.modelschedulers.defaultscheduler import ModelRunScheduler
from smif.controller.modelschedulers.dafnischeduler import DafniScheduler

# Define what should be imported as * ::
#         from smif.controller.modelschedulers import *
__all__ = ['ModelRunScheduler', 'DafniScheduler']
