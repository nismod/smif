"""Implements procedures and tools to run a system-of-systems model
Provides methods for copying a sample project into a local directory
and for or running a system-of-systems model.


Example
-------
Copy a project folder::

>>> copy_project_folder('/projects/smif/')

Run a single system-of-systems model::

>>> execute_model_run('energy_supply_demand', store)

Use the SubProcessRunScheduler to run a system-of-systems model::

    >>> my_scheduler = SubProcessRunScheduler()
    >>> my_scheduler.add(
            'energy_supply_demand',
            {
                'directory': '/projects/smif'
            }
        )
    >>> my_scheduler.status('energy_supply_demand')
    {
        'message': 'running'
    }
    >>> my_scheduler.status('energy_supply_demand')
    {
    "err": "",
    "output": "20102010\\n2015\\n2015\\nModel run '20170918_energy_water' complete\\n",
    "status": "done"
    }
"""

# import classes for access like ::
#         from smif.controller import ModelRunner
from smif.controller.execute_run import execute_model_run
from smif.controller.execute_step import (execute_decision_step,
                                          execute_model_before_step,
                                          execute_model_step)
from smif.controller.modelrun import ModelRunner
from smif.controller.setup import copy_project_folder

# Define what should be imported as * ::
#         from smif.controller import *
__all__ = ['ModelRunner', 'execute_decision_step', 'execute_model_before_step',
           'execute_model_run', 'execute_model_step', 'copy_project_folder']
