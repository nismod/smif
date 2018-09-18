"""Implements procedures and tools to run a system-of-systems model
Provides methods for copying a sample project into a local directory
and for or running a system-of-systems model.

Provides a ModelRunScheduler that can run modelruns as a subprocess.

Example
-------
Copy a project folder::

>>> copy_project_folder('/projects/smif/')

Run a single system-of-systems model::

>>> execute_model_run('energy_supply_demand', '/projects/smif/')

Use the ModelRunScheduler to run a system-of-systems model::

    >>> my_scheduler = ModelRunScheduler()
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
#         from smif.controller import ModelRunScheduler
from smif.controller.scheduler import ModelRunScheduler
from smif.controller.execute import execute_model_run
from smif.controller.setup import copy_project_folder
from smif.controller.modelrun import ModelRunner

# Define what should be imported as * ::
#         from smif.controller import *
__all__ = ['ModelRunner', 'ModelRunScheduler', 'execute_model_run', 'copy_project_folder']
