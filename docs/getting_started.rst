.. _getting_started

Getting Started 
===============

To wrap a sector model, you need to write a model configuration using yaml
files.

Setup a new system-of-systems modelling project with the following structure::

        /config
        /planning
        /data
        /models

The ``config`` folder contains the configuration for the system-of-systems
model::

        /config/model.yaml
        /config/timesteps.yaml

The ``model.yaml`` file contains the following::

The ``timesteps.yaml`` contains the following::


The ``planning`` folder contains one file for each ::

        /planning/pre-specified.yaml


The ``data`` folder contains a subfolder for each sector model::

        /data/<sector_model_1>
        /data/<sector_mdoel_2>


The ``/data/<sector_model>`` folder contains all the configuration files for a
particular sector model.  See adding a sector model for more information.::

        /data/<sector_model>/inputs.yaml
        /data/<sector_model>/outputs.yaml
        /data/<sector_model>/time_intervals.yaml
        /data/<sector_model>/regions.geojson
        /data/<sector_model>/interventions/
        /data/<sector_model>/assets/

The ``/models/<sector_model/`` contains the executable for a sector model,
as well as a Python file which implements :class:`smif.sector_model.SectorModel`
and provides a way for `smif` to run the model, and access model outputs.
See adding a sector model for more information.::


       /models/<sector_model>/model_wrapper.py
       /models/<sector_model>/<executable or library>