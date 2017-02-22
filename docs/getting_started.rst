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

The ``planning`` folder contains one file for each ::

        /planning/pre-specified.yaml

The ``data`` folder contains a subfolder for each sector model::

        /data/<sector_model_1>
        /data/<sector_model_2>

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

       /models/<sector_model>/run.py
       /models/<sector_model>/<executable or library>

System-of-Systems Model File
----------------------------

The ``model.yaml`` file contains the following::

        sector_models:
        - name: energy_supply
          path: ../../models/energy_supply/run.py
          classname: EnergySupplyWrapper
          config_dir: .
          initial_conditions:
          - initial_conditions.yaml
          interventions:
          - interventions.yaml
        timesteps: timesteps.yaml
        planning:
          pre_specified:
            use: true
            files:
            - pre-specified.yaml # The build instructions
          rule_based:
            use: false
            files: []
          optimisation:
            use: false
            files: []
        assets:
        - assets.yaml

System-of-systems Planning Years
--------------------------------

The ``timesteps.yaml`` should contain a list of planning years::

        - 2010
        - 2011
        - 2012

This is a list of planning years over which the system of systems model will
run.

Inputs File
-----------

The ``inputs.yaml`` file defines the dependencies of one model upon another.
Enter a list of dependencies, each with four keys, ``name``, 
``spatial_resolution``, ``temporal_resolution`` and ``from_model``.
For example, in energy supply::

        dependencies: 
        - name: electricity_demand
          spatial_resolution: DEFAULT
          temporal_resolution: DEFAULT
          from_model: [energy_demand, transport]
        - name: gas_demand
          spatial_resolution: DEFAULT
          temporal_resolution: DEFAULT
          from_model: energy_demand

The keys ``spatial_resolution`` and ``temporal_resolution`` define the 
resolution at which the data are required.  ``from_model`` defines the model
from which the dependendency is required.

Outputs File
------------

The ``outputs.yaml`` file defines the output metrics from the model.
For example::

        metrics:
          - name: total_cost
          - name: water_demand
          - name: total_emissions

Wrapping a Sector Model
-----------------------

To integrate a sector model into the system-of-systems model, it is necessary
to write a Python wrapper, 
which implements :class:`smif.sector_model`.

The key methods which need to be overridden are:

- :py:meth:`smif.sector_model.SectorModel.simulate`
- :py:meth:`smif.sector_model.SectorModel.get_results`
- :py:meth:`smif.sector_model.SectorModel.extract_obj`

The path to the location of the ``run.py`` file should be entered in the
``model.yaml`` file under the ``path`` key 
(see System-of-Systems Model File above).

Interventions
~~~~~~~~~~~~~

Define all possible interventions in an ``interventions.yaml`` file.
For example::

        - name: nuclear_power_station
          capital_cost:
            value: 3.5
            units: £(million)/MW
          economic_lifetime:
            value: 30
            units: years
          operational_life:
            value: 40
            units: years
          operational_Year:
            value: 2030
            units: year
          capacity:
            value: 1000
            units: MW
          location:
            value: England
            units: string
          power_generation_type:
            value: 4
            units: number
        - name: IOG_gas_terminal_expansion
          capital_cost:
            value: 10
            units: £(million)/mcm
          economic_lifetime:
            value: 25
            units: years
          operational_life:
            value: 30
            units: years
          operational_Year:
            value: 2020
            units: year
          capacity:
            value: 10
            units: mcm
          location:
            value: England
            units: string
          gas_terminal_number:
            value: 8
            units: number

Existing Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~

Define existing infrasture in an ``initial_conditions.yaml`` file.

Planning
--------

Pre-Specified Planning
~~~~~~~~~~~~~~~~~~~~~~

Define a pipeline of interventions in a ``pre-specified.yaml`` file::

        - name: nuclear_power_station
          build_date: 2017
          location:
            lat: 51.745560
            lon: -1.240528

Rule Based Planning
~~~~~~~~~~~~~~~~~~~

This feature is not yet implemented

Optimisation
~~~~~~~~~~~~

This feature is not yet implemented