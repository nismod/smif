.. _getting_started:

Getting Started
===============

There are three layers of configuration in order to use the simulation modelling
integration framework to conduct system-of-system modelling.

A project is the highest level container which holds all the elements required 
to run models, configure simulation models and define system-of-system models.

Project Configuration
---------------------

The basic folder structure looks like this::

  project.yml
    /config
      /sector_models
        energy_demand.yml
        energy_supply.yml
      /sos_models
        energy.yml
      /model_runs
        20170918_energy.yml
    /data
      /initial_conditions
        energy_demand_existing.yml
        energy_supply_existing.yml
      /intervals
        hourly.csv
        annual.csv
      /interventions
        energy_demand.yml
      /narratives
        energy_demand_high_tech.yml
        central_planning.yml
      /regions
        lad.shp
      /scenarios
        population_high.csv
      units.yml

The folder structure is divided into a ``config`` subfolder and a ``data``
subfolder.

The Configuration Folder
------------------------

This folder holds configuration and metadata on simulation models, 
system-of-system models and model runs.

The Project File
~~~~~~~~~~~~~~~~

This file holds all the project configuration.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :caption: project.yml
   :language: yaml
   :linenos:

We'll step through this configuration file section by section.

The first line gives the project name, a unique identifier for this project.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 1

The next section lists the scenario sets. These give the categories into which
scenarios are collected.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 2-6

Narrative sets collect together the categories into which narrative files are
collected.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 7-9
  
Region definitions list the collection of region files and the mapping to a 
unique name which can be used in scenarios and sector models.
Region definitions define the spatial resolution of data.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 10,12-17

Interval definitions list the collection of interval files and the mapping to a 
unique name which can be used in scenarios and sector models.
Interval definitions define the temporal resolution of data.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 18,20-22

Unit definitions references a file containing custom units, not included in
the Pint library default unit register (e.g. non-SI units).

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 23

The ``scenarios`` section lists the scenarios and corresponding collections of
data associated with scenarios.

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 24,26-43

The ``narratives`` section lists the narratives and mapping to one or more
narrative files

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 44-48

A Simulation Model File
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/config/sector_models/energy_demand.yml
   :language: yaml


A System-of-System Model File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/config/sos_models/energy_water.yml
   :language: yaml

A Model Run File
~~~~~~~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml

Data Folder
-----------

This folder holds data like information to define the spatial and 
temporal resolution of data, 
as well as exogenous environmental data held in scenarios.

Initial Conditions
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/data/initial_conditions/reservoirs.yml
   :language: yaml

Interval definitions
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/data/interval_definitions/annual_intervals.yml
   :language: csv

Region definitions
~~~~~~~~~~~~~~~~~~

Cannot view shape files.

Interventions
~~~~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/data/interventions/water_supply.yml
   :language: yaml

Narratives
~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/data/narratives/high_tech_dsm.yml
   :language: yaml

Scenarios
~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/data/scenarios/population.csv
   :language: csv


The metadata required to define a particular scenario are shown in the table
below.
It is possible to associate a number of different data sets with
the same scenario, so that, for example, choosing the `High Population`
scenario allows users to access both population count and density data
in the same or different spatial and temporal resolutions.

| Attribute | Type | Example | Notes |
| --- | --- | --- | --- |
| name | string | `High Population (ONS)` | |
| description | string | `The High ONS Forecast for UK population out to 2050` ||
| scenario_set | string | `population` | |
| parameters | list | [see below](./smif-prerequisites.html#scenario-parameters) | |

#### Scenario Parameters

For each entry in the scenario parameters list, the following metadata
is required:

| Attribute | Type | Example | Notes |
| --- | --- | --- | --- |
| name | string | `density` ||
| spatial_resolution | string | `lad` ||
| temporal_resolution |string | `annual` ||
| units | string | `people/km^2` ||
| filename | string | `population_density_high.csv` | Name of the file in the `project/data/scenarios` folder |




To specify a system-of-systems model, you must configure one or more simulation
models, outlined in the section below, and configure a system-of-systems
model, as outlined immediately below.

First, setup a new system-of-systems modelling project with the following
folder structure::

        /config
        /planning
        /data
        /models

This folder structure is optional, but helps organise the configuration files,
which can be important as the number and complexity of simulation models
increases.

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
        /data/<sector_model>/interventions.yaml

The ``/models/<sector_model/`` contains the executable for a sector model,
as well as a Python file which implements :class:`smif.sector_model.SectorModel`
and provides a way for `smif` to run the model, and access model outputs.
See adding a sector model for more information.::

       /models/<sector_model>/run.py
       /models/<sector_model>/<executable or library>

System-of-Systems Model File
----------------------------

The ``model.yaml`` file contains the following::

        timesteps: timesteps.yaml
        region_sets:
        - name: energy_regions
          file: regions.shp
        interval_sets:
        - name: energy_timeslices
          file: time_intervals.yaml
        - name: annual_interval
          file: annual_interval.yaml
        scenario_data:
        - file: electricity_demand.yaml
          parameter: electricity_demand
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval
        - file: gas_demand.yaml
          parameter: gas_demand
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval
        sector_models:
        - name: energy_supply
          path: ../../../models/energy_supply/run.py
          classname: EnergySupplyWrapper
          config_dir: .
          initial_conditions:
          - initial_conditions.yaml
          interventions:
          - interventions.yaml
        planning:
          pre_specified:
            use: true
            files:
            - pre-specified.yaml
          rule_based:
            use: false
            files: []
          optimisation:
            use: false
            files: []


System-of-Systems Planning Years
--------------------------------

The ``timesteps.yaml`` should contain a list of planning years::

        - 2010
        - 2011
        - 2012

This is a list of planning years over which the system of systems model will
run. Each of the simulation models will be run once for each
planning year.

Wrapping a Sector Model
-----------------------

To integrate a sector model into the system-of-systems model, it is necessary
to write a Python wrapper,
which implements :class:`smif.sector_model.SectorModel`.

The key methods which need to be overridden are:

- :py:meth:`smif.sector_model.SectorModel.initialise`
- :py:meth:`smif.sector_model.SectorModel.simulate`
- :py:meth:`smif.sector_model.SectorModel.extract_obj`

The wrapper should be written in a python file, e.g. ``run.py``.
The path to the location of this ``run.py`` file should be entered in the
``model.yaml`` file under the ``path`` key
(see System-of-Systems Model File above).

To integrate an infrastructure simulation model within the system-of-systems
modelling framework, it is also necessary to provide the configuration
data.
This configuration data includes definitions of the spatial and temporal resolutions 
of the input and output data to and from the models. 
This enables the framework to convert data from one spatio-temporal resolution 
to another.

Geographies
-----------
Define the set of unique regions which are used within the model as polygons.
The spatial resolution of the model may be implicit, and even a national model
needs to have a national region defined.
Inputs and outputs are assigned a model-specific geography from this list
allowing automatic conversion from and to these geographies.

Model regions are specified in ``regions.*``.

The file format must be possible to parse with GDAL, and must contain
an attribute "name" to use as an identifier for the region.

The sets of geographic regions are specified in the ``model.yaml`` file using
a ``region_sets`` attributes as shown below::

        region_sets:
        - name: energy_regions
          file: regions.shp

This links a name, used elsewhere in the configuration with inputs, outputs and scenarios
with a file containing the geographic data.

Temporal Resolution
-------------------
The attribution of hours in a year to the temporal resolution used
in the sectoral model.

Within-year time intervals are specified in yaml files, and as for regions,
specified in the ``model.yaml`` file with an ``interval_sets`` attribute::

        interval_sets:
        - name: energy_timeslices
          file: time_intervals.yaml
        - name: annual_interval
          file: annual_interval.yaml

This links a unique name with the definitions of the intervals in a yaml file.
The data in the file specify the mapping of model timesteps to durations within a year
(assume modelling 365 days: no extra day in leap years, no leap seconds)

Each time interval must have

- start (period since beginning of year)
- end (period since beginning of year)
- id (label to use when passing between integration layer and sector model)

use ISO 8601 [1]_ duration format to specify periods::

    P[n]Y[n]M[n]DT[n]H[n]M[n]S

For example::

    - end: P7225H
      id: '1_0'
      start: P7224H
    - end: P7226H
      id: '1_1'
      start: P7225H
    - end: P7227H
      id: '1_2'
      start: P7226H
    - end: P7228H
      id: '1_3'
      start: P7227H
    - end: P7229H
      id: '1_4'
      start: P7228H

Inputs
------
Define the collection of inputs required from external sources
to run the model.  For example
"electricity demand (<region>, <interval>)".
Inputs are defined with a name, spatial resolution and temporal-resolution.

Only those inputs required as dependencies are defined here, although
dependencies are activated when configured in the system-of-systems model.

The ``inputs.yaml`` file defines the dependencies of one model upon another.
Enter a list of dependencies, each with three keys, ``name``,
``spatial_resolution`` and ``temporal_resolution``.
For example, in energy supply::

      - name: electricity_demand
        spatial_resolution: energy_regions
        temporal_resolution: annual_interval
      - name: gas_demand
        spatial_resolution: energy_regions
        temporal_resolution: annual_interval

The keys ``spatial_resolution`` and ``temporal_resolution`` define the
resolution at which the data are required.


Outputs
-------
Define the collection of outputs model parameters used for the purpose 
of optimisation or rule-based planning approaches 
(so normally a cost-function), and those
outputs required for accounting purposes, such as operational cost and
emissions, or as a dependency in another model.

The ``outputs.yaml`` file defines the output parameters from the model.
For example::

        - name: total_cost
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval
        - name: water_demand
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval
        - name: total_emissions
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval

Scenarios
---------

The ``scenario_date:`` section of the system-of-systems configuration file allows
you to define static sources for simulation model dependencies.

In the case of the example show above, reproduced below::

        scenario_data:
        - file: electricity_demand.yaml
          parameter: electricity_demand
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval
        - file: gas_demand.yaml
          parameter: gas_demand
          spatial_resolution: energy_regions
          temporal_resolution: annual_interval

we define two yaml files, one each for the parameters `electricity_demand` and `gas_demand`.
The ``temporal_resolution`` attribute allows the use of time intervals in the scenario files which
are at a different temporal resolution to that expected by the sector model.  In this case,
both electricity_demand and gas_demand are linked to the same ``annual_interval.yaml`` file.

The scenario data should contain entries for (time_interval) ``id``, region, value,
units and timestep (year).  For example::

      - interval: 1_0
        region: "England"
        value: 23.48
        units: GW
        year: 2015
      - interval: 1_1
        region: "England"
        value: 17.48
        units: GW
        year: 2015
      - interval: 1_2
        region: "England"
        value: 16.48
        units: GW
        year: 2015


State Parameters
----------------
Some simulation models require that state is passed between years, 
for example reservoir level in the water-supply model.
These are treated as self-dependencies with a temporal offset. For example,
the sector model depends on the result of running the model for a previous
timeperiod.

Interventions
-------------

An Intervention is an investment which has a name (or name),
other attributes (such as capital cost and economic lifetime), and location,
but no build date.

An Intervention is a possible investment, normally an infrastructure asset,
the timing of which can be decided by the logic-layer.

An exhaustive list of the Interventions (normally infrastructure assets)
should be defined.
These are represented internally in the system-of-systems model,
collected into a gazateer and allow the framework to reason on
infrastructure assets across all sectors.
Interventions are instances of :class:`~smif.intervention.Intervention` and are
held in :class:`~smif.intervention.InterventionRegister`.
Interventions include investments in assets,
supply side efficiency improvements, but not demand side management (these
are incorporated in the strategies).

Define all possible interventions in an ``interventions.yaml`` file.
For example::

        - name: nuclear_power_station_england
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


Planning
--------

Existing Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~
Existing infrastructure is specified in a
``*.yaml`` file.  This uses the following format::

    - name: CCGT
      description: Existing roll out of gas-fired power stations
      timeperiod: 1990 # 2010 is the first year in the model horizon
      location: "oxford"
      new_capacity:
        value: 6
        unit: GW
      lifetime:
        value: 20
        unit: years

Pre-Specified Planning
~~~~~~~~~~~~~~~~~~~~~~

A fixed pipeline of investments can be specified using the same format as for
existing infrastructure, in the ``*.yaml`` files.

The only difference is that pre-specified planning investments occur in the
future (in comparison to the initial modelling date), whereas existing
infrastructure occur in the past. This difference is semantic at best, but a
warning is raised if future investments are included in the existing
infrastructure files in the situation where the initial model timeperiod is
altered.

Define a pipeline of interventions in a ``pre-specified.yaml`` file::

        - name: nuclear_power_station_england
          build_date: 2017

Rule Based Planning
~~~~~~~~~~~~~~~~~~~

This feature is not yet implemented

Optimisation
~~~~~~~~~~~~

This feature is not yet implemented

References
----------
.. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations