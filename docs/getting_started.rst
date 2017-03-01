.. _getting_started

Getting Started 
===============

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
        /data/<sector_model>/assets.yaml

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

- :py:meth:`smif.sector_model.SectorModel.simulate`
- :py:meth:`smif.sector_model.SectorModel.extract_obj`

The path to the location of the ``run.py`` file should be entered in the
``model.yaml`` file under the ``path`` key 
(see System-of-Systems Model File above).

To integrate an infrastructure simulation model within the system-of-systems
modelling framework, it is also necessary to provide the following configuration
data.

Geographies
-----------
Define the set of unique regions which are used within the model as polygons.
Inputs and outputs are assigned a model-specific geography from this list
allowing automatic conversion from and to these geographies.

Model regions are specified in ``regions.*``.

The file format must be possible to parse with GDAL, and must contain
an attribute "name" to use as an identifier for the region.

Temporal Resolution
-------------------
The attribution of hours in a year to the temporal resolution used
in the sectoral model.

Within-year time intervals are specified
in ``time_intervals.yaml``

These specify the mapping of model timesteps to durations within a year
(assume modelling 365 days: no extra day in leap years, no leap seconds)

Each time interval must have

- start (period since beginning of year)
- end (period since beginning of year)
- id (label to use when passing between integration layer and sector model)

use ISO 8601 [1]_ duration format to specify periods::

    P[n]Y[n]M[n]DT[n]H[n]M[n]S

References
----------
.. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations

Inputs
------
Define the collection of inputs required from external sources
to run the model.  For example
"electricity demand (kWh, <region>, <hour>)".
Inputs are defined with a spatial and temporal-resolution, a unit
and a ``from_model``.

Only those inputs required as dependencies are defined here, although
dependencies are activated when configured in the system-of-systems model.

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

The entry for the ``from_model`` attribute can be ``scenario``. This allows
definition of statically defined data for each model year to be specified in
a ``<name>.yaml`` file, in conjunction with a scenario-specific time-intervals
file.

Outputs
-------
Define the collection of outputs used as metrics, 
for the purpose of optimisation or
rule-based planning approaches (so normally a cost-function), and those
outputs required for accounting purposes, such as operational cost and
emissions, or as a dependency in another model.

The ``outputs.yaml`` file defines the output metrics from the model.
For example::

        metrics:
          - name: total_cost
          - name: water_demand
          - name: total_emissions

State Parameters
----------------
Some simulation models require that state is passed between years, for example
reservoir level in the water-supply model.
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
Interventions are instances of :class:`~smif.asset.Intervention` and are
held in :class:`~smif.asset.InterventionRegister`.
Interventions include investments in assets,
supply side efficiency improvements, but not demand side management (these
are incorporated in the strategies).

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


Planning
--------

Existing Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~
Existing infrastructure is specified in a
``*.yaml`` file.  This uses the following format::
   -
    name: CCGT
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