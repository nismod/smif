.. _getting_started:

Getting Started
===============

Once you have installed **smif**, the quickest way to get started is to use the
included test project. You can find the test project in the development version
of the package, in the ``tests/fixtures/single_run`` folder.

On the command line, type the following command to list the available model 
runs::

  $ smif list -d smif/tests/fixtures/single_run
  20170918_energy_water_short.yml
  20170918_energy_water.yml

To run a model run, type the following command::

  $ smif run 20170918_energy_water.yml -d smif/tests/fixtures/single_run
  Model run complete

Note that the ``-d`` directory flag should point to the single_run folder, so
check you are pointing to the correct directory if this doesn't work first time.

Project Configuration
---------------------

There are three layers of configuration in order to use the simulation modelling
integration framework to conduct system-of-system modelling.

A project is the highest level container which holds all the elements required 
to run models, configure simulation models and define system-of-system models.

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

A simulation model file contains all the configuration data necessary for smif
to run the model, and link the model to data sources and sinks. 
This file also contains a list of parameters, the 'knobs and dials' 
the user wishes to expose to smif which can be adjusted in narratives.
Intervention files and initial condition files contain the collections of data
that are needed to expose the model to smif's decision making functionality.

.. literalinclude:: ../tests/fixtures/single_run/config/sector_models/water_supply.yml
   :language: yaml
   
Inputs
^^^^^^

Define the collection of inputs required from external sources
to run the model.  
Inputs are defined with a name, spatial resolution, 
temporal-resolution and units.

.. literalinclude:: ../tests/fixtures/single_run/config/sector_models/water_supply.yml
   :language: yaml
   :lines: 6-14

.. csv-table:: Input Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   name, string, "A unique name within the input defintions"
   spatial_resolution, string, "References an entry in the region definitions"
   temporal_resolution, string, "References an entry in the interval definitions"
   units, string, "References an entry in the unit definitions"

Outputs
^^^^^^^

Define the collection of output model parameters used for the purpose 
of metrics, for accounting purposes, such as operational cost and
emissions, or as the source of a dependency in another model.

.. literalinclude:: ../tests/fixtures/single_run/config/sector_models/water_supply.yml
   :language: yaml
   :lines: 19-27

.. csv-table:: Output Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   name, string, "A unique name within the output definitions"
   spatial_resolution, string, "References an entry in the region definitions"
   temporal_resolution, string, "References an entry in the interval definitions"
   units, string, "References an entry in the unit definitions"

Parameters
^^^^^^^^^^

.. literalinclude:: ../tests/fixtures/single_run/config/sector_models/water_supply.yml
   :language: yaml
   :lines: 37-43


.. csv-table:: Parameter Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   name,	string,	 "A unique name within the simulation model"
   description,	string,	"Include sources of assumptions around default value"
   absolute_range,tuple, "Raises an error if bounds exceeded"	 
   suggested_range,	tuple, "Provides a hint to a user as to sensible ranges"	 
   default_value,	float, "The default value for the parameter"	 
   units,	string,	""

A System-of-System Model File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A system-of-systems model collects together scenario sets and simulation models.
Users define dependencies between scenario and simulation models.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_models/energy_water.yml
   :language: yaml
   
Scenario Sets
^^^^^^^^^^^^^

Scenario sets are the categories in which scenario data are organised.
Choosing a scenario set at this points allows different scenario data 
to be chosen in model runs which share the same system-of-systems model
configuration defintion.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_models/energy_water.yml
   :language: yaml
   :lines: 3-5

.. csv-table:: Scenario Sets Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   names,	list,	 "A list of scenario set names"

Simulation Models
^^^^^^^^^^^^^^^^^

This section contains a list of pre-configured simulation models which exist in
the current project.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_models/energy_water.yml
   :language: yaml
   :lines: 6-8

.. csv-table:: Simulation Models Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   names,	list,	 "A list of simulation model names"

Dependencies
^^^^^^^^^^^^

In this section, dependencies are defined between sources and sinks.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_models/energy_water.yml
   :language: yaml
   :lines: 9-17

.. csv-table:: Dependency Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   source_model,	string,	 "The source model of the data"
   source_model_output,	string,	 "The output in the source model"
   sink_model,	string,	 "The model which depends on the source"
   sink_model_input,	string,	 "The input which should receive the data"

A Model Run File
~~~~~~~~~~~~~~~~

A model run brings together a system-of-systems model definition with timesteps
over which planning takes place, and a choice of scenarios and narratives to
population the placeholder scenario sets in the system-of-systems model.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   
Timesteps
^^^^^^^^^

A list of timesteps define the years in which planning takes place, 
and the simulation models are executed.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   :lines: 4-7

.. csv-table:: Timestep Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   timesteps,	list,	 "A list of integer years"

Scenarios
^^^^^^^^^

For each scenario set available in the contained system-of-systems model,
one scenario should be chosen.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   :lines: 10-12

.. csv-table:: Model Run Scenario Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   scenarios,	list,	 "A list of tuples of scenario sets and scenarios"

Narratives
^^^^^^^^^^

For each narrative set available in the project, zero or more available 
narratives should be chosen.

.. literalinclude:: ../tests/fixtures/single_run/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   :lines: 13-15

Note that narrative files override the values of parameters in specific
simulation models. 
Selecting a narrative file which overrides parameters in an absent simulation
model will have no effect.

.. csv-table:: Model Run Narrative Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   scenarios,	list,	 "A list of mappings between narrative sets and list of narrative files"

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

The attribution of hours in a year to the temporal resolution used
in the sectoral model.

Within-year time intervals are specified in yaml files, and as for regions,
specified in the ``*.yml`` file in the ``project/data/intervals`` folder.

This links a unique name with the definitions of the intervals in a yaml file.
The data in the file specify the mapping of model timesteps to durations within a year
(assume modelling 365 days: no extra day in leap years, no leap seconds)

Use ISO 8601 [1]_ duration format to specify periods::

    P[n]Y[n]M[n]DT[n]H[n]M[n]S

For example:

.. literalinclude:: ../tests/fixtures/single_run/data/interval_definitions/annual_intervals.csv
   :language: csv

In this example, the interval with id ``1`` begins in the first hour of the year
and ends in the last hour of the year. This represents one, year-long interval.

.. csv-table:: Interval Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   id, string, "The unique identifier used by the simulation model"
   start_hour, string, "Period since beginning of year"
   end_hour, string, "Period since beginning of year"

Region definitions
~~~~~~~~~~~~~~~~~~

Define the set of unique regions which are used within the model as polygons.
The spatial resolution of the model may be implicit, and even a national model
needs to have a national region defined.
Inputs and outputs are assigned a model-specific geography from this list
allowing automatic conversion from and to these geographies.

Model region files are stored in ``project/data/region_defintions``.

The file format must be possible to parse with GDAL, and must contain
an attribute "name" to use as an identifier for the region.

The sets of geographic regions are specified in the project configuration file 
using a ``region_definitions`` attributes as shown below:

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 10,12-17

This links a name, used elsewhere in the configuration with inputs, outputs and scenarios
with a file containing the geographic data.

Interventions
~~~~~~~~~~~~~

An Intervention is an investment which has a name (or name),
other attributes (such as capital cost and economic lifetime), and location,
but no build date.

An intervention is a possible investment, normally an infrastructure asset,
the timing of which can be decided by the logic-layer.

An exhaustive list of the interventions (often infrastructure assets)
should be defined.
These are represented internally in the system-of-systems model,
collected into a gazateer and allow the framework to reason on
infrastructure assets across all sectors.
Interventions are instances of :class:`~smif.intervention.Intervention` and are
held in :class:`~smif.intervention.InterventionRegister`.
Interventions include investments in assets,
supply side efficiency improvements, but not demand side management (these
are incorporated in the strategies).

Define all possible interventions in an ``*.yml`` file 
in the ``project/data/interventions`` For example:

.. literalinclude:: ../tests/fixtures/single_run/data/interventions/water_supply.yml
   :language: yaml
   :lines: 6-19

Narratives
~~~~~~~~~~

.. literalinclude:: ../tests/fixtures/single_run/data/narratives/high_tech_dsm.yml
   :language: yaml

Scenarios
~~~~~~~~~

The ``scenarios:`` section of the project configuration file allows
you to define static sources for simulation model dependencies.

In the case of the example project file shown earlier, 
the ``scenarios`` section lists the scenarios and corresponding collections of
data associated with the scenario sets:

.. literalinclude:: ../tests/fixtures/single_run/config/project.yml
   :language: yaml
   :lines: 24,26-43

The data files are stored in the ``project/data/scenarios` folder.

The metadata required to define a particular scenario are shown in the table
below.
It is possible to associate a number of different data sets with
the same scenario, so that, for example, choosing the `High Population`
scenario allows users to access both population count and density data
in the same or different spatial and temporal resolutions.

.. csv-table:: Scenario Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   name, string , 
   description, string , 
   scenario_set, string ,  
   parameters, list , 

Scenario Parameters
^^^^^^^^^^^^^^^^^^^

The filename in the ``parameters`` section within the scenario definition
points to a comma-seperated-values file stored in the ``project/data/scenarios`
folder. For example:

.. literalinclude:: ../tests/fixtures/single_run/data/scenarios/population.csv
   :language: csv

For each entry in the scenario parameters list, the following metadata
is required:

.. csv-table:: Scenario Parameter Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   name, string, 
   spatial_resolution, string,
   temporal_resolution, string,
   units, string,
   filename, string,

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


References
----------
.. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations