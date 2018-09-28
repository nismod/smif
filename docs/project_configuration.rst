=====================
Project Configuration
=====================

There are three layers of configuration in order to use the simulation modelling
integration framework to conduct system-of-system modelling.

A project is the highest level container which holds all the elements required
to run models, configure simulation models and define system-of-system models.

The basic folder structure looks like this::

    project.yml
    /config
        /dimensions

        /narratives

        /scenarios

        /sector_models
            energy_demand.yml
            water_supply.yml
        /sos_models
            energy_water.yml
            energy.yml
        /sos_model_runs
            energy_central.yml
            energy_water_cp_cr.yml
    /data
        /dimensions
            hourly.csv
            annual.csv
            lad.shp
        /initial_conditions
            energy_demand_existing.yml
            energy_supply_existing.yml
        /interventions
            energy_demand.yml
            energy_supply.yml
        /narratives
            energy_demand_high_tech.csv
            central_planning.csv
        /scenarios
            population_high.csv
            population_low.csv
        /strategies
            pipeline_2020.yml
    /models
        energy_demand.py
        water_supply.py
    /results
        /energy_central
            /energy_demand
            /water_supply


The folder structure is divided into a ``config`` subfolder and a ``data``
subfolder.


The Project File
~~~~~~~~~~~~~~~~

This file holds a small amount of project-level configuration.

The project name is a unique identifier for this project.

Unit definitions references a file containing custom units, not included in
the Pint library default unit register (e.g. non-SI units).


Model Run
~~~~~~~~~

A model run brings together a system-of-systems model definition with timesteps over which
planning takes place, and a choice of scenarios and narratives to population the placeholder
scenario sets in the system-of-systems model.


.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml


.. <<This figure can be regenerated using the script in docs/gui/screenshot.sh>>
.. figure:: gui/configure.png
    :target: _images/configure.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    A model run overview


.. topic:: Hints

    [A] Create a new model run

    [B] Click on the row to edit an existing model run

    [C] Click on the bin icon to delete a configuration


.. <<This figure can be regenerated using the script in docs/gui/screenshot.sh>>
.. figure:: gui/configure-sos-model-run.png
    :target: _images/configure-sos-model-run.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    The Model Run configuration


.. csv-table::
   :header:  "#", "Attribute", "Notes"
   :widths: 3, 10, 45

   1, Name, "A unique name that identifies the Model Run configuration. Note: this field is non-editable. See also :ref:`A Model Run File`"
   2, Description, "A description that shortly describes the Model Run for future reference. See also :ref:`A Model Run File`"
   3, Created, "A timestamp that identifies at which time this Model Run was created. Note: this field is non-editable. See also :ref:`A Model Run File`"
   4, System-of-System model, "The System-of-Systems Model that this Model Run configuration is using. See also :ref:`A Model Run File`"
   5, Scenarios, "The selected Scenario for this Model Run within each of the available Scenario Sets. Note: Only the Scenario Sets that were configured in the selected System-of-System Model will be available here. See also :ref:`Scenarios`"
   6, Narrative, "The selected Narratives for this Model Run within each of the available Narrative Sets. Note: Only the Narrative Sets that were configured in the selected System-of-System Model will be available here. See also :ref:`Narratives`"
   7, Resolution, "The number of years between each of the Timesteps. See also :ref:`Timesteps`"
   8, Base year, "The Timestep where this Model Run must start the simulation. See also :ref:`Timesteps`"
   9, End year, "The last Timestep that this Model Run must simulate. See also :ref:`Timesteps`"


.. topic:: Hints

    [A] "Save" will save changes to this configuration. Click "Cancel" to leave the
    configuration without saving.


Timesteps
^^^^^^^^^

A list of timesteps define the years in which planning takes place, and the simulation models
are executed.

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   :lines: 4-7

.. csv-table:: Timestep Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   timesteps,	list,	 "A list of integer years"


Scenarios
^^^^^^^^^

For each scenario set available in the contained system-of-systems model, one scenario should
be chosen.

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   :lines: 10-12

.. csv-table:: Model Run Scenario Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   scenarios,	list,	 "A list of tuples of scenario sets and scenarios"


Narratives
^^^^^^^^^^

For each narrative set available in the project, any number of narratives can be
chosen (or none at all).

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml
   :lines: 13-15

Narrative files override the values of parameters in specific simulation models. Selecting a
narrative file which overrides parameters in an absent simulation model will have no effect.

.. csv-table:: Model Run Narrative Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   scenarios,	list,	 "A list of mappings between narrative sets and list of narrative files"


System-of-Systems Models
~~~~~~~~~~~~~~~~~~~~~~~~

A system-of-systems model collects together scenario sets and simulation models.
Users define dependencies between scenario and simulation models.

.. literalinclude:: ../src/smif/sample_project/config/sos_models/energy_water.yml
   :language: yaml

   .. <<This figure can be regenerated using the script in docs/gui/screenshot.sh>>
.. figure:: gui/configure-sos-models.png
    :target: _images/configure-sos-models.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    The System-of-System Model configuration


.. csv-table::
   :header:  "#", "Attribute", "Notes"
   :widths: 3, 10, 45

   1, Name, "A unique name that identifies the System-of-Systems model configuration. Note: this field is non-editable. See also :ref:`A System-of-Systems Model File`"
   2, Description, "A description that shortly describes the System-of-Systems model for future reference. See also :ref:`A System-of-Systems Model File`"
   3, Sector Models, "The selection of Simulation Models that are used in this System-of-Systems Model. See also :ref:`A System-of-Systems Model File`"
   4, Scenario Sets, "The selection of Scenario Sets that are used in this System-of-Systems Model. See also :ref:`A System-of-Systems Model File`"
   5, Narrative Sets, "The selection of Narrative Sets that are used in this System-of-Systems Model. See also :ref:`A System-of-Systems Model File`"
   6, Dependencies, "The list of Dependencies that are defined between sources and links. See also :ref:`Dependencies`"


.. topic:: Hints

    [A] "Add Dependency" opens a form to add a new dependency

    [B] "Save" will save changes to this configuration. Click "Cancel" to leave the
    configuration without saving.


Scenarios
^^^^^^^^^

Scenario sets are the categories in which scenario data are organised. Choosing a scenario set
at this points allows different scenario data to be chosen in model runs which share the same
system-of-systems model configuration defintion.

.. literalinclude:: ../src/smif/sample_project/config/sos_models/energy_water.yml
   :language: yaml
   :lines: 3-5

.. csv-table:: Scenario Sets Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   names,	list,	 "A list of scenario set names"


Simulation Models
^^^^^^^^^^^^^^^^^

This section contains a list of pre-configured simulation models which exist in the current
project.

.. literalinclude:: ../src/smif/sample_project/config/sos_models/energy_water.yml
   :language: yaml
   :lines: 6-8

.. csv-table:: Simulation Models Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   names,	list,	 "A list of simulation model names"


Dependencies
^^^^^^^^^^^^

In this section, dependencies are defined between sources and sinks.

.. literalinclude:: ../src/smif/sample_project/config/sos_models/energy_water.yml
   :language: yaml
   :lines: 9-17

.. csv-table:: Dependency Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   source_model,	string,	 "The source model of the data"
   source_model_output,	string,	 "The output in the source model"
   sink_model,	string,	 "The model which depends on the source"
   sink_model_input,	string,	 "The input which should receive the data"



Simulation Models
~~~~~~~~~~~~~~~~~

A model file contains all the configuration data necessary for smif to run the model, and link
the model to data sources and sinks. This file also contains a list of parameters, the 'knobs
and dials' the user wishes to expose to smif which can be adjusted in narratives. Intervention
files and initial condition files contain the collections of data that are needed to expose the
model to smif's decision making functionality.



.. <<This figure can be regenerated using the script in docs/gui/screenshot.sh>>
.. figure:: gui/configure-sector-models.png
    :target: _images/configure-sector-models.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    The Model Wrapper configuration


.. csv-table::
   :header:  "#", "Attribute", "Notes"
   :widths: 3, 10, 45

   1, Name, "A unique name that identifies the simulation model that is wrapped. Note: this field is non-editable. See also :ref:`A Simulation Model File`"
   2, Description, "A description that shortly describes the simulation model for future reference. See also :ref:`A Simulation Model File`"
   3, Class Name, "Name of the Class that is used in the smif wrapper. See also :ref:`Wrapping a Sector Model: Overview`"
   4, Path, "The location of the python wrapper file. See also :ref:`Wrapping a Sector Model: Overview`"
   5, Inputs, "The simulation model inputs with their name, units and temporal-spatial resolution. See also :ref:`Inputs`"
   6, Outputs, "The simulation model outputs with their name, units and temporal-spatial resolution. See also :ref:`Outputs`"
   7, Parameters, "The simulation model parameters. See also :ref:`Parameters`"


.. topic:: Hints

    [A] "Add Input" to open a form to add a new input

    [B] "Add Output" to open a form to add a new output

    [C] "Add Parameter" to open a form to add a new parameter

    [D] "Save" to save changes to this configuration. Click on "Cancel" to leave the
    configuration without saving.




Inputs
^^^^^^

Define the collection of inputs required from external sources to run the model. Inputs are
defined with a name, spatial resolution, temporal-resolution and units.

.. literalinclude:: ../src/smif/sample_project/config/sector_models/water_supply.yml
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

Define the collection of output model parameters used for the purpose of metrics, for
accounting purposes, such as operational cost and emissions, or as the source of a dependency
in another model.

.. literalinclude:: ../src/smif/sample_project/config/sector_models/water_supply.yml
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

.. literalinclude:: ../src/smif/sample_project/config/sector_models/water_supply.yml
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



Scenarios
~~~~~~~~~

One section lists the scenario sets. These give the categories into which scenarios are
collected.

The ``scenarios`` section lists the scenarios and corresponding collections of data associated
with scenarios.


Narratives
~~~~~~~~~~

Narrative sets collect together the categories into which narrative files are collected.

The ``narratives`` section lists the narratives and mapping to one or more narrative files


Dimensions
~~~~~~~~~~

Region definitions list the collection of region files and the mapping to a unique name which
can be used in scenarios and sector models. Region definitions define the spatial resolution of
data.

Interval definitions list the collection of interval files and the mapping to a unique name
which can be used in scenarios and sector models. Interval definitions define the temporal
resolution of data.



Data Folder
-----------

This folder holds data like information to define the spatial and temporal resolution of data,
as well as exogenous environmental data held in scenarios.


Initial Conditions
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../src/smif/sample_project/data/initial_conditions/reservoirs.yml
   :language: yaml


Dimensions
~~~~~~~~~~


Temporal dimensions
^^^^^^^^^^^^^^^^^^^

The attribution of hours in a year to the temporal resolution used in the sectoral model.

Within-year time intervals are specified in yaml files, and as for regions, specified in the
``*.yml`` file in the ``project/data/intervals`` folder.

This links a unique name with the definitions of the intervals in a yaml file. The data in the
file specify the mapping of model timesteps to durations within a year (assume modelling 365
days: no extra day in leap years, no leap seconds)

Use ISO 8601 [1]_ duration format to specify periods::

    P[n]Y[n]M[n]DT[n]H[n]M[n]S

For example:

.. literalinclude:: ../src/smif/sample_project/data/interval_definitions/annual_intervals.csv
   :language: text

In this example, the interval with id ``1`` begins in the first hour of the year and ends in
the last hour of the year. This represents one, year-long interval.

.. csv-table:: Interval Attributes
   :header: "Attribute", "Type", "Notes"
   :widths: 15, 10, 30

   id, string, "The unique identifier used by the simulation model"
   start_hour, string, "Period since beginning of year"
   end_hour, string, "Period since beginning of year"


Spatial dimensions
^^^^^^^^^^^^^^^^^^

Define the set of unique regions which are used within the model as polygons. The spatial
resolution of the model may be implicit, and even a national model needs to have a national
region defined. Inputs and outputs are assigned a model-specific geography from this list
allowing automatic conversion from and to these geographies.

Model region files are stored in ``project/data/region_defintions``.

The file format must be possible to parse with GDAL, and must contain an attribute "name" to
use as an identifier for the region.

The sets of geographic regions are specified in the project configuration file using a
``region_definitions`` attributes as shown below:

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 10,12-17

This links a name, used elsewhere in the configuration with inputs, outputs and scenarios with
a file containing the geographic data.


Interventions
~~~~~~~~~~~~~

Interventions are the atomic units which comprise the infrastructure systems in the simulation
models. Interventions can represent physical assets such as pipes, and lines (edges in a
network) or power stations and reservoirs (nodes in a network). Interventions can also
represent intangibles which affects the operation of a system, such as a policy.


An exhaustive list of the interventions (often infrastructure assets) should be defined. These
are represented internally in the system-of-systems model, collected into a gazateer and allow
the framework to reason on infrastructure assets across all sectors.

Interventions are instances of :class:`~smif.intervention.Intervention` and are held in
:class:`~smif.intervention.InterventionRegister`. Interventions include investments in assets,
supply side efficiency improvements, but not demand side management (these are incorporated in
the strategies).

Define all possible interventions in an ``*.yml`` file in the ``project/data/interventions``
For example:

.. literalinclude:: ../src/smif/sample_project/data/interventions/water_supply.yml
   :language: yaml
   :lines: 6-19

Alternatively define all possible interventions in an ``*.csv`` file in the
``project/data/interventions`` For example:

.. literalinclude:: ../src/smif/sample_project/data/interventions/energy_supply.csv
   :language: csv
   :lines: 1-5

Note that the ``_value`` and ``_unit`` suffixes of the column names are used to unpack the data
internally.

Some attributes are required:

- technical_lifetime
  (years are assumed as unit and can be omitted)


Narratives
~~~~~~~~~~

A narrative file contains references to 0 or more parameters defined in the simulation models,
as well as special ``global`` parameters. Whereas model parameters are available only to
individual simulation models, global parameters are available across all models. Use global
paramaters for system-wide constants, such as emission coefficients, exchange rates, conversion
factors etc.

Value specified in the narrative file override the default values specified in the simulation
model configuration. If more than one narrative file is selected in the sos model
configuration, then values in later files override values in earlier files.

.. literalinclude:: ../src/smif/sample_project/data/narratives/high_tech_dsm.yml
   :language: yaml


Scenarios
~~~~~~~~~

The ``scenarios`` config folder  allows you to define static sources for simulation model
dependencies.

The data files are stored in the ``project/data/scenarios`` folder.

The metadata required to define a particular scenario are shown in the table below. It is
possible to associate a number of different data sets with the same scenario, so that, for
example, choosing the `High Population` scenario allows users to access both population count
and density data in the same or different spatial and temporal resolutions.

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
points to a comma-seperated-values file stored in the ``project/data/scenarios``
folder. For example:

.. literalinclude:: ../src/smif/sample_project/data/scenarios/population_high.csv
   :language: text

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
