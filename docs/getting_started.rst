.. _getting_started:

Getting Started
===============

Once you have installed **smif**, the quickest way to get started is to use the
included sample project. You can make a new directory and copy the sample
project files there by running::

  $ mkdir sample_project
  $ cd sample_project
  $ smif setup
  $ ls
  config/ data/ models/ planning/ results/ smif.log

On the command line, from within the project directory, type the following
command to list the available model runs::

  $ smif list
  20170918_energy_water_short
  20170918_energy_water

To run a model run, type the following command::

  $ smif run 20170918_energy_water
  Model run complete

Note that the ``-d`` directory flag can be used to point to the project folder,
so you can run smif commands explicitly::

  $ smif list -d ~/projects/smif_sample_project/
  ...

Project Configuration
---------------------

There are three layers of configuration in order to use the simulation modelling
integration framework to conduct system-of-system modelling.

A project is the highest level container which holds all the elements required
to run models, configure simulation models and define system-of-system models.

The basic folder structure looks like this::

    /config
        project.yml
        /sector_models
            energy_demand.yml
            water_supply.yml
        /sos_models
            energy_water.yml
        /sos_model_runs
            20170918_energy_water.yml
            20170918_energy_water_short.yml
    /data
        /initial_conditions
            energy_demand_existing.yml
            energy_supply_existing.yml
        /interval_definitions
            hourly.csv
            annual.csv
        /interventions
            energy_demand.yml
        /narratives
            energy_demand_high_tech.yml
            central_planning.yml
        /region_definitions
            lad.shp
        /scenarios
            population_high.csv
    /models
        energy_demand.py
        water_supply.py
    /results
        /20170918_energy_water_short
            /energy_demand
            /water_supply

The folder structure is divided into a ``config`` subfolder and a ``data``
subfolder.

The Configuration Folder
------------------------

This folder holds configuration and metadata on simulation models,
system-of-system models and model runs.

The Project File
~~~~~~~~~~~~~~~~

This file holds all the project configuration.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :caption: project.yml
   :language: yaml

We'll step through this configuration file section by section.

The first line gives the project name, a unique identifier for this project.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 1

One section lists the scenario sets. These give the categories into which
scenarios are collected.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 2-7

Narrative sets collect together the categories into which narrative files are
collected.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 13-15

Region definitions list the collection of region files and the mapping to a
unique name which can be used in scenarios and sector models.
Region definitions define the spatial resolution of data.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 25-28

Interval definitions list the collection of interval files and the mapping to a
unique name which can be used in scenarios and sector models.
Interval definitions define the temporal resolution of data.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 16-19

Unit definitions references a file containing custom units, not included in
the Pint library default unit register (e.g. non-SI units).

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 69

The ``scenarios`` section lists the scenarios and corresponding collections of
data associated with scenarios.

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 32-41

The ``narratives`` section lists the narratives and mapping to one or more
narrative files

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 20-24

A Simulation Model File
~~~~~~~~~~~~~~~~~~~~~~~

A simulation model file contains all the configuration data necessary for smif
to run the model, and link the model to data sources and sinks.
This file also contains a list of parameters, the 'knobs and dials'
the user wishes to expose to smif which can be adjusted in narratives.
Intervention files and initial condition files contain the collections of data
that are needed to expose the model to smif's decision making functionality.

.. literalinclude:: ../src/smif/sample_project/config/sector_models/water_supply.yml
   :language: yaml

Inputs
^^^^^^

Define the collection of inputs required from external sources
to run the model.
Inputs are defined with a name, spatial resolution,
temporal-resolution and units.

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

Define the collection of output model parameters used for the purpose
of metrics, for accounting purposes, such as operational cost and
emissions, or as the source of a dependency in another model.

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

A System-of-System Model File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A system-of-systems model collects together scenario sets and simulation models.
Users define dependencies between scenario and simulation models.

.. literalinclude:: ../src/smif/sample_project/config/sos_models/energy_water.yml
   :language: yaml

Scenario Sets
^^^^^^^^^^^^^

Scenario sets are the categories in which scenario data are organised.
Choosing a scenario set at this points allows different scenario data
to be chosen in model runs which share the same system-of-systems model
configuration defintion.

.. literalinclude:: ../src/smif/sample_project/config/sos_models/energy_water.yml
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

A Model Run File
~~~~~~~~~~~~~~~~

A model run brings together a system-of-systems model definition with timesteps
over which planning takes place, and a choice of scenarios and narratives to
population the placeholder scenario sets in the system-of-systems model.

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
   :language: yaml

Timesteps
^^^^^^^^^

A list of timesteps define the years in which planning takes place,
and the simulation models are executed.

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
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

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
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

.. literalinclude:: ../src/smif/sample_project/config/sos_model_runs/20170918_energy_water.yml
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

.. literalinclude:: ../src/smif/sample_project/data/initial_conditions/reservoirs.yml
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

.. literalinclude:: ../src/smif/sample_project/data/interval_definitions/annual_intervals.csv
   :language: text

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

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 10,12-17

This links a name, used elsewhere in the configuration with inputs, outputs and scenarios
with a file containing the geographic data.

Interventions
~~~~~~~~~~~~~

Interventions are the atomic units which comprise the infrastructure systems 
in the simulation models.
Interventions can represent physical assets such as pipes, 
and lines (edges in a network) or power stations and reservoirs 
(nodes in a network).
Interventions can also represent intangibles which affects the operation 
of a system, such as a policy.


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

.. literalinclude:: ../src/smif/sample_project/data/interventions/water_supply.yml
   :language: yaml
   :lines: 6-19

Narratives
~~~~~~~~~~

A narrative file contains references to 0 or more parameters defined in the
simulation models, as well as special ``global`` parameters. Whereas model 
parameters are available only to individual simulation models, 
global parameters are available across all models. 
Use global paramaters for system-wide constants, such as emission coefficients,
exchange rates, conversion factors etc.

Value specified in the narrative file override the default values specified
in the simulation model configuration. If more than one narrative file is
selected in the sos model configuration, then values in later files override
values in earlier files.

.. literalinclude:: ../src/smif/sample_project/data/narratives/high_tech_dsm.yml
   :language: yaml

Scenarios
~~~~~~~~~

The ``scenarios:`` section of the project configuration file allows
you to define static sources for simulation model dependencies.

In the case of the example project file shown earlier,
the ``scenarios`` section lists the scenarios and corresponding collections of
data associated with the scenario sets:

.. literalinclude:: ../src/smif/sample_project/config/project.yml
   :language: yaml
   :lines: 24,26-43

The data files are stored in the ``project/data/scenarios`` folder.

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

Wrapping a Sector Model: Overview
---------------------------------

In addition to collecting the configuration data listed above, 
to integrate a new sector model into the system-of-systems model 
it is necessary to write a Python wrapper function.
The template class :class:`smif.model.sector_model.SectorModel` enables a user
to write a script which runs the wrapped model, passes in parameters and writes 
out results.

The wrapper acts as an interface between the simulation modelling
integration framework and the simulation model, keeping all the code necessary
to implement the conversion of data types in one place.

In particular, the wrapper must take the smif formatted data, which includes
inputs, parameters, state and pass this data into the wrapped model. After the
:py:meth:`~smif.model.sector_model.SectorModel.simulate` has run, results from
the sector model must be formatted and passed back into smif.

The handling of data is aided through the use of a set of methods provided by 
:class:`smif.data_layer.data_handle.DataHandle`, namely:

- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_data`
- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameter`
- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameters`
- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_results`

and 

- :py:meth:`~smif.data_layer.data_handle.DataHandle.set_results`

In this section, we describe the process necessary to correctly write this
wrapper function, referring to the example project included with the package.

It is difficult to provide exhaustive details for every type of sector model
implementation - our decision to leave this largely up to the user 
is enabled by the flexibility afforded by python. The wrapper can write to a
database or structured text file before running a model from a command line 
prompt, or import a python sector model and pass in parameters values directly.
As such, what follows is a recipe of components from which you can construct
a wrapper to full integrate your simulation model within smif.

For help or feature requests, please raise issues at the github repository [2]_ 
and we will endeavour to provide assistance as resources allow.

Example Wrapper
~~~~~~~~~~~~~~~

Here's a reproduction of the example wrapper in the sample project included
within smif. In this case, the wrapper doesn't actually call or run a separate
model, but demonstrates calls to the data handler methods necessary to pass
data into an external model, and send results back to smif.

.. literalinclude:: ../src/smif/sample_project/models/energy_demand.py
   :language: python
   :lines: 8-70

The key methods in the SectorModel class which need to be overridden are:

- :py:meth:`~smif.model.sector_model.SectorModel.initialise`
- :py:meth:`~smif.model.sector_model.SectorModel.simulate`
- :py:meth:`~smif.model.sector_model.SectorModel.extract_obj`

The wrapper should be written in a python file, e.g. ``water_supply.py``.
The path to the location of this file should be entered in the
sector model configuration of the project.
(see A Simulation Model File above).

Wrapping a Sector Model: Simulate
---------------------------------

The most common workflow that will need to be implemented in the simulate 
method is:

1. Retrieve model input and parameter data from the data handler
2. Write or pass this data to the wrapped model
3. Run the model
4. Retrieve results from the model
5. Write results back to the data handler

Accessing model parameter data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameter` or 
:py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameters` method as shown in the
example:

.. literalinclude:: ../src/smif/sample_project/models/energy_demand.py
   :language: python
   :lines:  22
   :dedent: 8


Note that the name argument passed to the :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameter` is that which is defined in
the sector model configuration file.

Accessing model input data for the current year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method :py:meth:`~smif.data_layer.data_handle.DataHandle.get_data()` allows a user to
get the value for any model input that has been defined in the sector model's
configuration.  In the example, the option year argument is omitted, and it
defaults to fetching the data for the current timestep.

.. literalinclude:: ../src/smif/sample_project/models/energy_demand.py
   :language: python
   :lines: 27
   :dedent: 8


Accessing model input data for the base year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access model input data from the timestep prior to the current timestep, 
you can use the following argument:

.. literalinclude:: ../src/smif/sample_project/models/energy_demand.py
   :language: python
   :lines:  33
   :dedent: 8


Accessing model input data for a previous year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access model input data from the timestep prior to the current timestep, 
you can use the following argument:

.. literalinclude:: ../src/smif/sample_project/models/energy_demand.py
   :language: python
   :lines:  41
   :dedent: 8

Passing model data directly to a Python model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the wrapped model is a python script or package, then the wrapper can
import and instantiate the model, passing in data directly.

.. literalinclude:: ../src/smif/sample_project/models/water_supply.py
   :language: python
   :lines:  73-80
   :dedent: 8

In this example, the example water supply simulation model is instantiated
within the simulate method, data is written to properties of the instantiated 
class and  the ``run()`` method of the simulation model is called. 
Finally, (dummy) results are written back to the data handler using the 
:py:meth:`~smif.data_layer.data_handle.DataHandle.set_results` method.

Alternatively, the wrapper could call the model via the command line 
(see below).

Passing model data in as a command line argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model is fairly simple, or requires a parameter value or input data to
be passed as an argument on the command line, 
use the methods provided by :py:mod:`subprocess` to call out to the model 
from the wrapper::

    parameter = data.get_parameter('command_line_argument')
    arguments = ['path/to/model/executable',
                 '-my_argument={}'.format(parameter)]
    output = subprocess.run(arguments, check=True)

Writing data to a text file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, the exact implementation of writing data to a text file for subsequent
reading into the wrapped model will differ on a case-by-case basis.
In the following example, we write some data to a comma-separated-values (.csv)
file::

    with open(path_to_data_file, 'w') as open_file:
        fieldnames = ['year', 'PETROL', 'DIESEL', 'LPG', 
                      'ELECTRICITY', 'HYDROGEN', 'HYBRID']
        writer = csv.DictWriter(open_file, fieldnames)
        writer.writeheader()

        now = data.current_timestep
        base_year_enum = RelativeTimestep.BASE

        base_price_set = {
            'year': base_year_enum.resolve_relative_to(now, data.timesteps),
            'PETROL': data.get_data('petrol_price', base_year_enum),
            'DIESEL': data.get_data('diesel_price', base_year_enum),
            'LPG': data.get_data('lpg_price', base_year_enum),
            'ELECTRICITY': data.get_data('electricity_price', base_year_enum),
            'HYDROGEN': data.get_data('hydrogen_price', base_year_enum),
            'HYBRID': data.get_data('hybrid_price', base_year_enum)
        }

        current_price_set = {
            'year': now,
            'PETROL': data.get_data('petrol_price'),
            'DIESEL': data.get_data('diesel_price'),
            'LPG': data.get_data('lpg_price'),
            'ELECTRICITY': data.get_data('electricity_price'),
            'HYDROGEN': data.get_data('hydrogen_price'),
            'HYBRID': data.get_data('hybrid_price')
        }

        writer.writerow(base_price_set)
        writer.writerow(current_price_set)

Writing data to a database
~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact implementation of writing input and parameter data will differ on a
case-by-case basis. In the following example, we write model inputs 
``energy_demand`` to a postgreSQL database table ``ElecLoad`` using the 
psycopg2 library [3]_ ::

    def simulate(self, data):

        # Open a connection to the database
        conn = psycopg2.connect("dbname=vagrant user=vagrant")
        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Returns a numpy array whose dimensions are defined by the interval and
        # region definitions
        elec_data = data.get_data('electricity_demand')

        # Build the SQL string
        sql = """INSERT INTO "ElecLoad" (Year, Interval, BusID, ElecLoad) 
                 VALUES (%s, %s, %s, %s)"""

        # Get the time interval definitions associated with the input
        time_intervals = self.inputs[name].get_interval_names()
        # Get the region definitions associated with the input
        regions = self.inputs[name].get_region_names()
        # Iterate over the regions and intervals (columns and rows) of the numpy
        # array holding the energy demand data and write each value into the table
        for i, region in enumerate(regions):
            for j, interval in enumerate(time_intervals):
                # This line calls out to a helper method which associates 
                # electricity grid bus bars to energy demand regions
                bus_number = get_bus_number(region)
                # Build the tuple to write to the table
                insert_data = (data.current_timestep,
                               interval,
                               bus_number,
                               data[i, j])
                cur.execute(sql, insert_data)

        # Make the changes to the database persistent
        conn.commit()

        # Close communication with the database
        cur.close()
        conn.close()

Writing model results to the data handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writing results back to the data handler is as simple as calling the 
:py:meth:`~smif.data_layer.data_handle.DataHandle.set_results` method::

    data.set_results("cost", np.array([[1.23, 1.543, 2.355]])

The expected format of the data is a 2-dimensional numpy array with the 
dimensions described by the tuple ``(len(regions), len(intervals))`` 
as defined in the model's output configuration.
Results are expected to be set for each of the model outputs defined in the 
output configuration and a warning is raised if these are not present at 
runtime.

The interval definitions associated with the output can be interrogated from
within the SectorModel class using ``self.outputs[name].get_interval_names()``
and the regions using ``self.outputs[name].get_region_names()`` and these can
then be used to compose the numpy array.

References
----------
.. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations
.. [2] https://github.com/nismod/smif/issues
.. [3] http://initd.org/psycopg/
