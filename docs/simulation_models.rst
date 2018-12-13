===========================
Wrapping a Simulation Model
===========================

Overview
--------

The Python class :class:`smif.model.sector_model.SectorModel` is
script which runs the wrapped model, passes in parameters and writes out results.

The wrapper acts as an interface between the simulation modelling integration framework and the
simulation model, keeping all the code necessary to implement the conversion of data types in
one place.

In particular, the wrapper must take the smif formatted data, which includes inputs,
parameters, state and pass this data into the wrapped model. After the
:py:meth:`~smif.model.sector_model.SectorModel.simulate` has run, results from the sector model
must be formatted and passed back into smif.

The handling of data is aided through the use of a set of methods provided by
:class:`smif.data_layer.data_handle.DataHandle`, namely:

- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_data`
- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameter`
- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameters`
- :py:meth:`~smif.data_layer.data_handle.DataHandle.get_results`

and

- :py:meth:`~smif.data_layer.data_handle.DataHandle.set_results`

In this section, we describe the process necessary to correctly write this wrapper function,
referring to the example project included with the package.

It is difficult to provide exhaustive details for every type of sector model implementation -
our decision to leave this largely up to the user is enabled by the flexibility afforded by
python. The wrapper can write to a database or structured text file before running a model from
a command line prompt, or import a python sector model and pass in parameters values directly.
As such, what follows is a recipe of components from which you can construct a wrapper to full
integrate your simulation model within smif.

For help or feature requests, please raise issues at the github repository [1]_ and we will
endeavour to provide assistance as resources allow.

Example Wrapper
~~~~~~~~~~~~~~~

Here's a reproduction of the example wrapper in the sample project included within smif. In
this case, the wrapper doesn't actually call or run a separate model, but demonstrates calls to
the data handler methods necessary to pass data into an external model, and send results back
to smif.

.. literalinclude:: ../src/smif/sample_project/models/energy_demand.py
   :language: python
   :lines: 8-70

The key methods in the SectorModel class which need to be overridden are:

- :py:meth:`~smif.model.sector_model.SectorModel.initialise`
- :py:meth:`~smif.model.sector_model.SectorModel.simulate`
- :py:meth:`~smif.model.sector_model.SectorModel.extract_obj`

The wrapper should be written in a python file, e.g. ``water_supply.py``. The path to the
location of this file should be entered in the sector model configuration of the project. (see
A Simulation Model File above).


Implementing a `simulate` method
--------------------------------

The most common workflow that will need to be implemented in the simulate method is:

1. Retrieve model input and parameter data from the data handler
2. Write or pass this data to the wrapped model
3. Run the model
4. Retrieve results from the model
5. Write results back to the data handler


Accessing model parameter data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the :py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameter` or
:py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameters` method as shown in the
example::

        parameter_value = data.get_parameter('smart_meter_savings')


Note that the name argument passed to the
:py:meth:`~smif.data_layer.data_handle.DataHandle.get_parameter` is that which is defined in
the sector model configuration file.


Accessing model input data for the current year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method :py:meth:`~smif.data_layer.data_handle.DataHandle.get_data()` allows a user to get
the value for any model input that has been defined in the sector model's configuration.  In
the example, the option year argument is omitted, and it defaults to fetching the data for the
current timestep::

    current_energy_demand = data.get_data('energy_demand')


Accessing model input data for the base year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access model input data from the timestep prior to the current timestep, you can use the
following argument::

    base_energy_demand = data.get_base_timestep_data('energy_demand')


Accessing model input data for a previous year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access model input data from the timestep prior to the current timestep, you can use the
following argument::

    prev_energy_demand = data.get_previous_timestep_data('energy_demand')


Passing model data directly to a Python model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the wrapped model is a python script or package, then the wrapper can import and instantiate
the model, passing in data directly.

.. literalinclude:: ../src/smif/sample_project/models/water_supply.py
   :language: python
   :lines:  76-80
   :dedent: 8

In this example, the example water supply simulation model is instantiated within the simulate
method, data is written to properties of the instantiated class and  the ``run()`` method of
the simulation model is called. Finally, (dummy) results are written back to the data handler
using the :py:meth:`~smif.data_layer.data_handle.DataHandle.set_results` method.

Alternatively, the wrapper could call the model via the command line (see below).


Passing model data in as a command line argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model is fairly simple, or requires a parameter value or input data to be passed as an
argument on the command line, use the methods provided by :py:mod:`subprocess` to call out to
the model from the wrapper::

    parameter = data.get_parameter('command_line_argument')
    arguments = ['path/to/model/executable',
                 '-my_argument={}'.format(parameter)]
    output = subprocess.run(arguments, check=True)


Writing data to a text file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, the exact implementation of writing data to a text file for subsequent reading into the
wrapped model will differ on a case-by-case basis. In the following example, we write some data
to a comma-separated-values (.csv) file::

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

The exact implementation of writing input and parameter data will differ on a case-by-case
basis. In the following example, we write model inputs ``energy_demand`` to a postgreSQL
database table ``ElecLoad`` using the psycopg2 library [2]_ ::

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


    data.set_results("cost", np.array([1.23, 1.543, 2.355])

The expected format of the data is an n-dimensional numpy array with the dimensions described
by the shape tuple ``(len(dim), ...)`` where there is an entry for each dimension defined in
the model's output specification.

A model wrapper can reflect on its outputs and their specs::

    # find the spec for a given output
    spec = self.outputs[output_name]
    # spec dimensions
    spec.dims  # e.g. ['lad', 'month', 'economic_sector']
    # spec shape (length of each dimension)
    spec.shape  # e.g. (370, 12, 46)
    # dimension names (labels for each element in a given dimension)
    spec.dim_names('lad')  # e.g. ['E070001', 'E060002', ...]
    # full metadata about dimension elements
    spec.dim_elements('lad')  # [{'name': 'E070001', 'feature': {'properties': {...}, 'coordinates': {...}}}, ...]

Results are expected to be set for each of the model outputs defined in the output
configuration and a warning is raised if these are not present at runtime.

The interval definitions associated with the output can be interrogated from within the
SectorModel class using ``self.outputs[name].get_interval_names()`` and the regions using
``self.outputs[name].get_region_names()`` and these can then be used to compose the numpy
array.


References
----------
.. [1] https://github.com/nismod/smif/issues
.. [2] http://initd.org/psycopg/
