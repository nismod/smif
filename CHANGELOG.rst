=========
Changelog
=========

Version 1.0.2
=============

Minor updates:

- Update smif app readme
- Bump babel and webpack major versions, update other npm packages
- Validate self-dependencies (between timesteps is okay)
- Add validation methods for narratives
- Make illegal parameters visible in narrative configuration
- Pass path to binary filestore, extract method to parent class
- Allow adaptors to be directly included in a system of systems model
- Provide useful error message when there are missing data in a data array
- Add profiling to some places in the program, provide summary at end of modelrun
- Don't read dimension elements through API

Fixes:

- Fix and test reading from timeseries, including zero-d case
- Fix API calling old store methods
- Pin libgcc as possible cause of shared library import errors
- Fix up test_validate to use conftest configs
- Fix react-icons imports, drop reactstrap
- Silence mocha deprecation warning.
- Ensure smif npm package is private
- Fix update_model method store
- Adopt fix for DataFrame.to_dict('records') from future pandas


Version 1.0
===========

Functionality:

- GUI improved usability

  - Forms now ask users to discard or save changes
  - Configuration lists can be sorted and filtered
  - Single click navigation between linked configurations
  - First steps of input validation (in SosModel configurations)

- Define model data (inputs/parameters/outputs) using arbitrary dimensions (may be spatial,
  temporal, categorical)

  - Dimension conversions can be performed by an ``Adaptor``, represented as another
    ``SectorModel`` within a ``SosModel``

- Data layer refactor to enable various Store implementations, separately for configuration
  data, metadata and input/parameter/results and interventions/decisions/state data.

  - ``DataArray`` and ``Spec`` handle input/parameter/results data and metadata
  - Groundwork for a PostgreSQL ``DbConfigStore`` implementation

- Separation of ``SosModel`` construction and configuration from ``ModelRun`` running:
  introduce a ``JobScheduler`` that runs directed graphs of simulation jobs, connected by
  dependency edges.

  - Initial ``JobScheduler`` is purely serial
  - Remove ``ModelSet``, removing the capability to handle within-timestep dependency loops
  - Introduce explicit between-timestep dependencies (including model self-dependency)


Version 0.8
===========

Functionality:

- GUI redesiged to include sidebar, jobs, modelrun scheduler
- Decision architecture reaches maturity

  - Initial conditions and pre-specified planning concepts merged
  - Pre-Specified Planning strategies can be defined in model run
  - Strategy contains a list of planning decisions (name, build_year) tuples
  - Interventions file contains list of interventions

- Interventions can be defined in yml or csv format

  - CSV format is parsed so that <attribute_name>_value and <attribute_name>_unit
    suffixes to column names populate a nested dict
    ``{attribute_name: {'value': x, 'unit': y}}`` in memory
  - yml format is declared using ``attribute_name: {'value': x, 'unit': y}}``
    structure

- CLI code refactored out to seperate build, execute, load and setup modules in
  a new ``smif.controller`` subpackage


Version 0.7
===========

Functionality:

- Renamed ScenarioSets parameters to facets which constrain the dimensions of
  data defined in Scenarios
- Numerous functionality and usability improvements to the smif GUI
- Refactored and generalised conversion of space and time to use numpy operations
- Conversion coefficients are cached and loaded instead of being regenerated each run
- Added a warm start argument ``--warm`` to the smif command line inteface which
  resumes a model run from the last successfully completed time interval of a run
- Added timestamps to results
- Add a binary file interface ``-i`` argument to the command line interface that
  writes intermediate model results using pyarrow resulting in much smaller file
  sizes than csv and a great speedup
- Write out a link to the ``smif app`` in the console, instead of opening the app
  in the default browser automatically

Bugs:

- Fixes to the GUI to avoid locking due to threading
- Fixed a bug in datafileinterface where an infinite loop was entered when an
  interval definition did not exist
- Datafileinterface validates data from the set of unique interval and region
  names
- Updated SectorModel calls to region register to return lists of intervals and
  regions in same order as the datafileinterface
- Fixes to the GUI server to enable port-forwarding through a virtual machine
- Fixes bug in smif --warm, where certain keywords caused the warm start to not
  being able to find previous modelrun results
- Fixes loading modelruns interactively, resolve error when loading duplicate
  region/interval definitions
- Fixes region and interval columns of scenario data files are read as integers
  from csv but IDs of regions and intervals could be read as strings or integers
  from shapefiles and csvs respectively raising validation errors


Version 0.6
===========

Functionality:

- Getting started documentation updated to reflect new concepts and
  folder structure
- First version of web app GUI suitable for configuring simulation models,
  system of system models and model runs
- Implemented HTTP API whcih exposes smif data interface to the GUI
- Added ``smif app`` command to start the GUI server and open web package
  from the command line
- Added ``smif setup`` command to copy bundled example project to user folder
- Added functionality to SectorModel wrapper which enables introspection of
  configuration data - managed by the ``DataHandle`` class and accessed at
  runtime in SectorModel.simulate() via the ``self.data`` property. This gives
  access to timesteps, input data, region and interval sets, model parameters.
- Added unit conversion and the ability to load custom units from a file, the
  location to which is specified under the ``units`` key in the project file

Development:

- Build documentation using better-api package to better order and display the
  code on readthedocs
- Added class diagram for data DataHandle class
- Migrated code coverage to codecov.io
- Updated pyscaffold dependency to v3.0 (removes pbr which causes issues with
  e.g. submodules among other things)
- GUI is now built on travis in deploy stage
- Travis build stages are used to separate testing and deployment

Bugs:

- Fixed incorrect datetime parsing
- Fixed assumption over http app location for debug
- Fixed lack of error warning when running a modelrun when no timesteps defined

Version 0.5
===========

- Complete reconfiguration of project folder structure
- Implemented a datalayer

  - Datafileinterface provides read and write methods to file system
  - Databaseinterface will provides read and write methods to database

- Model parameters are passed into a simulation model from narratives
- Added a code of conduct
- Reconfigured builders expect contained objects to be constructed
- Scenario data filtered on available timesteps at runtime
- Updated documentation
- Added prototype (template) smif GUI using web app (in progress)
- Updated command line interface with new commands ``list`` and ``run``
- Introduced concepts of simulation model, scenario model,
  system-of-systems model, narratives and model run.

Version 0.4
===========

- Implemented continuous deployment to PyPi using Travis CI
- Uses numpy arrays for passing data between scenarios and models
- Refactored space-time convertor functions
- Read ModelSet convergence settings from model configuration data
- Added units to model metadata class and require as well as spatial and
  temporal resolutions
- Added UML class diagrams to documentation
- Refactored to create discrete model objects which inherit from an
  abstractclass
- Complete restructuring of package


Version 0.3
===========

- Fast, more compact YAML
- Input, output and pre-specified planning files can now be empty
- State is passed between successive time steps
- Interdependencies (cycles in dependencies) are now supported,
  models are run in cycles stopping at convergence or timeout
- Non-unique time interval definitions are supported

Version 0.2
===========

- Basic conversion of time intervals (aggregation, disaggregation, remapping) and regions (aggregation, disaggregation)
- Results are written out in a yaml dump with the ``-o`` flag e.g. ``smif run -o results.yaml model.yaml``
- Single one-way dependencies with spatio-temporal conversion are supported
- Simplified and harmonised implementation of model inputs and outputs

Version 0.1
===========

- Run a single simulation model for a single timestep
- Provide a model with scenario data and planned interventions
- Configure a model with sets of regions and sets of time intervals for within-
  timestep simulation
