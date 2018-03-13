=========
Changelog
=========

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

Bugs:

- Fixes to the GUI to avoid locking due to threading
- Fixed a bug in datafileinterface where an infinite loop was entered when an 
  interval definition did not exist
- Datafileinterface validates data from the set of unique interval and region 
  names


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
- GUI is now build on travis in deploy stage
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
- Complete restructuring of packagea


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
