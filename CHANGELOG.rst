=========
Changelog
=========

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
