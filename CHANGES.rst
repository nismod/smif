=========
Changelog
=========

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
