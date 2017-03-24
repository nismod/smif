.. _concept:

Concept
=======

The section outlines the underlying principles of **smif**.


Running a System-of-Systems Model
---------------------------------

Once **smif** has been used to configure a system-of-systems model, all that is
needed to run the model is the command ``smif run``.

**smif** handles the loading of input data, spinning up the simulation
models, extracting a graph of dependencies from the network of inputs and
outputs, running the models in the order defined by this graph and finally
persisting state and results from the simulation models to a data store.


Operational Simulation and Capacity Expansion
---------------------------------------------

Fundamental to the design of **smif** is the distinction between the
simulation of the operation of a particular system,
and the long-term expansion of the capacity which underpin this operation.

The former is the domain of the simulation models,
while the latter is handled by **smif**.
**smif** provides the architecture to handle the capacity expansion problem
using one of three approaches: a fully specified approach,
a rule based approach and an optimisation approach.

In each of these three approaches, decisions regarding the increase or
decrease in the capacity of an asset are propagated into the model inputs via
a *state transition function*.


State
-----

`State` refers to the information which must be persisted over time.  Normally,
this will refer to the capacity of an asset (e.g. number of wind turbines),
the level of storage (e.g. the volume of water stored in a reservoir).
Other information, including metrics, such as CO\ :sub:`2` emissions,
or cumulative costs, may also be relevant.

**smif** handles `State` for the management of the capacity expansion.
The process of passing state from one time-period to another is managed by
**smif**.  In this respect, note the distinction between time-steps for
the capacity expansion problem, which will normally be measured in years
or decades, versus the time-steps for each instance of a simulation model,
which will run within a year or decade.


Wrapping Simulation Models
--------------------------

At the core of **smif** are the target simulation models which we wish to
integrate into a system-of-systems model. A simple example simulation model
is included in
:class:`tests.fixtures.water_supply.ExampleWaterSupplySimulation`.
A simulation model has inputs, and produces outputs, which are a function of
the inputs.
The :class:`smif.abstract.SectorModel` is used to wrap an individual simulation
model, and provides a uniform API to other parts of **smif**.

An input can correspond to:

- model parameters, whose source is either from a scenario, 
  or from the outputs from another model (a dependency)
- model state (not yet implemented)
