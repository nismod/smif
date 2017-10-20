.. _concept:

Concept
=======

The section outlines the underlying principles of **smif**.


Running a System-of-Systems Model Run
-------------------------------------

Once **smif** has been used to configure a system-of-systems model and model
run, all that is needed to run the model is the command ``smif run
<model_run_name>``.

**smif** handles the loading of input data, spinning up the simulation
models, extracting a graph of dependencies from the network of inputs and
outputs, running the models in the order defined by this graph and finally
persisting state and results from the simulation models to a data store.


Operational Simulation and Planning of Interventions
----------------------------------------------------

Fundamental to the design of **smif** is the distinction between the
simulation of the operation of a particular system,
and the long-term planning of interventions which change the structure of these
systems.

The former is the domain of the simulation models,
while the latter is handled by **smif**.
**smif** provides the architecture to handle the planning of interventions
using one of three approaches: a fully specified approach (a planning pipeline),
a rule based approach and an optimisation approach.

In each of these three approaches, decisions regarding which interventions to
choose are propagated into the model by **smif**.

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
