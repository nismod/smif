.. _concepts:

Concepts
========

This section introduces the key concepts used in **smif**. **smif** has been developed for
research into the connections between infrastructure systems, so the examples here are drawn
from water supply and energy systems.

If you prefer a more hands-on approach, try starting with the sample project in :ref:`Getting
Started`.


Simulation Models
-----------------

At the core of **smif** are the simulation models which we want to connect.

A :class:`~smif.model.sector_model.SectorModel` represents a simulation model, and
provides a uniform API to other parts of **smif**.


System-of-Systems Models
------------------------

A system-of-systems model is collection of connected simulation models, represented by
:class:`~smif.model.sos_model.SosModel`.


Inputs, Outputs and Parameters - Using 'Spec' for Metadata
----------------------------------------------------------

A simulation model receives inputs and parameters and produces outputs.

A :class:`~smif.metadata.spec.Spec` defines the name, units, absolute and expected ranges,
dimensions and default values of each input, output and parameter.

Each input, output or parameter may be defined over multiple dimensions.


Dimensions
----------

A dimension is indexed by :class:`~smif.metadata.coordinates.Coordinates` - these might be
spatial (a set of regions), temporal (a set of intervals) or categorical.


Scenarios
---------

A scenario defines data which can be used as input to simulation models. The data that a
scenario provides is defined by a :class:`~smif.metadata.spec.Spec` in the same way as model
outputs.

A scenario has at least one variant. A scenario with several variants could be used to explore
different possibilities for uncertain exogenous quantities.

For example, a climate scenario might have variants for each of the representative
concentration pathways (RCPs), where each variant provides data for temperature, precipitation
and insolation.


Narratives
----------

A narrative defines data which can be used to provide parameters to simulation models.

Like a scenario, a narrative has at least one variant. However, it is possible to compose
multiple narrative variants, where values may be overridden as defined by the order in which
variants are selected.

If a model parameter is not provided by any narrative, **smif** will fall back to providing the
default value for that parameter, as defined by its spec.


Dependencies
------------

A dependency represents the link between a data source and the place it is required. A
simulation model input might depend on the output of another simulation model, or on data from
a scenario.

**smif** won't do any data processing behind the scenes, so dependencies can only link inputs
and outputs where the metadata match exactly on units, dtype and dimensions. It is okay for
names and suggested ranges not to match. For example, a model providing 'regional_gva' can be
happily be connected to an input called 'rGVA' as long as the rest of the spec matches.


Adaptors
--------

When models need to exchange data, the units or dimensions used might not match exactly. For
example, one model might provide residential and commercial electricity demand in MWh on an
hourly basis, per Local Authority District, where another model needs to know total electricity
demand in GWh, per NUTS3 region.

An :class:`~smif.convert.adaptor.Adaptor` can convert between units, or aggregate or
disaggregate along dimensions. Various presets are available in :py:mod:`smif.convert`.


Interventions
-------------

Interventions change how a simulated system operates. An intervention can represent building or
upgrading a physical thing (like a reservoir or power station), or could be something less
tangible like imposing a congestion charging zone over a city centre.

A system of interest can in principle be composed entirely of a series of interventions. For
example, the electricity generation and transmission system is composed of a set of generation
sites (power stations, wind farms...), transmission lines and bus bars.


Decision Models
---------------

**smif** makes a sharp distinction between *simulating* the operation of a system, and
*deciding* on which interventions to introduce to meet goals or constraints on the whole
system-of-systems.

A decision model might use one of three approaches: a fully specified approach (testing a given
planning pipeline), a rule based approach (using some heuristic rules), or an optimisation
approach.

In each of these three approaches, the decision model provides a bundle of interventions and
planning timesteps, which are then simulated, after which the decision model may request
further simulation of different timesteps and/or combinations of interventions.


Model Runs
----------

A model run brings together all of the above:

- a system-of-systems model, comprising:

  - simulation models
  - scenarios, providing input data
  - dependencies, connecting inputs and outputs
  - narratives, providing parameter values

- the choice of which scenario and narrative variants to use
- decision models
- the choice of which strategy configurations to use

A project might develop several sets of model runs, perhaps in order to methodically explore
combinations of scenarios and strategies, or to run different combinations of models against
a shared library of scenarios.
