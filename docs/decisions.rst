.. _decisions:

Strategies, Interventions and Decision Modules
==============================================

**smif** makes a sharp distinction between *simulating* the operation of a system, and
*deciding* on which interventions to introduce to meet goals or constraints on the whole
system-of-systems.

The decision aspects of **smif** include a number of components.

- The DecisionManager interacts with the ModelRunner and provides a list of
  timesteps and iterations to run
- The DecisionManager also acts as the interface to a user implemented DecisionModule,
  which may implement a particular decision approach.

A decision module might use one of three approaches:

- a rule based approach (using some heuristic rules), or
- an optimisation approach.

A pre-specified approach (testing a given planning pipeline) is included in the
core **smif** code.

The Decision Manager
--------------------

A DecisionManager is initialised with a DecisionModule implementation. This is
referenced in the strategy section of a Run configuration.

The DecisionManager presents a simple decision loop interface to the model runner,
in the form of a generator which allows the model runner to iterate over the
collection of independent simulations required at each step.

The DecisionManager collates the output of the decision algorithm and
writes the post-decision state to the store. This allows Models
to access a given decision state in each timestep and decision iteration id.

Decision Module Implementations
-------------------------------

Users must implement a DecisionModule and pass this to the DecisionModule by
declaring it under a ``strategy`` section of a Run configuration.

The DecisionModule implementation influences the combination and ordering of
decision iterations and model timesteps that need to be performed to complete
the run. To do this, the DecisionModule implementation must yield a bundle
of interventions and planning timesteps, which are then simulated,
after which the decision module may request further simulation of different
timesteps and/or combinations of interventions.

The composition of the yielded bundle will change as a function of the implementation
type. For example, a rule-based approach is likely to iterate over individual
years until a threshold is met before proceeding.

A DecisionModule implementation can access results of previous iterations using
methods available on the ResultsHandle it is passed at runtime. These include
``ResultsHandle.get_results``.  The property ``DecisionModule.available_interventions``
returns the entire collection of interventions that are available for deployment
in a particular iteration.

Interventions
-------------

Interventions change how a simulated system operates.
An intervention can represent a building or upgrading a physical thing
(like a reservoir or power station), or could be something less
tangible like imposing a congestion charging zone over a city centre.

A system of interest can in principle be composed entirely of a series of interventions. For
example, the electricity generation and transmission system is composed of a set of generation
sites (power stations, wind farms...), transmission lines and bus bars.

A simulation model has access to several methods to obtain its current *state*.
The DataHandle.get_state and DataHandle.get_current_interventions provide
direct access the database of interventions relevant for the current timestep.

Deciding on Interventions
-------------------------

The set of all interventions $I$ includes all interventions for all models in a
system of systems.
As the Run proceeds,
and interventions are chosen by the DecisionModule implementation,
then the set of available interventions is modified.

Set of pre-specified or planned interventions $P{\subset}I$

Available interventions $A=P{\cap}I$

Decisions at time t ${D_t}\subset{A}-{D_{t-1}}$

Pre-Specified Planning
----------------------

In a pre-specified planning strategy, a pipeline of interventions is forced into
the system-of-systems.

This requires the provision of data and configuration, described step by step below

- define the set of interventions
- define the planning strategy
- add the pre-specified strategy to the model run configuration

Define interventions
~~~~~~~~~~~~~~~~~~~~

Interventions are associated with an individual model, listed in a csv file and
added to the model configuration as described in the project configuration part
of the documentation <project_configuration>.

Note that each intervention is identified by a ``name`` entry that must be unique
across the system of systems. To ensure this, one suggestion is to use a pre-fix
with the initals of the sector model to which the intervention belows.

An example intervention file has the headers

- name
- location
- capacity_value
- capacity_units
- operational_lifetime_value
- operational_lifetime_units
- technical_lifetime_value
- technical_lifetime_units
- capital_cost_value
- capital_cost_units

and contents as follows::

        nuclear_large,Oxford,1000,MW,40,years,25,years,2000,million £
        carrington_retire,Oxford,-500,MW,0,years,0,years,0,million £

Define the planning strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A planning strategy consists of the set of (name, build_timestep) tuples, where
each name must belong to the set of interventions.

An example from the sample project looks like this::

        name,build_year
        nuclear_large,2010
        carrington_retire,2015
        ac_line1,2010

Add the pre-specified strategy to the model run configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final step is to add the pre-specified planning stategy to the run
configuration::

    strategies:
    - type: pre-specified-planning
      description: Future energy plan
      filename: energy_supply/strategies/plan.csv

The entry should take the above format, where the filename entry refers to the
planning strategy file composed in step two.