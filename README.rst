.. _readme:

====
smif
====

Simulation Modelling Integration Framework

.. image:: https://travis-ci.org/nismod/smif.svg?branch=master
    :target: https://travis-ci.org/nismod/smif
    :alt: Travis CI build status

.. image:: https://ci.appveyor.com/api/projects/status/g1x12yfwb4q9kjad/branch/master?svg=true
    :target: https://ci.appveyor.com/project/nismod/smif
    :alt: Appveyor CI Build status

.. image:: https://img.shields.io/codecov/c/github/nismod/smif/master.svg
    :target: https://codecov.io/gh/nismod/smif?branch=master
    :alt: Code Coverage

.. image:: https://img.shields.io/pypi/v/smif.svg
    :target: https://pypi.python.org/pypi/smif
    :alt: PyPI package

.. image:: https://img.shields.io/conda/vn/conda-forge/smif.svg
    :target: https://anaconda.org/conda-forge/smif
    :alt: conda-forge package

.. image:: https://zenodo.org/badge/67128476.svg
   :target: https://zenodo.org/badge/latestdoi/67128476
   :alt: Archive

.. image:: https://readthedocs.org/projects/smif/badge/?version=latest
   :target: https://smif.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**smif** is a framework for handling the creation, management and running of system-of-systems
models.

A system-of-systems model is a collection of system simulation models that are coupled through
dependencies on data produced by each other.

**smif** provides a user with the ability to

- create system-of-systems models

  - add simulation models to a system-of-systems model
  - create dependencies between models by linking model inputs and outputs
  - pick from a library of data adapters which perform common data conversions across
    dependencies
  - create user-defined data adapters for more special cases
  - add scenario data sources and link those to model inputs within a system-of-systems

- add a simulation model to a library of models

  - write a simulation model wrapper which allows **smif** to run the model
  - define multi-dimensional model inputs, outputs and parameters and appropriate metadata

- run system-of-systems models

  - link concrete scenario data sets to a system-of-systems model
  - define one or more decision modules that operate across the system-of-systems
  - define a narrative to parameterise the contained models
  - persist intermediate data for each model output, and write results to a data store for
    subsequent analysis

In summary, the framework facilitates the hard coupling of complex systems models into a
system-of-systems.

Should I use **smif**?
======================

There are number of practical limits imposed by the implementation of **smif**.
These are a result of a conscious design decision that stems from the requirements of
coupling the infrastructure system models to create the next generation
National Infrastructure System Model (NISMOD2).

The discussion below may help you determine whether **smif** is an appropriate
tool for you.

- **smif** *is not* a scheduler, but has been designed to make performing
  system-of-systems analyses with a scheduler easier

- Geographical extent is expected to be defined explicitly by a vector geometry

  - **smif** *is not* optimised for models which simulate on a grid,
    though they can be accomodated
  - **smif** *is* designed for models that read and write spatial data
    defined over irregular grids or polygons using any spatial format readable
    by `fiona <https://github.com/Toblerity/Fiona>`_

- Inputs and outputs are exchanged at the ‘planning timestep’ resolution

  - **smif** makes a distinction between simulation of operation, which happens
    at a model-defined timestep resolution, and application of
    planning decisions which happens at a timestep which is synchronised
    between all models
  - **smif** *is not* focussed on tight coupling between models which need to exchange
    data at every simulation timestep (running in lockstep)
  - **smif** *does* accomodate individual models with different spatial and temporal
    (and other dimensional) resolutions, by providing data adaptors to convert from one
    resolution to another

- **smif** has been designed to support the coupling of bottom-up, engineering
  simulation models built to simulate the operation of a given infrastructure system

  - **smif** *provides* a mechanism for passing information from the system-of-systems
    level (at planning timesteps scale) to the contained models
  - **smif** *is* appropriate for coupling large complex models that exchange
    resources and information at relatively course timesteps

- **smif** is not appropriate for

  - discrete event system simulation models (e.g. queuing systems)
  - dynamical system models (e.g. predator/prey)
  - equilibrium models without explicit timesteps (e.g. Land-Use Transport Interaction)
  - for simulating 100s of small actor-scale entities within a system-level environment

Installation and Configuration
==============================

**smif** is written in Python (Python>=3.5) and has a number of dependencies.
See `requirements.txt` for a full list.

Using conda
-----------

The recommended installation method is to use `conda
<http://conda.pydata.org/miniconda.html>`_, which handles packages and virtual environments,
along with the `conda-forge` channel which has a host of pre-built libraries and packages.

Create a conda environment::

    conda create --name smif_env python=3.6

Activate it (run each time you switch projects)::

    conda activate smif_env

Add the conda-forge channel, which has smif available::

    conda config --add channels conda-forge

Finally install ``smif``::

    conda install smif


Installing `smif` with other methods
------------------------------------

Once the dependencies are installed on your system,
a normal installation of `smif` can be achieved using pip on the command line::

        pip install smif

Versions under development can be installed from github using pip too::

        pip install git+http://github.com/nismod/smif

To install from the source code in development mode::

        git clone http://github.com/nismod/smif
        cd smif
        python setup.py develop


Spatial libraries
-----------------

``smif`` optionally depends on `fiona <https://github.com/Toblerity/Fiona>`_ and `shapely
<https://github.com/Toblerity/Shapely>`_, which depend on the GDAL and GEOS libraries. These
add support for reading and writing common spatial file formats and for spatial data
conversions.

If not using conda, on Mac or Linux these can be installed with your OS package manager::

    # On debian/Ubuntu:
    apt-get install gdal-bin libspatialindex-dev libgeos-dev

    # or on Mac
    brew install gdal
    brew install spatialindex
    brew install geos

Then to install the python packages, run::

    pip install smif[spatial]


Running `smif` from the command line
====================================

Follow the `getting started guide
<http://smif.readthedocs.io/en/latest/getting_started.html>`_ to help set up the
necessary configuration.

To set up an sample project in the current directory, run::

        $ smif setup

To list available model runs::

        $ smif list
        demo_model_run
        ...

To start the smif app, a user-interface that helps to display, create and edit a configuration,
run::

        $ smif app

To run a system-of-systems model run::

        $ smif run demo_model_run
        ...
        Model run complete

By default, results will be stored in a results directory, grouped by model run
and simulation model.

To see all options and flags::

        $ smif --help
        usage: smif [-h] [-V] [-v] {setup,list,run} ...

        Command line tools for smif

        positional arguments:
        {setup,list,app,run}  available commands
            setup               Setup the project folder
            list                List available model runs
            app                 Open smif app
            run                 Run a model

        optional arguments:
        -h, --help        show this help message and exit
        -V, --version     show the current version of smif
        -v, --verbose     show messages: -v to see messages reporting on progress,
                            -vv to see debug messages.

Citation
========

If you use **smif** for research, please cite the software directly:

* Will Usher, Tom Russell, & Roald Schoenmakers. (2018). nismod/smif
  vX.Y.Z (Version vX.Y.Z). Zenodo. http://doi.org/10.5281/zenodo.1309336

Here's an example BibTeX entry::

        @misc{smif,
              author       = {Will Usher and Tom Russell and Roald Schoenmakers},
              title        = {nismod/smif vX.Y.Z},
              month        = Aug,
              year         = 2018,
              doi          = {10.5281/zenodo.1309336},
              url          = {https://doi.org/10.5281/zenodo.1309336}
        }


A word from our sponsors
========================

**smif** was written and developed at the `Environmental Change Institute, University of Oxford
<http://www.eci.ox.ac.uk>`_ within the EPSRC sponsored MISTRAL programme, as part of the
`Infrastructure Transition Research Consortium <http://www.itrc.org.uk/>`_.
