.. _readme:

====
smif
====

Simulation Modelling Integration Framework

.. image:: https://travis-ci.org/nismod/smif.svg?branch=master
    :target: https://travis-ci.org/nismod/smif

.. image:: https://readthedocs.org/projects/smif/badge/?version=latest
    :target: http://smif.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/codecov/c/github/nismod/smif/master.svg   
    :target: https://codecov.io/gh/nismod/smif?branch=master
    :alt: Code Coverage

Description
===========

**smif** is a framework for handling the creation of system-of-systems
models.  The framework handles inputs and outputs, dependencies between models,
persistence of data and the communication of state across years.

This early version of the framework handles simulation models that simulate the
operation of a system within a year.
**smif** exposes an interface to a planning module which will allows different
algorithms to be used against a common API.

Setup and Configuration
=======================

**smif** is written in Python (Python>=3.5) and has a number of dependencies.
See `requirements.txt` for a full list.


Using conda
-----------

The recommended installation method is to use `conda
<http://conda.pydata.org/miniconda.html>`_, which handles packages and virtual
environments, along with the `conda-forge` channel which has a host of pre-built
libraries and packages.

Create a conda environment::

    conda create --name smif_env python=3.6

Activate it (run each time you switch projects)::

    activate smif_env

Note that you ``source activate smif_env`` on OSX and Linux (or e.g. Git Bash on
Windows).

Add the conda-forge channel, which has smif available::

    conda config --add channels conda-forge

Finally install ``smif``::

    conda install smif


GLPK
----

The optimisation routines currently use GLPK - the GNU Linear Programming Kit.
To install the **glpk** solver:

* on Linux or Mac OSX, you can likely use a package manager, e.g. ``apt install
  python-glpk glpk-utils`` for Ubuntu or ``brew install glpk`` for OSX.
* on Windows, `GLPK for Windows <http://winglpk.sourceforge.net/>`_ provide
  executables. For 64bit Windows, download and unzip the distribution files then
  add the ``w64`` folder to your ``PATH``.

fiona, GDAL and GEOS
--------------------

We use `fiona <https://github.com/Toblerity/Fiona>`_, which depends on GDAL and
GEOS libraries.

On Mac or Linux these can be installed with your OS package manager, then
install the python packages as usual using::

    # On debian/Ubuntu:
    apt-get install gdal-bin libspatialindex-dev libgeos-dev

    # or on Mac
    brew install gdal
    brew install spatialindex
    brew install geos


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
        {setup,list,run}  available commands
            setup           Setup the project folder
            list            List available model runs
            run             Run a model

        optional arguments:
        -h, --help        show this help message and exit
        -V, --version     show the current version of smif
        -v, --verbose     show messages: -v to see messages reporting on progress,
                            -vv to see debug messages.


A word from our sponsors
========================

**smif** was written and developed at the `Environmental Change Institute,
University of Oxford <http://www.eci.ox.ac.uk>`_ within the
EPSRC sponsored MISTRAL programme, as part of the `Infrastructure Transition
Research Consortium <http://www.itrc.org.uk/>`_.
