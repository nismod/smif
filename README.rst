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

.. image:: https://coveralls.io/repos/github/nismod/smif/badge.svg?branch=master
    :target: https://coveralls.io/github/nismod/smif?branch=master

Description
===========

**smif** is a framework for handling the creation of system-of-systems
models.  The framework handles inputs and outputs, dependencies between models,
persistence of data and the communication of state across years.

This early version of the framework handles simple models that simulate the
operation of a system.
**smif** will eventually implement optimisation routines which will allow,
for example, the solution of capacity expansion problems.

Setup and Configuration
=======================

**smif** is written in Python (Python>=3.5) and has a number of dependencies.
See `requirements.txt` for a full list.

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

    pip install -r requirements.txt

On Windows, the simplest approach seems to be using
`conda <http://conda.pydata.org/miniconda.html>`_, which handles packages and
virtual environments, along with the `conda-forge` channel which has a host of
pre-built libraries and packages.

Create a conda environment::

    conda create --name smif python=3.5 numpy scipy

Activate it (run each time you switch projects)::

    activate smif

Note that you ``source activate smif`` on OSX and Linux.

Add the conda-forge channel, which has shapely and fiona available::

    conda config --add channels conda-forge


Install python packages, along with GDAL and dependencies::

    conda install fiona shapely rtree
    pip install -r requirements.txt


Installing `smif`
=================

Once the dependencies are installed on your system,
a normal installation of `smif` can be achieved using pip on the command line::

        pip install smif

Versions under development can be installed from github using pip too::

        pip install git+http://github.com/nismod/smif#egg=v0.2

The suffix ``#egg=v0.2`` refers to a specific version of the source code.
Omitting the suffix installs the latest version of the library.

To install from the source code in development mode::

        git clone http://github.com/nismod/smif
        cd smif
        python setup.py develop


A word from our sponsors
========================

**smif** was written and developed at the `Environmental Change Institute,
University of Oxford <http://www.eci.ox.ac.uk>`_ within the
EPSRC sponsored MISTRAL programme, as part of the `Infrastructure Transition
Research Consortium <http://www.itrc.org.uk/>`_.
