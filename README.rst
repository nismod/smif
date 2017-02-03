.. _readme:

====
smif
====

Systems Modelling of Infrastructure Framework

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

**smif** is written in pure Python (Python>=3.5), with very few dependencies.

The optimisation routines currently use GLPK - the GNU Linear Programming Kit.
To install the **glpk** solver:

* on Linux or Mac OSX, you can likely use a package manager, e.g. ``apt install 
  glpk`` for Ubuntu or ``brew install glpk`` for OSX.
* on Windows, `GLPK for Windows <http://winglpk.sourceforge.net/>`_ provide 
  executables. For 64bit Windows, download and unzip the distribution files then
  add the ``w64`` folder to your ``PATH``.

A word from our sponsors
========================

**smif** was written and developed at the `Environmental Change Institute,
University of Oxford <http://www.eci.ox.ac.uk>`_ within the
EPSRC sponsored MISTRAL programme, as part of the `Infrastructure Transition
Research Consortium <http://www.itrc.org.uk/>`_.
