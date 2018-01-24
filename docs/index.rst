smif
----

.. image:: https://img.shields.io/badge/github-nismod%2Fsmif-brightgreen.svg
    :target: https://github.com/nismod/smif/
    :alt: nismod/smif on github

.. image:: https://travis-ci.org/nismod/smif.svg?branch=master
    :target: https://travis-ci.org/nismod/smif
    :alt: Travis CI build status

.. image:: https://coveralls.io/repos/github/nismod/smif/badge.svg?branch=master
    :target: https://coveralls.io/github/nismod/smif?branch=master
    :alt: Coveralls code coverage

.. image:: https://img.shields.io/pypi/v/smif.svg
    :target: https://pypi.python.org/pypi/smif
    :alt: PyPI package

.. image:: https://img.shields.io/conda/vn/conda-forge/smif.svg
    :target: https://anaconda.org/conda-forge/smif
    :alt: conda-forge package

**smif** (a simulation modelling integration framework) is a framework for
handling the creation of system-of-systems models. The framework handles inputs
and outputs, dependencies between models, persistence of data and the
communication of state across years.

This early version of the framework handles simulation models that simulate the
operation of a system within a year and exposes an interface to a planning
module which will allow different algorithms to be used against a common API.

Contents
========

.. toctree::
   :maxdepth: 2

   Getting Started <getting_started>
   Concept <concept>


.. toctree::
   :maxdepth: 3

   Reference <api/modules>


.. toctree::
   :maxdepth: 1

   Developing `smif` <developers>
   Contributing <contributing>
   License <license>
   Authors <authors>
   Changelog <changes>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
