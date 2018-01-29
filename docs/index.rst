smif
----

.. image:: https://img.shields.io/badge/github-nismod%2Fsmif-brightgreen.svg
    :target: https://github.com/nismod/smif/
    :alt: nismod/smif on github

.. image:: https://travis-ci.org/nismod/smif.svg?branch=master
    :target: https://travis-ci.org/nismod/smif
    :alt: Travis CI build status

.. image:: https://img.shields.io/codecov/c/github/nismod/smif/master.svg
    :target: https://codecov.io/gh/nismod/smif?branch=master
    :alt: Code Coverage

.. image:: https://img.shields.io/pypi/v/smif.svg
    :target: https://pypi.python.org/pypi/smif
    :alt: PyPI package

.. image:: https://img.shields.io/conda/vn/conda-forge/smif.svg
    :target: https://anaconda.org/conda-forge/smif
    :alt: conda-forge package

**smif** is a framework for handling the creation of system-of-systems models.
The framework handles inputs and outputs, dependencies between models,
persistence of data and the communication of state across years.

This early version of the framework handles simulation models that simulate the
operation of a system within a year. **smif** will expose an interface to a
planning module which allows different decision-making algorithms to work
against a common API.

**smif** is written in Python (>=3.5) and uses wrappers which implement a
consistent interface in order to run models written in many languages.

A word from our sponsors
========================

**smif** is being written and developed in the `Environmental Change Institute,
in the University of Oxford <http://www.eci.ox.ac.uk>`_ funded by EPSRC as part
of the `Infrastructure Transitions Research Consortium
<http://www.itrc.org.uk/>`_ MISTRAL research programme.

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
   Changelog <changelog>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
