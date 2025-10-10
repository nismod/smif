# Installation

**smif** is written in Python (Python >=3.11 ) and has a number of dependencies.
See `requirements.txt` for a full list.

## Using micromamba

The recommended installation method is to use
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
which handles packages and virtual environments, along with the
`conda-forge` channel which has a host of
pre-built libraries and packages.

Create an environment:

    micromamba env create -n smif --channel conda-forge --channel nodefaults python=3.12 smif

Activate it (run each time you switch projects):

    micromamba activate smif

## Installing `smif` with other methods

Once the dependencies are installed on your system, a normal
installation of `smif` can be achieved
using pip on the command line:

    pip install smif

Versions under development can be installed from github using pip too:

    pip install git+http://github.com/nismod/smif

To install from the source code in development mode:

    git clone http://github.com/nismod/smif
    cd smif
    pip install -e .

## Spatial libraries

`smif` optionally depends on [fiona](https://github.com/Toblerity/Fiona)
and [shapely](https://github.com/Toblerity/Shapely), which depend on the
GDAL and GEOS libraries. These add support for reading and writing
common spatial file formats and for spatial data conversions.

If not using conda, on Mac or Linux these can be installed with your OS
package manager:

    # On debian/Ubuntu:
    apt-get install gdal-bin libspatialindex-dev libgeos-dev

    # or on Mac
    brew install gdal
    brew install spatialindex
    brew install geos

Then to install the python packages, run:

    pip install smif[spatial]

## Running `smif` from the command line

Follow the [getting started
guide](http://smif.readthedocs.io/en/latest/getting_started.html) to
help set up the necessary configuration.

To set up an sample project in the current directory, run:

    $ smif setup
