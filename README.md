# smif

Simulation Modelling Integration Framework

[![GitHub Actions build status](https://github.com/nismod/smif/actions/workflows/test.yml/badge.svg)](https://github.com/nismod/smif/actions/workflows/test.yml)

[![PyPI package](https://img.shields.io/pypi/v/smif.svg)](https://pypi.python.org/pypi/smif)

[![conda-forge package](https://img.shields.io/conda/vn/conda-forge/smif.svg)](https://anaconda.org/conda-forge/smif)

[![Archive](https://zenodo.org/badge/67128476.svg)](https://zenodo.org/badge/latestdoi/67128476)

[![Journal of Open Research Software paper](https://img.shields.io/badge/JORS-10.5334%2fjors.265-blue.svg)](https://doi.org/10.5334/jors.265)

[![Documentation Status](https://readthedocs.org/projects/smif/badge/?version=latest)](https://smif.readthedocs.io/en/latest/?badge=latest)

**smif** is a framework for handling the creation, management and
running of system-of-systems models.

A system-of-systems model is a collection of system simulation models
that are coupled through dependencies on data produced by each other.

**smif** provides a user with the ability to

- create system-of-systems models
  - add simulation models to a system-of-systems model
  - create dependencies between models by linking model inputs and
    outputs
  - pick from a library of data adapters which perform common data
    conversions across dependencies
  - create user-defined data adapters for more special cases
  - add scenario data sources and link those to model inputs within a
    system-of-systems
- add a simulation model to a library of models
  - write a simulation model wrapper which allows **smif** to run the
    model
  - define multi-dimensional model inputs, outputs and parameters and
    appropriate metadata
- run system-of-systems models
  - link concrete scenario data sets to a system-of-systems model
  - define one or more decision modules that operate across the
    system-of-systems
  - define a narrative to parameterise the contained models
  - persist intermediate data for each model output, and write results
    to a data store for subsequent analysis

In summary, the framework facilitates the hard coupling of complex
systems models into a system-of-systems.

## Should I use **smif**?

There are number of practical limits imposed by the implementation of
**smif**. These are a result of a conscious design decision that stems
from the requirements of coupling the infrastructure system models to
create the next generation National Infrastructure System Model
(NISMOD2).

The discussion below may help you determine whether **smif** is an
appropriate tool for you.

- **smif** _is not_ a scheduler, but has been designed to make
  performing system-of-systems analyses with a scheduler easier
- Geographical extent is expected to be defined explicitly by a vector
  geometry
  - **smif** _is not_ optimised for models which simulate on a grid,
    though they can be accomodated
  - **smif** _is_ designed for models that read and write spatial data
    defined over irregular grids or polygons using any spatial format
    readable by [fiona](https://github.com/Toblerity/Fiona)
- Inputs and outputs are exchanged at the ‘planning timestep’ resolution
  - **smif** makes a distinction between simulation of operation, which
    happens at a model-defined timestep resolution, and application of
    planning decisions which happens at a timestep which is synchronised
    between all models
  - **smif** _is not_ focussed on tight coupling between models which
    need to exchange data at every simulation timestep (running in
    lockstep)
  - **smif** _does_ accomodate individual models with different spatial
    and temporal (and other dimensional) resolutions, by providing data
    adaptors to convert from one resolution to another
- **smif** has been designed to support the coupling of bottom-up,
  engineering simulation models built to simulate the operation of a
  given infrastructure system
  - **smif** _provides_ a mechanism for passing information from the
    system-of-systems level (at planning timesteps scale) to the
    contained models
  - **smif** _is_ appropriate for coupling large complex models that
    exchange resources and information at relatively course timesteps
- **smif** is not appropriate for
  - discrete event system simulation models (e.g. queuing systems)
  - dynamical system models (e.g. predator/prey)
  - equilibrium models without explicit timesteps (e.g. Land-Use
    Transport Interaction)
  - for simulating 100s of small actor-scale entities within a
    system-level environment

## Installation and Configuration

**smif** is written in Python (Python\>=3.5) and has a number of
dependencies. See <span class="title-ref">requirements.txt</span> for a
full list.

### Using micromamba

The recommended installation method is to use
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
which handles packages and virtual environments, along with the
<span class="title-ref">conda-forge</span> channel which has a host of
pre-built libraries and packages.

Create an environment:

    micromamba env create -n smif --channel conda-forge --channel nodefaults python=3.12 smif

Activate it (run each time you switch projects):

    micromamba activate smif

### Installing <span class="title-ref">smif</span> with other methods

Once the dependencies are installed on your system, a normal
installation of <span class="title-ref">smif</span> can be achieved
using pip on the command line:

    pip install smif

Versions under development can be installed from github using pip too:

    pip install git+http://github.com/nismod/smif

To install from the source code in development mode:

    git clone http://github.com/nismod/smif
    cd smif
    pip install -e .

### Spatial libraries

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

## Running <span class="title-ref">smif</span> from the command line

Follow the [getting started
guide](http://smif.readthedocs.io/en/latest/getting_started.html) to
help set up the necessary configuration.

To set up an sample project in the current directory, run:

    $ smif setup

To list available model runs:

    $ smif list
    demo_model_run
    ...

To run a system-of-systems model run:

    $ smif run demo_model_run
    ...
    Model run complete

By default, results will be stored in a results directory, grouped by
model run and simulation model.

To see all options and flags:

    $ smif --help
    usage: smif [-h] [-V] {setup,list,run} ...

    Command line tools for smif

    positional arguments:
    {setup,list,run}  available commands
        setup               Setup the project folder
        list                List available model runs
        run                 Run a model

    optional arguments:
    -h, --help        show this help message and exit
    -V, --version     show the current version of smif

## Citation

If you use **smif** for research, please cite the software directly:

- Will Usher, Tom Russell, Roald Schoenmakers, Craig Robson, Fergus
  Cooper, Thibault Lestang & Rose Dickinson. (2019). nismod/smif vX.Y.Z
  (Version vX.Y.Z). Zenodo. <http://doi.org/10.5281/zenodo.1309336>

Here's an example BibTeX entry:

    @misc{smif_software,
          author       = {Will Usher and Tom Russell and Roald Schoenmakers and Craig Robson and Fergus Cooper and Thibault Lestang and Rose Dickinson},
          title        = {nismod/smif vX.Y.Z},
          month        = Aug,
          year         = 2018,
          doi          = {10.5281/zenodo.1309336},
          url          = {https://doi.org/10.5281/zenodo.1309336}
    }

Please also cite the software description paper:

- Will Usher and Tom Russell. (2019) A Software Framework for the
  Integration of Infrastructure Simulation Models. Journal of Open
  Research Software, 7: 16 DOI: <https://doi.org/10.5334/jors.265>

Here's an example BibTeX entry:

    @misc{smif_paper,
          author       = {Will Usher and Tom Russell},
          title        = {A Software Framework for the Integration of Infrastructure Simulation Models},
          journal      = {Journal of Open Research Software},
          volume       = {7},
          number       = {16},
          pages        = {1--5},
          month        = May,
          year         = {2019},
          doi          = {10.5334/jors.265},
          url          = {https://doi.org/10.5334/jors.265}
    }

## Acknowledgments

**smif** was written and developed at the [Environmental Change
Institute, University of Oxford](http://www.eci.ox.ac.uk) within the
EPSRC sponsored MISTRAL programme, as part of the [Infrastructure
Transition Research Consortium](http://www.itrc.org.uk/).
