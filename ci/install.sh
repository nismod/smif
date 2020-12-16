#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is taken from Scikit-Learn (http://scikit-learn.org/)
#

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda -f
    export PATH=$HOME/miniconda/bin:$PATH
    rm miniconda.sh -f
    # Don't ask for confirmation
    conda config --set always_yes true
    # Update conda
    conda update conda
    # Follow channel priority strictly
    conda config --set channel_priority strict
    # Don't change prompt
    conda config --set changeps1 false

    if [[ "$PYTHON_VERSION" != "3.5" ]]; then
        # Add conda-forge as priority channel
        # conda-forge builds packages for Python 3.6 and above as of 2018-10-01
        conda config --add channels conda-forge
    fi

    # Create the conda environment
    conda create -n testenv --yes \
        python=$PYTHON_VERSION \
        pytest \
        pytest-cov \
        gdal \
        numpy \
        requests \
        rtree \
        xarray \
        pandas \
        psycopg2 \
        pyarrow \
        shapely \
        fiona

    source activate testenv
fi

pip install 'flake8>=3.7'

if [[ "$PYTHON_VERSION" == "3.5" ]]; then
    pip install 'Pint==0.9'
    pip install 'jinja2>=2,<3'
fi

python setup.py develop

# Install node and npm dependencies
nvm install 14.15.1
nvm use 14.15.1
cd $TRAVIS_BUILD_DIR/src/smif/app && npm ci
cd $TRAVIS_BUILD_DIR
