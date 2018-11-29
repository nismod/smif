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
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda -f
    export PATH=$HOME/miniconda/bin:$PATH
    rm miniconda.sh -f

    conda update --yes conda -c conda-forge

    conda config --add channels conda-forge
    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes -c conda-forge \
        python=$PYTHON_VERSION \
        pytest \
        pytest-cov \
        numpy \
        rtree \
        xarray \
        pandas \
        psycopg2 \
        shapely \
        fiona && source activate testenv
fi

python setup.py develop

# Install node and npm dependencies
nvm install 6.11.4
nvm use 6.11.4
cd $TRAVIS_BUILD_DIR/src/smif/app && npm install
cd $TRAVIS_BUILD_DIR
