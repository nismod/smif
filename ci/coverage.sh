#!/bin/bash
# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

set -e # exit on error
set -x # echo commands

if [[ "$COVERAGE" == "true" ]]; then
    cd $TRAVIS_BUILD_DIR

    # Upload python coverage
    bash <(curl -s https://codecov.io/bash) -cF python || echo "failed"

    # Run node coverage
    cd $TRAVIS_BUILD_DIR/src/smif/app && npm run-script report

    # Upload javascript coverage
    bash <(curl -s https://codecov.io/bash) -cF javascript || echo "failed"
fi
