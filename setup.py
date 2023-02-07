#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for smif.

    This file was generated with PyScaffold 3.0.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: http://pyscaffold.readthedocs.org/
"""

import sys

from setuptools import setup


def setup_package():
    needs_sphinx = {"build_sphinx", "upload_docs"}.intersection(sys.argv)
    sphinx = ["sphinx"] if needs_sphinx else []
    setup(
        setup_requires=["setuptools_scm"] + sphinx,
        use_scm_version=True,
        entry_points={"console_scripts": ["smif = smif.cli:main"]},
    )


if __name__ == "__main__":
    setup_package()
