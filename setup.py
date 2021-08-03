"""
Setup configuration for installation via pip
"""

import sys
from setuptools import setup, find_packages


# The find_packages function does a lot of the heavy lifting for us w.r.t.
# discovering any Python packages we ship.
setup(
    name='tensorForm',
    version='0.0.1dev',
    packages=find_packages(),

    # PyPI packages required for the *installation* and usual running of the
    # tools.
    install_requires=[

    ],

    # Metadata for PyPI (https://pypi.python.org).
    description='Manipulate weak forms with tensors.',
)