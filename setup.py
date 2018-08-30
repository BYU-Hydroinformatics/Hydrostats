# -*- coding: utf-8 -*-

from setuptools import setup
import os

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(
    name='hydrostats',
    packages=['hydrostats'],  # this must be the same as the name above
    version='0.71',
    description='Tools for use in comparison studies, specifically for use in the field '
                'of hydrology',
    long_description=README,
    author='Wade Roberts',
    author_email='waderoberts123@gmail.com',
    url='https://github.com/waderoberts123/hydrostats',  # use the URL to the github repo
    download_url='https://github.com/waderoberts123/Hydrostats/archive/0.71.tar.gz',
    keywords=['hydrology', 'error', 'metrics', 'comparison', 'statistics', 'forecast', 'observed'],
    classifiers=["License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.6",
                 ],
    install_requires=[
        'numpy >= 1.14.2',
        'numba >= 0.37.0',
        'pandas >= 0.22.0',
        'matplotlib >= 2.2.0',
        'scipy >= 1.0.0',
        'sympy >= 1.1.1',
        'openpyxl >= 2.5.2',
        'HydroErr'
    ],
)
