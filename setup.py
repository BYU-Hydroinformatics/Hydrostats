# -*- coding: utf-8 -*-

from setuptools import setup
import os

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(
    name='hydrostats',
    packages=['hydrostats'],
    version='0.72',
    description='Tools for use in comparison studies, specifically for use in the field '
                'of hydrology',
    long_description=README,
    author='Wade Roberts',
    author_email='waderoberts123@gmail.com',
    url='https://github.com/waderoberts123/hydrostats',
    download_url='https://github.com/waderoberts123/Hydrostats/archive/0.72.tar.gz',
    keywords=['hydrology', 'error', 'metrics', 'comparison', 'statistics', 'forecast', 'observed'],
    classifiers=["License :: MIT License",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 ],
    license='MIT',
    install_requires=[
        'numpy >= 1.14.2',
        'numba >= 0.37.0',
        'pandas >= 0.22.0',
        'matplotlib >= 2.2.0',
        'scipy >= 1.0.0',
        'openpyxl >= 2.5.2',
        'HydroErr'
    ],
)
