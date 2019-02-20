# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as f:
    README = f.read()

setup(
    name='hydrostats',
    packages=['hydrostats'],
    version='0.75',
    description='Tools for use in comparison studies, specifically for use in the field '
                'of hydrology',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Wade Roberts',
    author_email='waderoberts123@gmail.com',
    url='https://github.com/waderoberts123/hydrostats',
    download_url='https://github.com/waderoberts123/Hydrostats/archive/0.75.tar.gz',
    keywords=['hydrology', 'error', 'metrics', 'comparison', 'statistics', 'forecast', 'observed'],
    classifiers=["Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 ],
    license='MIT',
    install_requires=[
        'numpy',
        'numba',
        'pandas',
        'matplotlib',
        'scipy',
        'HydroErr'
    ],
)
