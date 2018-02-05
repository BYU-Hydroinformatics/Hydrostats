# -*- coding: utf-8 -*-

from setuptools import setup
setup(
  name = 'hydrostats',
  packages = ['hydrostats'], # this must be the same as the name above
  version = '0.40',
  description = 'Error metrics for use in comparison studies, specifically for use in the field of hydrology',
  author = 'Wade Roberts',
  author_email = 'waderoberts123@gmail.com',
  url = 'https://github.com/waderoberts123/hydrostats', # use the URL to the github repo
  download_url = 'https://github.com/waderoberts123/Hydrostats/archive/3.7.tar.gz',
  keywords = ['hydrology', 'error', 'metrics', 'comparison', 'statistics', 'forecast', 'observed'], # arbitrary keywords
  classifiers = ["License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.6",
                 ],
  install_requires=[
          'numpy',
          'numba',
          'pandas',
          'matplotlib',
          'scipy',
          'sympy',
      ]
)
