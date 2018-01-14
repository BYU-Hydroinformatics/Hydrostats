# -*- coding: utf-8 -*-

from setuptools import setup
setup(
  name = 'hydrostats',
  packages = ['hydrostats'], # this must be the same as the name above
  version = '0.31',
  description = 'Error metrics for use in comparison studies, specifically for use in the field of hydrology',
  author = 'Wade Roberts',
  author_email = 'waderoberts123@gmail.com',
  url = 'https://github.com/waderoberts123/hydrostats', # use the URL to the github repo
  download_url = 'https://github.com/peterldowns/mypackage/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['hydrology', 'error', 'metrics', 'comparison', 'statistics'], # arbitrary keywords
  classifiers = ["License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.6",
],
  install_requires=[
          'numpy',
          'numexpr',
          'pandas'
      ]
)
