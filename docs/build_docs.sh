#!/bin/bash

make html
cd ../../Hydrostats_docs
cp -R html/* .
rm -rfd html
rm -rfd doctrees