#!/bin/zsh

conda init zsh
conda activate hcat
python3 setup.py sdist bdist_wheel
conda create -n hcat_test python=3.10 -y
conda activate hcat_test
pip uninstall hcat -y
pip install /Users/chrisbuswinka/Documents/Projects/hcat/dist/hcat-0.0.1-cp310-cp310-macosx_10_9_x86_64.whl
hcat

