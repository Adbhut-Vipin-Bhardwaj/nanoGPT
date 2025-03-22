#!/bin/bash

apt-add-repository multiverse
apt-get update
apt install nvtop nano screen

python -m venv .venv
source .venv/bin/activate
pip install torch datasets
