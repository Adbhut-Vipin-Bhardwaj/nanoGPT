#!/bin/bash

RUNPOD_PROJECT_NAME=$1

mkdir -p app/${RUNPOD_PROJECT_NAME}
cd app/${RUNPOD_PROJECT_NAME}

apt-add-repository multiverse && apt-get update
apt install nvtop
