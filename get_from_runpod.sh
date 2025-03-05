#!/bin/bash

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_PROJECT_NAME=$3

mkdir -p ./models
scp -P ${RUNPOD_PORT} -r root@${RUNPOD_HOST}:~/app/${RUNPOD_PROJECT_NAME}/models/* ./models/
