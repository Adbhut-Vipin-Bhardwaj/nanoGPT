#!/bin/bash

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_PROJECT_NAME=$3

scp -P ${RUNPOD_PORT} -p ./setup_runpod.sh root@${RUNPOD_HOST}:~/app/${RUNPOD_PROJECT_NAME}
scp -P ${RUNPOD_PORT} -r ./datasets root@${RUNPOD_HOST}:~/app/${RUNPOD_PROJECT_NAME}
scp -P ${RUNPOD_PORT} *.py root@${RUNPOD_HOST}:~/app/${RUNPOD_PROJECT_NAME}
scp -P ${RUNPOD_PORT} requirements.txt root@${RUNPOD_HOST}:~/app/${RUNPOD_PROJECT_NAME}
