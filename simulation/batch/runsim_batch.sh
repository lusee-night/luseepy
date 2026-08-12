#!/usr/bin/bash
date
export LOCAL=`pwd -P`
cd ${LOCAL}
echo Running $1

# Runs against a local luseepy checkout. The Condor job definition uses
# `getenv = True`, so LUSEEPY_PATH and LUSEE_DRIVE_DIR are inherited from the
# submitting shell and must be set there.
export LUSEE_OUTPUT_DIR=${LOCAL}
export PYTHONPATH=${LUSEEPY_PATH}:${LUSEEPY_PATH}/simulation/driver

python ${LUSEEPY_PATH}/simulation/driver/run_batch.py \
    ${LUSEEPY_PATH}/simulation/config/pdr_run.yaml \
    ${LUSEEPY_PATH}/simulation/config/pdr_config.batch $1
