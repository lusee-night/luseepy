#!/usr/bin/bash
date
export LOCAL=`pwd -P`
cd ${LOCAL}

# Runs against a local luseepy checkout. The Condor job definition uses
# `getenv = True`, so LUSEEPY_PATH and LUSEE_DRIVE_DIR are inherited from the
# submitting shell and must be set there.
export LUSEE_OUTPUT_DIR=${LOCAL}
export PYTHONPATH=${LUSEEPY_PATH}

python ${LUSEEPY_PATH}/simulation/driver/run_sim.py ${LUSEEPY_PATH}/simulation/config/example.yaml
