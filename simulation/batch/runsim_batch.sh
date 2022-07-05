#!/usr/bin/bash
date
export LOCAL=`pwd -P`
cd ${LOCAL}

singularity exec --env LUSEE_OUTPUT_DIR=/output --bind .:/output \
--env LUSEE_DRIVE_DIR=/gpfs02/astro/workarea/LuSEE_Drive --env PYTHONPATH=/app \
-B /gpfs02/astro/workarea/LuSEE_Drive -B ${LOCAL} docker://lusee/lusee-night-jupyter:0.1 \
/app/simulation/driver/run_batch.py  /app/simulation/config/pdr_run.yaml /app/simulation/config/pdr_config.batch $1
