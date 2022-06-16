#!/bin/bash

export LUSEE_IMAGE='lusee/lusee-night-jupyter:0.1'
export LUSEE_DRIVE_DIR='/home/maxim/data/lusee/'

/app/simulation/driver/run_sim.py /app/simulation/config/example.yaml
