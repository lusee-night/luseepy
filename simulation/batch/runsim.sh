#!/usr/bin/bash
date
export LOCAL=`pwd -P`
cd ${LOCAL}
singularity exec -B ${LOCAL} docker://lusee/lusee-night-jupyter:0.1 ls


# ${LOCAL}/make_maps.py -v -a -f $1
