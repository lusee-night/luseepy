export LUSEE_IMAGE="lusee/lusee-night-unity-luseepy:1.2"

# This part was folded in the image: -e PYTHONPATH=/user/luseepy:/user/refspec/cppyy -e LD_LIBRARY_PATH=/user:/usr/local/lib
export PYTHON_SETUP=" -e LUSEE_DRIVE_DIR -e REFSPEC_PATH"

export DEV_MOUNT="-v $LUSEEPY_PATH:/user/luseepy -v $REFSPEC_PATH:/user/refspec"

# Utility funcitons:

lbash() { docker run  -e HOME $PYTHON_SETUP  -w $PWD -v $HOME:$HOME  --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  bash $@; }
lpython() { docker run  -e HOME $PYTHON_SETUP -w $PWD -v $HOME:$HOME  --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  /usr/local/bin/python $@; }

lbash_dev() { docker run  -e HOME $PYTHON_SETUP $DEV_MOUNT -w $PWD -v $HOME:$HOME  --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  bash $@; }
lpython_dev() { docker run  -e HOME $PYTHON_SETUP $DEV_MOUNT -w $PWD -v $HOME:$HOME  --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  /usr/local/bin/python $@; }


ljupyter() { port=9500; docker run  -e HOME $PYTHON_SETUP -w $PWD -v $HOME:$HOME  --user $(id -u):$(id -g) -itp $port:$port $LUSEE_IMAGE  /bin/bash -c "/usr/local/bin/jupyter lab  --ip='*' --port=$port --no-browser  "; }
ljupyter_dev() { port=9600; docker run  -e HOME $PYTHON_SETUP $DEV_MOUNT -w $PWD -v $HOME:$HOME  --user $(id -u):$(id -g) -itp $port:$port $LUSEE_IMAGE  /bin/bash -c "/usr/local/bin/jupyter lab  --ip='*' --port=$port --no-browser  "; }

################################
# --- ATTIC ---
# this seems less reliable, need to investigate: export LUSEEPY_PATH=$(dirname $BASH_SOURCE)

# For your own environment, set these variables as shown in these examples

# LUSEEPY_PATH="/home/user/work/lusee/luseepy"
# REFSPEC_PATH="/home/user/work/lusee/luseepy"
# LUSEE_DRIVE_DIR="/home/user/work/lusee/Drive/"
# LUSEE_OUTPUT_DIR="/home/user/work/lusee/luseepy/simulation/output" 
