export LUSEE_IMAGE=buddhasystem/lusee-night-luseepy-jupyter:0.1
export LUSEEPY_PATH=$PWD

lpython() { docker run  -e HOME -e PYTHONPATH=/app -w $PWD -v $HOME:$HOME -e LUSEE_DRIVE_DIR --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  python $@; }
ljupyter() { port=9500; docker run  -e HOME -e PYTHONPATH=/app -w $PWD -v $HOME:$HOME -e LUSEE_DRIVE_DIR --user $(id -u):$(id -g) -itp $port:$port $LUSEE_IMAGE  /bin/bash -c "/usr/local/bin/jupyter notebook  --ip='*' --port=$port --no-browser  "; }

lpython_dev() { docker run  -e HOME -e PYTHONPATH=$LUSEEPY_PATH -e LUSEE_DRIVE_DIR -w $PWD -v $HOME:$HOME --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  python $@; }
ljupyter_dev() { port=9600; docker run  -e HOME -e PYTHONPATH=$LUSEEPY_PATH -e LUSEE_DRIVE_DIR -w $PWD -v $HOME:$HOME --user $(id -u):$(id -g) -itp $port:$port $LUSEE_IMAGE  /bin/bash -c "/usr/local/bin/jupyter notebook  --ip='*' --port=$port --no-browser  "; }
