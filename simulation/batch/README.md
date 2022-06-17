# Utilities for running luseepy simulations in batch

Example of settings for Docker experimentation:
```bash
export LUSEE_IMAGE='lusee/lusee-night-jupyter:0.1'
export LUSEE_DRIVE_DIR='/home/maxim/data/lusee/' # note the local foler
```

The starting point is the bash function defined in the setup file:
```bash

lpython() { docker run  -e HOME -e PYTHONPATH=/app -w $PWD -v $HOME:$HOME -e LUSEE_DRIVE_DIR --user $(id -u):$(id -g) -it  $LUSEE_IMAGE  python $@; }

lpython_dev driver/run_sim.py config/example.yaml
```

An example of a custom _entrypoint_:

```bash
$ docker run -it --entrypoint /bin/bash $IMAGE_NAME -s
```

The updated Jupyter image can be run with the LUSEE data mounted to the new `/data ` folder.

```bash
$ docker run -it --rm -v $LUSEE_DRIVE_DIR:/data --env LUSEE_DRIVE_DIR=/data lusee/lusee-night-jupyter:0.1 bash
```

# The prototype

```bash
docker run -it --rm -v $LUSEE_DRIVE_DIR:/data --env LUSEE_DRIVE_DIR=/data --env PYTHONPATH=/app --entrypoint /app/simulation/batch/run_sim_docker.sh lusee/lusee-night-jupyter:0.1
```

