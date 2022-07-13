# Utilities for running luseepy simulations in batch

__


Example of settings for Docker experimentation:
```bash
export LUSEE_IMAGE='lusee/lusee-night-jupyter:0.1'
export LUSEE_DRIVE_DIR='/home/maxim/data/lusee/'
#
# In the above, note the local folder for the LuSEE data.
# At the SDCC facility, the path would be /gpfs02/astro/workarea/LuSEE_Drive e.g.

export LUSEE_DRIVE_DIR=/gpfs02/astro/workarea/LuSEE_Drive
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

---

# The simulation driver

## Testing with Docker

```bash
docker run -it --rm -v $LUSEE_DRIVE_DIR:/data -v $PWD:/output --env LUSEE_DRIVE_DIR=/data --env PYTHONPATH=/app --env LUSEE_OUTPUT_DIR=/output --entrypoint /app/simulation/driver/run_sim.py lusee/lusee-night-jupyter:0.1 /app/simulation/config/example.yaml
```


## Testing with Singularity - the basic driver (interactive)

```bash
singularity exec --env LUSEE_OUTPUT_DIR=/output --bind .:/output --env LUSEE_DRIVE_DIR=/gpfs02/astro/workarea/LuSEE_Drive --env PYTHONPATH=/app -B /gpfs02/astro/workarea/LuSEE_Drive -B ${LOCAL} docker://lusee/lusee-night-jupyter:0.1 /app/simulation/driver/run_sim.py  /app/simulation/config/example.yaml
```

## Testing with Singularity - the modified batch driver (interactive)

```bash
singularity exec --env LUSEE_OUTPUT_DIR=/output --bind .:/output --env LUSEE_DRIVE_DIR=/gpfs02/astro/workarea/LuSEE_Drive --env PYTHONPATH=/app -B /gpfs02/astro/workarea/LuSEE_Drive -B `pwd`:/app docker://lusee/lusee-night-jupyter:0.1 python /app/simulation/driver/run_batch.py /app/simulation/config/pdr_run.yaml /app/simulation/config/pdr_config.batch 2
```
The last argument is the simulation descriptor number to be picked from `pdr_config.batch`.
For Condor integration please see the next paragraph.

## Submitting to HTCondor

* The Condor job definition is contained the file `runsim_batch.job`.
* This jobs definition requires a mandatory parameter: `runs`. This parameter defines the size of the Condor cluster (i.e. the number of jobs to be simultaneously created in one batch run)
* The executable in `runsim_batch.job` is `runsim_batch.sh`
* `runsim_batch.sh` is a utility wrapper around `run_batch.py` which updates the environment variable `PYTHONPATH` to ensure the script runs in the batch mode.

Every HTCondor job has the internal `ProcId` identifier which will be used to refer to a specific line in the file `pdr_config.batch`.
This is achieved by using `ProcId` as an argument to `runsim_batch.sh`. For this to work properly, the `runs` parameter should be less or
equal to the number of entries in this file.

```bash
condor_submit runs=2 runsim_batch.job
```
