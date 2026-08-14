# Utilities for running luseepy simulations in batch

Batch runs use a local luseepy checkout. Set the usual environment variables in
the submitting shell (the Condor job definitions use `getenv = True`, so the
job inherits them):

```bash
export LUSEEPY_PATH=/path/to/luseepy
export LUSEE_DRIVE_DIR=/gpfs02/astro/workarea/LuSEE_Drive
```

A single simulation can be run directly:

```bash
python $LUSEEPY_PATH/simulation/driver/run_sim.py $LUSEEPY_PATH/simulation/config/example.yaml
```

The batch driver takes a run config, a batch descriptor file, and the index of
the descriptor line to run:

```bash
python $LUSEEPY_PATH/simulation/driver/run_batch.py \
    $LUSEEPY_PATH/simulation/config/pdr_run.yaml \
    $LUSEEPY_PATH/simulation/config/pdr_config.batch 2
```

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
