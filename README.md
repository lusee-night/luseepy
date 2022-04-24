# luseepy
## About
A set of python utilities for performing various LuSEE Night related calculations.

## Docker
We build images for "vanilla" luseepy and also for Jupyter notebooks, based on same.

ARES and PERSES are being added and are in the process of testing. These dependencies
need to be set for the PERSES to load. SymPy was added to the environment,
and ```pylinex``` was checked out of GitHub.

Example of the provisional setup, will automate later:

```bash
/home/maxim/projects/lusee-night/distpy/:/home/maxim/projects/lusee-night/ares:/home/maxim/projects/lusee-night/pylinex
```

## Developing

To develop on your laptop, the easiest thing is to use the latest docker environment.
Please install docker and pull the image

```
docker pull buddhasystem/lusee-night-luseepy-jupyter:0.1 
```
Next, checkout the lusee repo
```
git clone git@github.com:lusee-night/luseepy.git
```

Next, source the `setup_env.sh` script inside luseepy dir:

```
source lusee_env.sh
```

Now you have 4 utility functions:
 * `lpython` starts runs python using shipped luseepy (unless one is in current dir)
 * `lpython_dev` starts runs python using git checkout luseepy 
 * `ljupyter` starts jupyter notebook using shipped luseepy on port 9500
 * `ljupyter_dev` starts jupyter notebook using git checkout luseepy on port 9600
 
You can now try running
```
lpython tests/LunarCalendarTest.py
```
To start jupyter, simply say
```
ljupyter
```
and then connect to the address given in the terminal output.

## Tests and Singularity

The ```tests``` folder contains CI-related and other testing scripts. Here's an example of a simple test run with Singularity, on a SDCC/BNL node, from the ```luseepy``` folder:
```bash
singularity exec -B /direct/phenix+u/mxmp/projects/luseepy --env PYTHONPATH=/direct/phenix+u/mxmp/projects/luseepy docker://buddhasystem/lusee-night-foundation:0.1 ./tests/LunarCalendarTest.py
```

