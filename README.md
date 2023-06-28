# luseepy
![workflow](https://github.com/lusee-night/luseepy/actions/workflows/luseepy-test.yml/badge.svg)

## About
A set of python utilities for performing various LuSEE Night related calculations.
_Caveat_: as of April 2023, this software is undergoing various changes, and documentation
is being fixed. Pardon the dust.

## Cached Data

There are datasets stored on the LuSEE-Night [Google Drive](https://drive.google.com/drive/folders/0AM52i9DVjqkAUk9PVA){:target="_blank"}.

## Docker
For details, see the [README](docker/README.md) file in the `docker` folder.

## Developing

To develop on your laptop, the easiest thing is to use the latest docker environment.
Please install docker and pull the image

```
docker pull lusee/lusee-night-jupyter:0.1
```
Next, checkout the lusee repo
```
git clone git@github.com:lusee-night/luseepy.git
```

Next, source the `setup_env.sh` script inside luseepy dir and even better, put it into your `.bashrc`.

```
source setup_env.sh
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

## Environment variables

Among others, the following environment variables are set up by the `setup_env.sh` script:

 * `LUSEEPY_PATH` -- path to the luseepy checkout
 * `LUSEE_IMAGE` -- docker image that has everything to run lusee
 * `LUSEE_DRIVE_DIR` -- path to the checkout of the LuSEE-Night Google Drive


## Tests and Singularity

The `tests` folder contains CI-related and other testing scripts. Here's an example
of a simple test run with Singularity, on a SDCC/BNL node, from the `luseepy` folder:

```bash
singularity exec -B /direct/phenix+u/mxmp/projects/luseepy --env PYTHONPATH=/direct/phenix+u/mxmp/projects/luseepy docker://lusee/lusee-night-foundation:0.1 ./tests/LunarCalendarTest.py
```


## Cutting a new version

Cutting a new version entails:
 * having a clean (non dev) version in `__init__.py`
 * updating `setup_env.sh`
 * tagging the github
 * making new docker image
 * bumping version again in `__init__.py` to a +0.1 and a dev
 
Any small fixes after the fact should be cumping version by 0.01.
Large changes that break API should bump version into next integer.

