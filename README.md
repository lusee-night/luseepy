# luseepy
![workflow](https://github.com/lusee-night/luseepy/actions/workflows/luseepy-test.yml/badge.svg)

## About
A set of python utilities for performing various LuSEE-Night related calculations.

## Documentation

This software is documented on the ["Read the Docs" pages](https://luseepy.readthedocs.io/en/latest/)

## Cached Data

There are datasets stored in the LuSEE-Night [Google Drive](https://drive.google.com/drive/folders/0AM52i9DVjqkAUk9PVA).

## Docker
For details, see the [README](docker/README.md) file in the `docker` folder. The current setup
involves building the _luseepy_ image based on the _refspec_ image. The latter contains C++ based
software interfaces by means of the _cppyy_ package.

## Developing

To develop on your laptop, the easiest thing is to use the latest docker environment.
Please install docker and pull the image

```
docker pull lusee/lusee-night-unity-luseepy:1.0 # or other appropriate version.
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
 
To better understand the settings, it's worth it to take a look at the contents of `setup_env.sh`.
In the current version, the `HOME` directory is mounted in the container, so it's possible to
develop against the full current version of the luseepy/refspec suite. For example, you can now try running

```
# Simple calendar test
lpython tests/LunarCalendarTest.py
# to start jupyter
ljupyter
```
and then connect to the address given in the terminal output.

## Environment variables

User is expected to set up the following environment variables:

 * `LUSEEPY_PATH` -- path to the luseepy checkout
 * `LUSEEOPSIM_PATH` -- path to the lusee opsim package (if used).
 * `LUSEE_DRIVE_DIR` -- path to the checkout of the LuSEE-Night Google Drive
 
The following environment variables are set up by the `setup_env.sh` script:

 * `LUSEE_IMAGE` -- docker image that has everything to run lusee



## Singularity

__NB. The example below corresponds to an early verion of software, and reference to the image below is deprected.__

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


## Starting with simulations

Go to `simulation` sub-directory. Make sure the `$LUSEE_DRIVE_DIR` points to the stuff from the LUSEE Drive that the simulations needs (ULSA maps, beam). Run a short simulation as

```
python driver/run_sim.py config/realistic_example.yaml
```

In the same directory, open a jupyter notebook and plot the results for the NE cross-correlation, imaginary part as:
```
import lusee
D=lusee.Data('output/sim_output.fits')
plt.imshow(D[:,'01I',:],aspect='auto',extent=(D.freq[0], D.freq[-1],len(D.times),0))
plt.colorbar()
plt.xlabel('frequency (MHz)')
plt.ylabel('time number')
```
