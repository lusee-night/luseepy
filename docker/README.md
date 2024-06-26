# Docker setup for LuSEE-Night

## About

The aim of this project is to create a software stack which includes
the `refspec` _(Refernce Spectrometer)_ and the `luseepy` toolkit on top of it.
The main method of deployment for this software is a Docker
image covering both `refspec` and `luseepy`. Images thus created contain "unity-luseepy"
in their names. They are kept on __Docker Hub__ in repositories belongning to the
_lusee_ identity.

The `refspec` images are built on top of the base `python:3.10.1-bullseye` (Debian).
They contain the C++ `refspec` library and its Python bindings through the `cppyy` package.
Currently, the principal image is names `lusee/lusee-night-refspec-cppyy`.
The _luseepy_ image is built on top of that.

As usual, building the images is done from the folder one level above the `docker` folder,
and so dockerfiles need to be specified with the `-f` option.

## Images

### The "unity" Image

This is the minimal useable image based on ```requirements-foundation.txt```.
The main "Dockerfile" in the "docker" folder uses a ```build-arg``` argument,
which also has a reasonable default. For example, building the "unity"
image is done like this:

```bash
docker build . -f docker/Dockerfile-unity-luseepy -t lusee/lusee-night-unity-luseepy:1.0 --build-arg reqs=requirements-unity-luseepy.txt
```


### The legacy dockerfile

There is an appropriately labeled `legacy` Dockerfile, which differs
from the main official one in that it's using a locally cached data in the
`.astropy` folder. This file is kept to give the user a bit more flexibility
in how these data are managed. The updated file actually forces generation
of these data at runtime.

### Jupyter in VScode

The user has the ability to run Jupyter notebooks transparently within the VScode
environment, if they choose so. The easiest way to ensure that the OS environment
is inherited from the shell is to start VScode from the command line, e.g.

```bash
$ code
```


### Jupyter - approach one
This image also includes __Jupyter Lab__ software. Jupyter
is not started automatically, i.e. by default the user gets `bash` running and a functional
Python/refspec/luseepy environment. To get Jupyter running, one first starts the conainer like
this (or in a similar manner):


```bash
docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=docker lusee/lusee-night-unity-luseepy:1.0
```

It may be neccessary to run docker in interactive mode, especially when using with Docker Desktop on Windows. In that case, add "-it" immediately after the docker run command. Once the container is running, this command is invoked to bring up Jupyter:

```bash
jupyter lab --allow-root --ip 0.0.0.0 --port 8888
```

The port 8888 can be mapped to any other convenient port on the host machine,
and then access through `localhost` by entering: "localhost:8888" into the address bar of your browser.

### Jupyter - approach two

It is very convenient to use the "ljupyter" shell function defined in the setup script
one level above this folder. Please read the corresponding README.

## Misc dependencies

```bash
# Depending on the base system fitsio may need:
pip install wheel
sudo apt-get install libbz2-dev
```


