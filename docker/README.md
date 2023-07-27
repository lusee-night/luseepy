# Docker setup for LuSEE-Night

## About

This is work in progress. Please see relavant branches in the `refspec` project
for more information. The aim is to create a small number of unified images
containing expanded functionality, covering both `refspec` and `luseepy`.
Codename is `unity-luseepy`.

Images are kept on __Docker Hub__ in repositories belongning to the _lusee_ identity.

_We base our luseepy images on the refspec Docker images_.  The `refspec` images
are originally derived from `python:3.10.1-bullseye` (Debian). They contain
the `refspec` library and its Python bindings through the `cppyy` package.
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
docker build . -f docker/Dockerfile -t lusee/lusee-night-unity-luseepy:0.1 --build-arg reqs=requirements-unity-luseepy.txt
```


### Jupyter - approach one
This image also includes __Jupyter Lab__ software. Jupyter
is not started automatically, i.e. by default the user gets `bash` running and a functional
Python/refspec/luseepy environment. To get Jupyter running, one first starts the conainer like
this (or in a similar manner):


```bash
docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=docker lusee/lusee-night-unity-luseepy:0.1
```

Once the container is running, this command is invoked to bring up Jupyter:

```bash
jupyter lab --allow-root --ip 0.0.0.0 --port 8888
```

The port 8888 can be mapped to any other convenient port on the host machine,
and then access through `localhost`.

### Jupyter - approach two

It is very convenient to use the "ljupyter" shell function defined in the setup script
one level above this folder. Please read the corresponding README.

## Misc dependencies

```bash
# Depending on the base system fitsio may need:
pip install wheel
sudo apt-get install libbz2-dev
```


