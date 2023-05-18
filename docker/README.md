# Docker setup for LuSEE-Night

## About

This is work in progress. Please see relavant branches in the `refspec` project
for more information. The aim is to create a small number of unified images
containing expanded functionality, covering both `refspec` and `luseepy`.
Codename is `unity-luseepy`.

Images are kept on __Docker Hub__ in repositories belongning to the _lusee_ identity.

_We base our luseepy images on the refspec Docker images_.  The `refspec` images
are originally derived from `python:3.10.1-bullseye` (Debian). The contain
the `refspec` library and its Python bindings through the `cppyy` package.

As usual, building the images is done from the folder one level above the `docker` folder,
and so dockerfiles need to be specified with the `-f` option.


## Images

### The "unity" Image

This is the minimal useable image based on ```requirements-foundation.txt```.
The main "Dockerfile" in the "docker" folder uses a ```build-arg``` argument,
which also has a reasonable default. For example, building the "unity"
image is done like this:

```bash
docker build . -f docker/Dockerfile -t lusee/lusee-night-unity-luseepy:0.1 --build-arg reqs=requirements-foundation.txt
```

### Jupyter Lab

An image with Jupyter included is built on top of `lusee/lusee-night-unity-luseepy:0.1`
and is named `lusee/lusee-night-unity-jupyter:0.1`.

```bash
docker run -p 8000:8888 -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=docker lusee/lusee-night-unity-jupyter:0.1
```

By default, internally this command is invoked at start up:

```bash
jupyter lab --allow-root --ip 0.0.0.0 --port 8888
```

The port 8888 can be mapped to any other convenient port on the host machine.

## Misc dependencies

```bash
# Depending on the base system fitsio may need:
pip install wheel
sudo apt-get install libbz2-dev
```


