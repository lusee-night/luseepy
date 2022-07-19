# Docker setup for LuSEE-Night

## Docker Hub residency

Images are kept on __Docker Hub__ in repositories belongning to the _lusee_ identity.

## Basic OS and the Python version

This folder helps create and manage Python-based images for the **lusee-night**
CI and other containerized applications. The initial choice of the Python version
is ```3.10.1```, and the base image is derived from __Debian bullseye__:
```python:3.10.1-bullseye```.

Building the images is done from the folder one level above the ```docker``` folder,
and so dockerfiles need to be specified with the ```-f``` option.


## Images

For efficiency reasons, images are built in layers, in the following order:

* foundation
* base
* jupyter

### The "foundation" Image

This is the minimal useable image based on ```requirements-foundation.txt```.
The main "Dockerfile" in the "docker" folder uses a ```build-arg``` argument,
which allows to use any initial requirements file as needed. For example, building the "foundation"
image is done like this:

```bash
docker build . -f docker/Dockerfile -t lusee/lusee-night-foundation:0.1 --build-arg reqs=requirements-foundation.txt
```
This image is published on __Docker Hub__:
* [lusee/lusee-night-foundation:0.1](https://hub.docker.com/repository/docker/lusee/lusee-night-foundation)

```Dockerfile-foundation``` is kept for historical reasons and is deprecated.

### The "base" image

* Based on "foundation", with added ```ARES``` package.
* Uses ```Dockerfile-base```.
* Docker Hub reference: [lusee/lusee-night-base:0.1](https://hub.docker.com/repository/docker/lusee/lusee-night-base)


### The "jupyter" image

* Based on "base", with added ```pyshtools``` and ```jupyterlab``` packages.
*  Uses ```Dockerfile-jupyter```.
* Docker Hub reference: [lusee/lusee-night-jupyter:0.1](https://hub.docker.com/repository/docker/lusee/lusee-night-jupyter)

---

# Misc Notes

## Build

It is suggested that the build is from a folder one level above
this "docker" location. Example:

```bash
docker build -f docker/Dockerfile-jupyter -t buddhasystem/lusee-night-luseepy-jupyter:0.1 .
```

## Misc dependencies

```bash
# Depending on the base system fitsio may need:
pip install wheel
sudo apt-get install libbz2-dev
```


