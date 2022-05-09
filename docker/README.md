# Docker setup for lusee-night

## Basic OS and the Python version

This folder helps create and manage Python-based images for the **lusee-night**
CI and other containerized applications. The initial choice of the Python version
is ```3.10.1```, and the base image is derived from __Debian bullseye__:
```python:3.10.1-bullseye```.

Building the images is done from the folder one level above the ```docker``` folder,
and so dockerfiles need to be specified with the ```-f``` option.

## The "foundation" Image

This is the minimal useable image based on ```requirements-foundation.txt```.
The main "Dockerfile" in the "docker" folder now uses a ```build-arg``` argument,
which allows to use any requirements file. For example, building the "foundation"
image is done like this (you would need to change that to reflect the tag associated with
your own account):

```bash
docker build . -f docker/Dockerfile -t buddhasystem/lusee-night-foundation:0.1 --build-arg reqs=requirements-foundation.txt
```
This image is published on __Docker Hub__:
* [buddhasystem/lusee-night-foundation:0.1](https://hub.docker.com/repository/docker/buddhasystem/lusee-night-foundation)

## The "base" image

The "base" image is based on "foundation, but with added ARES package. Uses ```Dockerfile-base```. Docker Hub reference:
* [buddhasystem/lusee-night-base:0.1](https://hub.docker.com/repository/docker/buddhasystem/lusee-night-base)


## The "pyshtools" image

Based on "base", but with added ```pyshtools``` package. Uses ```Dockerfile-pyshtools```. Docker Hub reference:
* [buddhasystem/lusee-night-pyshtools:0.1](https://hub.docker.com/repository/docker/buddhasystem/lusee-night-pyshtools)

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

## Notes

May 9, 2022: Added pyshtools

