# Docker setup for lusee-night

This folder helps create and manage Python-based images for the **lusee-night**
CI and other containerized applications. The initial choice of the Python version
is ```3.10.1```, and the base image is derived from __Debian bullseye__:
```python:3.10.1-bullseye```.

# The "foundation"

The main "Dockerfile" now takes a ```build-arg``` argument:

```bash
docker build . -f docker/Dockerfile -t test:0.1 --build-arg reqs=requirements_short.txt
```

...which allows to use any ```requirements``` file depending on needs.
For example, the "foundation" image includes base packages
described in ```requirements_short.txt```. It's normally published on Docker Hub:

* [buddhasystem/lusee-night-foundation:0.1](https://hub.docker.com/repository/docker/buddhasystem/lusee-night-foundation)

NB. PERSES etc goes on top, to save on build time.

# Build

It is suggested that the build is from a folder one level above
this "docker" location. Under this assumption, this example will build
a ```luseepy-jupyter``` image designed to be hosted on Docker Hub in the
_buddhasystem_ account (hence the name of the tag used). Any tag naming
conventions can be used as needed, of course.

```bash
docker build -f docker/Dockerfile-jupyter -t buddhasystem/lusee-night-luseepy-jupyter:0.1 .
```

# Misc dependencies

```bash
# Depending on the base system fitsio may need:
pip install wheel
sudo apt-get install libbz2-dev
```

# Notes

Add pyshtools

