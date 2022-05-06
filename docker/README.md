# Docker setup for lusee-night

A folder to manage images for the **lusee-night** CI and other
applications that require containers.

Choice of the Python version: the initial choice is 3.10.1 in order
to be close to the SDCC environment.

# The "foundation"

The main "Dockerfile" now takes a ```build-arg``` argument:

```bash
docker build . -f docker/Dockerfile -t test:0.1 --build-arg reqs=requirements_short.txt
```

...which allows to use any ```requirements``` file depending on needs.
For example, the "foundation" image includes base packages
described in ```requirements_short.txt```. It's normally published on Docjer Hub:

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

