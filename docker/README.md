# Docker setup for lusee-night

A folder to manage images for the **lusee-night** CI and other
applications that require containers.

Choice of the Python version: the initial choice is 3.10.1 in order
to be close to the SDCC environment.

# The "foundation"

The basic set of packages is contained in the "foundation" image
defined in ```Dockerfile-foundation```. ARES, PERSES and everything
else goes on top, to save on build time.

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
# For fitsio
pip install wheel
sudo apt-get install libbz2-dev
```
