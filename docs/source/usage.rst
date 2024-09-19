Usage
======================

.. _installation:

Installation
------------

Docker
------------

To use luseepy, install the LuSEE-Night Docker image (For more information, see https://github.com/lusee-night/luseepy/tree/main/docker):

**The "unity" Image**

This is the minimal useable image based on requirements-foundation.txt. The main "Dockerfile" in the "docker" folder uses a build-arg argument, which also has a reasonable default. For example, building the "unity" image is done like this:

.. code-block:: console

    $ docker build . -f docker/Dockerfile-unity-luseepy -t lusee/lusee-night-unity-luseepy:1.0 --build-arg reqs=requirements-unity-luseepy.txt

**Jupyter**

This image also includes Jupyter Lab software. Jupyter is not started automatically, i.e. by default the user gets bash running and a functional Python/refspec/luseepy environment. To get Jupyter running, one first starts the container like this (or in a similar manner):

.. code-block:: console

    $ docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=docker lusee/lusee-night-unity-luseepy:1.0

It may be neccessary to run docker in interactive mode, especially when using with Docker Desktop on Windows. In that case, add "-it" immediately after the docker run command. Once the container is running, this command is invoked to bring up Jupyter:

.. code-block:: console

    $ jupyter lab --allow-root --ip 0.0.0.0 --port 8888

The port 8888 can be mapped to any other convenient port on the host machine, and then access through localhost by entering "localhost:8888" into the address bar of your browser.

Without Docker
---------------
Navigate to the directory where you wish to keep your luseepy code and check out the luseepy git repository.

.. code-block:: console
    
    $ git clone https://github.com/lusee-night/luseepy.git

In a separate directory, download the LuSEE Google Drive (or the necessary folders from the Drive) here: https://drive.google.com/drive/folders/0AM52i9DVjqkAUk9PVA

Set up the following environment variables:

- ``LUSEEPY_PATH`` -- path to the luseepy checkout
- ``LUSEE_DRIVE_DIR`` -- path to the checkout of the LuSEE-Night Google Drive

Create and activate a conda virtual environment

.. code-block:: console

    $ conda create -n lusee
    $ conda activate lusee

Then install the necessary packages

.. code-block:: console

    $ (lusee) conda install pip flit numpy scipy matplotlib fitsio
    $ (lusee) pip install pyshtools

If you run into installation errors, try the following commands instead:

.. code-block:: console

    $ (lusee) conda install conda-forge::fitsio
    $ (lusee) conda install conda-forge::flit

Go into the luseepy directory and install symlink

.. code-block:: console

    $ (lusee) cd lusee
    $ (lusee) flit install --symlink


The Observation
--------------

.. automodule:: lusee.Observation
   :members:

The Beams
--------------

.. automodule:: lusee.Beam
   :members:

The Gaussian Beam
--------------

.. automodule:: lusee.BeamGauss
   :members:

The Beam Couplings
--------------

.. automodule:: lusee.BeamCouplings
   :members:

The Simulator
--------------

.. automodule:: lusee.Simulation
   :members:

The Satellite classes
---------------------

.. automodule:: lusee.Satellite
   :members:

The Monopole Sky Model classes
----------------------

.. automodule:: lusee.MonoSkyModels
   :members:

The Sky Model classes
----------------------

.. automodule:: lusee.SkyModels
   :members:


The Lunar Calendar
----------------------

.. automodule:: lusee.LunarCalendar
   :members:

The Throughput
--------------

.. automodule:: lusee.Throughput
   :members:

The Data
--------------

.. automodule:: lusee.Data
   :members:

The PCA Analyzer
--------------

.. automodule:: lusee.PCAanalyzer
   :members:
