Usage
======================

.. _installation:

Installation
------------

To use luseepy, install the LuSEE-Night Docker image (For more information, see https://github.com/lusee-night/luseepy/tree/main/docker):

**The "unity" Image**

This is the minimal useable image based on requirements-foundation.txt. The main "Dockerfile" in the "docker" folder uses a build-arg argument, which also has a reasonable default. For example, building the "unity" image is done like this:

.. code-block:: bash
    docker build . -f docker/Dockerfile-unity-luseepy -t lusee/lusee-night-unity-luseepy:1.0 --build-arg reqs=requirements-unity-luseepy.txt

**Jupyter**

This image also includes Jupyter Lab software. Jupyter is not started automatically, i.e. by default the user gets bash running and a functional Python/refspec/luseepy environment. To get Jupyter running, one first starts the container like this (or in a similar manner):

.. code-block:: bash
docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=docker lusee/lusee-night-unity-luseepy:0.1

Once the container is running, this command is invoked to bring up Jupyter:

.. code-block:: bash
jupyter lab --allow-root --ip 0.0.0.0 --port 8888

The port 8888 can be mapped to any other convenient port on the host machine, and then access through localhost.

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
