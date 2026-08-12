Usage
======================

.. _installation:

Installation
------------

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
