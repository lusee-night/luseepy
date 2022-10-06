Usage
======================

.. _installation:

Installation
------------

To use luseepy, go to the GitHub repo

.. code-block:: console

   (.venv) $ # commands here

As an alternative, there are Docker images in development (TBD).

The Simulator
--------------

The main simulator class

.. autoclass:: lusee.Simulator


Beam preparation

.. autofunction:: lusee.Simulator.prepare_beams

The "simulate" method

.. autofunction:: lusee.Simulator.simulate

Writing out data

.. autofunction:: lusee.Simulator.write

The Satellite classes
---------------------

.. autoclass:: lusee.Satellite

.. autoclass:: lusee.ObservedSatellite

The 'Fit Sky' classes
---------------------

.. autoclass:: lusee.FitsSky
