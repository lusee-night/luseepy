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

Utility functions

.. autofunction:: simulation.mean_alm

.. autofunction:: simulation.rot2eul

.. autofunction:: simulation.eul2rot

The main simulation class

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

.. autofunction:: lusee.Satellite.predict_position_mcmf

.. autoclass:: lusee.ObservedSatellite

.. autofunction:: lusee.ObservedSatellite.alt_rad

.. autofunction:: lusee.ObservedSatellite.az_rad

.. autofunction:: lusee.ObservedSatellite.dist_km

.. autofunction:: lusee.ObservedSatellite.get_transit_indices

.. autofunction:: lusee.ObservedSatellite.plot_tracks

.. autofunction:: lusee.ObservedSatellite.get_track_coverage   

The Sky Models classes
----------------------

.. autoclass:: lusee.GalCenter

.. autoclass:: lusee.FitsSky


'Map alm'

.. autofunction:: lusee.FitsSky.get_alm

