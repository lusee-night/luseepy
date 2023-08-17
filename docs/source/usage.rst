Usage
======================

.. _installation:

Installation
------------

To use luseepy, go to the GitHub repo

.. code-block:: console

   (.venv) $ # commands here

As an alternative, there are Docker images in development (TBD).

The Beams
--------------

.. currentmodule:: lusee.Beam

Utility functions

.. autofunction:: getLegendre

.. autofunction:: grid2healpix_alm_reference

.. autofunction:: grid2healpix_alm_fast

.. autofunction:: grid2healpix

.. autofunction:: project_to_theta_phi

The main beam class

.. currentmodule:: None

.. autoclass:: lusee.Beam_class

.. autofunction:: lusee.Beam_class.rotate

.. autofunction:: lusee.Beam_class.flip_over_yz

.. autofunction:: lusee.Beam_class.power

.. autofunction:: lusee.Beam_class.power_stokes

.. autofunction:: lusee.Beam_class.cross_power

.. autofunction:: lusee.Beam_class.sky_fraction

.. autofunction:: lusee.Beam_class.ground_fraction

.. autofunction:: lusee.Beam_class.power_hp

.. autofunction:: lusee.Beam_class.copy_beam

.. autofunction:: lusee.Beam_class.plotE

.. autofunction:: lusee.Beam_class.get_healpix

The Simulator
--------------

.. currentmodule:: lusee.Simulation

Utility functions

.. autofunction:: mean_alm

.. autofunction:: rot2eul

.. autofunction:: eul2rot

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

.. currentmodule:: None

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

