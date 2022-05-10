#! /usr/bin/env python

from lusee import LObservation, LSatellite, ObservedSatellite
import numpy as np

###

print("Creating Observation")
L = LObservation(2501, deltaT_sec=12 * 24 * 3600)
print("Where is Jupiter? ...", end="")
(alt, az) = L.get_track_solar("jupiter")
assert np.allclose(alt, np.array([-0.05129193, 0.43802611, -0.84537346, 1.21373041]))
assert np.allclose(az, np.array([1.56099317, 4.81741633, 1.7388161, 5.27575664]))
print("   OK")

print("Where is Crab? ...", end="")
(alt, az) = L.get_track_ra_dec(ra="05h34m31.94s", dec="+22d00m52.2s")
assert np.allclose(alt, np.array([-0.25924483, 0.6322757, -1.00320146, 1.32990645]))
assert np.allclose(az, np.array([1.61330243, 4.84754857, 1.84215658, 5.53884205]))
print("   OK")

print("Where is LPF? ...", end="")
S = LSatellite()
OS = ObservedSatellite(L, S)
(alt, az) = OS.alt_rad(), OS.az_rad()
assert np.allclose(alt, np.array([0.75436834, 1.28474677, 0.03689239, -0.90291079]))
assert np.allclose(az, np.array([2.70386501, 6.1162517, 2.53861365, 4.47444415]))
print("   OK")


print("All OK!")
