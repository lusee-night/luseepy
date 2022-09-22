#! /usr/bin/env python

from lusee import LObservation, LSatellite, ObservedSatellite
import numpy as np

###

print("Creating Observation")
L = LObservation(2501, deltaT_sec=12 * 24 * 3600)
print("Where is Jupiter? ...", end="")
(alt, az) = L.get_track_solar("jupiter")
assert np.allclose(alt, np.array([-0.016094472754065968, 0.36446102683880527, -0.742282586599336, 1.0234862551211383]))
assert np.allclose(az, np.array([1.5576265729704628, 4.90516887823782, 1.9543177609988713, 5.589033405608409]))
print("   OK")

print("Where is Crab? ...", end="")
(alt, az) = L.get_track_ra_dec(ra="05h34m31.94s", dec="+22d00m52.2s")
assert np.allclose(alt, np.array([-0.20558058508392482, 0.5453246924579669, -0.8712342373563664, 1.1075629720529663]))
assert np.allclose(az, np.array([1.658400973508976, 4.989332762046785, 2.1139005746252524, 5.818167146594609]))
print("   OK")

print("Where is LPF? ...", end="")
S = LSatellite()
OS = ObservedSatellite(L, S)

(alt, az, dist) = OS.alt_rad(), OS.az_rad(), OS.dist_km()
assert np.allclose(alt, np.array([0.21588382388423585, -0.6236591760852072, -0.8960429960535476, 0.13056920428783125] ))
assert np.allclose(az, np.array([2.4739743855124954, 3.4561462659437576, 2.9963806694454393, 3.8503235017446786]))
assert np.allclose(dist, np.array([7148.589465319025, 8831.89598250433, 5878.029418859661, 8569.951255742464]))
print("   OK")


print("All OK!")
