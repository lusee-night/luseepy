#! /usr/bin/env python
'''
Another test of the LObservation class
'''

import numpy as np
from lusee.observation import LObservation

###
L = LObservation()

# Coordinates of the Crab nebula
crab_track = L.get_track_ra_dec(ra='05h34m31.94s',dec='+22d00m52.2s')

print ("Max alt of the Crab nebula:", crab_track[0].max()/np.pi*180, "degrees")

