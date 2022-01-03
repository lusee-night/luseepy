import sys
import numpy as np
sys.path.append('.')
import lusee

L = lusee.LObservation()
# coordinates of crab nebular
crab_track = L.get_track_ra_dec(ra='05h34m31.94s',dec='+22d00m52.2s')
print ("max alt of crab:", crab_track[0].max()/np.pi*180, "degrees")

