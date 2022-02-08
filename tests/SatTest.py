#! /usr/bin/env python

from lusee.lunar_satellite  import LSatellite
from lusee.observation      import LObservation

###
O = LObservation()
S = LSatellite()

pos = S.predict_position_mcmf(O.times)

print(pos)
