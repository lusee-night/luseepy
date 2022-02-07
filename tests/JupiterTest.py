#! /usr/bin/env python

from lusee.observation import LObservation
###

L           = LObservation()
(alt, az)   = L.get_track_solar('jupiter')

print(*alt, sep=' ')
print(*az, sep=' ')
