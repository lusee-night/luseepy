#!/usr/bin/env python
import lusee
import numpy  as np
import healpy as hp
import pickle
import os

O=lusee.LObservation('2025-02-01 13:00:00 to 2025-03-01 13:00:00',deltaT_sec=24*3600, lun_lat_deg=-00.0)
B = lusee.LBeam(os.environ['LUSEE_DRIVE_DIR']+'/AntennaResponse/Exported/Example/feko_bnl_monopole_1m_75deg.fits')

beams = []
for ofs,c in enumerate(["N","E","S","W"]):
    cB = B.rotate(-90*ofs)
    beams.append(cB)

    
lmax = 64

sky = lusee.sky.ConstSky(Nside = 32, lmax = lmax, T=200)

print ("Setting up object")
S = lusee.Simulator (O,beams, sky, freq_ndx=[0,1,2], lmax = lmax, combinations=[(0,0),(1,1),(1,3)],
                     Tground = 200. )
print ("Simulating")
WF = S.simulate(times=O.times)
galt, gaz = O.get_track_l_b (0.,0.)

print ("Are we close to 200K?")
assert (np.allclose(WF[:,:2,:],200))
print ("  OK")
