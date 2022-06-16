#!/usr/bin/env python
import lusee
import numpy  as np
import healpy as hp
import pickle
import os

O=lusee.LObservation('2025-02-01 13:00:00 to 2025-03-01 13:00:00',deltaT_sec=24*3600, lun_lat_deg=-00.0)
B = lusee.LBeam(os.environ['LUSEE_DRIVE_DIR']+'/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits')
print ("Setting up object with FITS beams")
beams = []
for ofs,c in enumerate(["N","E","S","W"]):
    cB = B.rotate(-90*ofs)
    beams.append(cB)
    
lmax = 64
freq=[1,2,3]
sky = lusee.sky.ConstSky(Nside = 32, lmax = lmax, freq=freq, T=200)
S = lusee.Simulator (O,beams, sky, freq=freq, lmax = lmax, combinations=[(0,0),(1,1),(1,3)],
                     Tground = 200. )
print ("Simulating")
WF = S.simulate(times=O.times)

print ("Are we close to 200K?")
assert (np.allclose(WF[:,:2,:],200))
print ("  OK")

print ("Setting up object with Gauss beam")
BG = lusee.LBeam_Gauss(dec_deg=50, sigma_deg=6, phi_deg=180)
beams = [BG]

lmax = 64
freq=[1,5,10]
sky = lusee.sky.ConstSky(Nside = 32, lmax = lmax, freq=freq, T=200)
S = lusee.Simulator (O,beams, sky, freq=freq, lmax = lmax, combinations=[(0,0)],
                     Tground = 0. )
print ("Simulating")
WF = S.simulate(times=O.times)

print ("Are we close to 200K?")
print ("We get:", WF[0,0,0])
assert (np.allclose(WF[:,:2,:],200,rtol=5e-4))
print ("  OK")



