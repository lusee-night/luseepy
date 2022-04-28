#!/usr/bin/env python
import lusee
import numpy  as np
import healpy as hp
import pickle
import sys

cc = int(sys.argv[1])
lat  = 0.0
Tground = 0
Dt = 24*3600
if cc==0:
    fname = 'output/sim_v1.fits'
elif cc==1:
     fname = 'output/sim_v1_csky.fits'
elif cc==2:
     fname = 'output/sim_v1_gcen.fits'
elif cc==3:
     fname = 'output/sim_v1_slat.fits'
     lat = -30
elif cc==4:
     fname = 'output/sim_v1_lmax32.fits'
elif cc==5:
     fname = 'output/sim_v1_const200.fits'
elif cc==6:
    fname = 'output/sim_v1_rsig.fits'

elif cc==7:
    fname = 'output/sim_v1_fine.fits'
    Dt = 60

O=lusee.LObservation('2025-02-01 13:00:00 to 2025-02-28 13:00:00',deltaT_sec=Dt, lun_lat_deg=lat)
    
antenna_sim_path = "../../Drive/AntennaResponse/Exported/"
antenna_fname = "feko_bnl_monopole_1m_75deg.fits"

B = lusee.LBeam(antenna_sim_path+'/'+antenna_fname)

beams = []
for ofs,c in enumerate(["N","E","S","W"]):
    cB = B.rotate(-90*ofs)
    beams.append(cB)

lmax = 64 if cc!=4 else 32
sky = lusee.sky.FitsSky ('sky_data/ULSA_32_ddi_smooth.fits', lmax = lmax)
skymeans = sky.maps.mean(axis=1)
if cc == 1:
    sky = lusee.sky.ConstSky(32,lmax,skymeans)
if cc == 2:
    sky = lusee.sky.GalCenter(32,lmax,skymeans*10)
if cc == 5:
    sky = lusee.sky.ConstSky(32,lmax,200.)
    Tground=200.
if cc == 6:
    sky = lusee.sky.ConstSky(32,lmax,[lusee.monosky.T_DarkAges(f) for f in np.arange(1,51)])

    
print ("Setting up object")
S = lusee.Simulator (O,beams, sky, freq_ndx=np.arange(50), lmax = lmax,
        combinations=[(0,0),(0,2),(2,2),(0,1),(1,1),(1,3),(3,3),(0,3),(1,2),(2,3)], Tground = Tground )
pickle.dump(S,open("Sim.pickle","wb"))
S=pickle.load(open("Sim.pickle","rb"))

print ("Simulating")
WF = S.simulate(times=O.times)
#galt, gaz = O.get_track_l_b (0.,0.)
#for time, res, ga in zip(O.times, WF, galt):
#    print (time, np.real(res[:,0]), ga)
print ("Writing to",fname)
S.write(fname)
