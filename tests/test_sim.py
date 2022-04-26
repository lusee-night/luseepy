#!/usr/bin/env python
import lusee
import numpy  as np
import healpy as hp
import pickle

O=lusee.LObservation('2025-02-01 13:00:00 to 2025-03-01 13:00:00',deltaT_sec=24*3600, lun_lat_deg=-00.0)
#pickle.dump(O,open("obs.pickle","wb"))
#O=pickle.load(open("obs.pickle","rb"))
    
antenna_sim_path = "../AntennaSimResults/"
fname = "004_Freq1-50MHz_Delta1MHz_AntennaLength1-6m_Delta1m_AntennaAngle75deg_LanderHeight2m/RadiatedElectricField_AntennaLength6m_AntennaAngle75deg_LanderHeight2m_LBoxZ70cm_monopole_Phase+0deg.fits"
#fname = "003_Freq1-50MHz_Delta1MHz_AntennaLength6m_AntennaAngle30deg_LanderHeight2m/RadiatedElectricField_AntennaLength6m_AntennaAngle30deg_LanderHeight2m_monopole.fits"

B = lusee.LBeam(antenna_sim_path+'/'+fname)
B.project_to_phi_theta()

beams = []
for ofs,c in enumerate(["N","E","S","W"]):
    cB = B.rotate(-90*ofs)
    beams.append(cB)

    
lmax = 64

#sky = lusee.ConstSky(128, 200, lmax = lmax)
sky = lusee.sky.GalCenter(32, lmax,50.)

print ("Setting up object")
S = lusee.Simulator (O,beams, sky, freq_ndx=[0,1,2], lmax = lmax, combinations=[(0,0),(0,2),(1,1),(1,3)], Tground = 0 )
#pickle.dump(S,open("Sim.pickle","wb"))
#S=pickle.load(open("Sim.pickle","rb"))
print ("Simulating")
WF = S.simulate(times=O.times)
galt, gaz = O.get_track_l_b (0.,0.)
for time, res, ga in zip(O.times, WF, galt):
    print (time, np.real(res[:,0]), ga)