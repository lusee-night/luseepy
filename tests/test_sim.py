#!/usr/bin/env python
import lusee
import numpy  as np
import healpy as hp
import pickle

O=lusee.LObservation('2025-02-01 13:00:00 to 2025-03-01 13:00:00',deltaT_sec=24*3600)
pickle.dump(O,open("obs.pickle","wb"))

O=pickle.load(open("obs.pickle","rb"))
    
antenna_sim_path = "../AntennaSimResults/"
fname = "004_Freq1-50MHz_Delta1MHz_AntennaLength1-6m_Delta1m_AntennaAngle75deg_LanderHeight2m/RadiatedElectricField_AntennaLength6m_AntennaAngle75deg_LanderHeight2m_LBoxZ70cm_monopole_Phase+0deg.fits"
B = lusee.LBeam(antenna_sim_path+'/'+fname)
beams = []
for ofs,c in enumerate(["N","E","S","W"]):
    cB = B.rotate(-90*ofs)
    beams.append(cB)

class ConstSkyAbove:
    def __init__ (self,Nside, T):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        Tmap = T*np.ones(self.Npix)
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        Tmap[theta>0.75*np.pi] = 0 
        self.mapalm = hp.map2alm(Tmap)
        self.frame = "MCMF"

    def get_alm(self, ndx):
        return [self.mapalm]*len(ndx)


class GalCenter:
    def __init__ (self,Nside, T):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        Tmap = np.exp(-(phi)**2/0.1-(theta-np.pi/2)**2/0.1)
        self.mapalm = hp.map2alm(T*Tmap)
        self.frame = "galactic"

    def get_alm(self, ndx):
        return [self.mapalm]*len(ndx)

    
#sky = ConstSkyAbove(16, 200)
sky = GalCenter(16, 1e4)

print ("Setting up object")
#S = lusee.Simulator (O,beams, sky, freq_ndx=[0,1],combinations=[(0,0),(0,2)])
#pickle.dump(S,open("Sim.pickle","wb"))
S=pickle.load(open("Sim.pickle","rb"))
print ("Simulating")
WF = S.simulate(times=O.times)
print (WF.shape)
print (WF[:,0,0],len(WF[:,0,0]))
