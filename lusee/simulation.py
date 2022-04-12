from .observation import LObservation
from .LBeam import LBeam, grid2healpix_alm_fast
import numpy as np
import healpy as hp

def mean_alm(alm1, alm2, lmax):
    prod = alm1*np.conj(alm2)
    sm = (np.real(prod[:lmax+1]).sum()+2*np.real(prod[lmax+1:]).sum())/(4*np.pi)
    return sm

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R



class Simulator:

    def __init__ (self, obs, beams, sky_model, 
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], lmax = 128,
                  taper = 0.03, Tground = 200.0, freq_ndx=None):
        self.obs = obs
        self.sky_model = sky_model
        self.lmax = lmax
        self.taper = taper
        self.Tground = Tground
        if freq_ndx is None:
            self.freq_ndx = np.arange(beams[0].Nfreq)
        else:
            self.freq_ndx = freq_ndx
        self.freq=beams[0].freq[self.freq_ndx]

        self.prepare_beams (beams, combinations)


    def prepare_beams(self,beams, combinations):
        self.efbeams = []
        thetas = beams[0].theta
        #gtapr = np.zeros(len(thetas))
        gtapr = (np.arctan((thetas-np.pi/2)/self.taper)/np.pi+0.5)**2
        tapr = 1.0 - gtapr
        bomega = []
        for b in beams:
            P = b.power()[self.freq_ndx,:,:]
            beamnorm =  np.array([grid2healpix_alm_fast(b.theta,b.phi[:-1], np.real(P[fi,:,:-1]),
                                                   lmax=1)[0]/np.sqrt(4*np.pi) for fi in self.freq_ndx])
            bomega.append(np.real(beamnorm))

        
        for i,j in combinations:
            bi , bj = beams[i], beams[j]
            xP = bi.cross_power(bj)[self.freq_ndx,:,:]
            norm = np.sqrt(bomega[i]*bomega[j])
            beam = xP*tapr[None,:,None]/norm[:,None,None]
            ground = xP*gtapr[None,:,None]/norm[:,None,None]
            ## now need to transfrom this to healpy
            beamreal =  np.array([grid2healpix_alm_fast(bi.theta,bi.phi[:-1], np.real(beam[fi,:,:-1]),
                                      self.lmax) for fi in self.freq_ndx])
            #groundPowerReal =  np.real(1-beamreal[:,0]/np.sqrt(4*np.pi))
            groundPowerReal = np.array([np.real(grid2healpix_alm_fast(bi.theta,bi.phi[:-1], np.real(ground[fi,:,:-1]),
                                                         1)[0])/np.sqrt(4*np.pi) for fi in self.freq_ndx])

            if i!=j:
                beamimag = np.array([grid2healpix_alm_fast(bi.theta,bi.phi[:-1], np.imag(beam[fi,:,:-1]),
                                         self.lmax) for fi in self.freq_ndx])
                groundPowerImag = np.array([np.real(grid2healpix_alm_fast(bi.theta,bi.phi[:-1], np.imag(ground[fi,:,:-1]),
                                                         1)[0]/np.sqrt(4*np.pi)) for fi in self.freq_ndx])

            else:
                beamimag = None
                groundPowerImag =0 


            self.efbeams.append((i,j,beamreal, beamimag, groundPowerReal,
                                 groundPowerImag))

                                
    def simulate (self,times=None):
        if times is None:
            times = self.obs.times
        if self.sky_model.frame=="galactic":
            do_rot = True
            lzl,bzl = self.obs.get_l_b_from_alt_az(np.pi/2,0., times)
            lyl,byl = self.obs.get_l_b_from_alt_az(0.,0., times)  ## astronomical azimuth = 0 = N = our y coordinate

        elif self.sky_model.frame=="MCMF":
            do_rot = False
        else:
            raise NotImplementedError

        wfall = []
        for ti, t in enumerate(times):
            sky = self.sky_model.get_alm (self.freq_ndx)
            if do_rot:
                lz,bz,ly,by = lzl[ti],bzl[ti],lyl[ti],byl[ti]
                zhat = np.array([np.cos(bz)*np.cos(lz), np.cos(bz)*np.sin(lz),np.sin(bz)])
                yhat = np.array([np.cos(by)*np.cos(ly), np.cos(by)*np.sin(ly),np.sin(by)])
                xhat = np.cross(yhat,zhat)
                assert(np.abs(np.dot(zhat,yhat))<1e-10)
                R = np.array([xhat,yhat,zhat]).T
                a,b,g = rot2eul(R)
                rot = hp.rotator.Rotator(rot=(g,-b,a),deg=False,eulertype='XYZ',inv=False)
                sky = [rot.rotate_alm(s_) for s_ in sky]
            res = []
            for ci,cj,beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
                T = np.array([mean_alm(br_,sky_,self.lmax) for br_,sky_ in zip(beamreal,sky)])
                T += np.array([self.Tground*gP for gP in groundPowerReal])
                if ci!=cj:
                    Timag = np.array([mean_alm(bi_,sky_,self.lmax) for bi_,sky_ in zip(beamimag,sky)])
                    Timag += np.array([self.Tground*gP for gP in groundPowerImag])
                    T = T+1j*Timag
                res.append(T)
            wfall.append(res)
        return np.array(wfall)
            
