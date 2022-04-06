#
# LuSEE Beam
#
import fitsio
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import sph_harm
import healpy as hp

def grid2healpix_alm(theta,phi, img, lmax):
    dphi = phi[1]-phi[0]
    dtheta = theta[1]-theta[0]
    img_flat = img.flatten()
    dA_theta = np.sin(theta)*dtheta*dphi
    #alm = np.zeros((lmax,lmax),complex)
    ell = np.arange(lmax)
    theta_list, phi_list = np.meshgrid(theta,2*np.pi-phi)
    mmax = lmax
    alm = []
    ell = []
    emm = []
    for m in range(lmax):
        for l in range(m,lmax):        
            harm = sph_harm (m,l, phi_list, theta_list) #yes idiotic convention
            assert(not np.any(np.isnan(harm)))
            alm.append((img*harm.T*dA_theta[:,None]).sum())
            ell.append(ell)
            emm.append(emm)
    alm = np.array(alm)

def grid2healpix(theta,phi, img, lmax, Nside):
    alm,_,_ = grid2healpix_alm(theta,phiu,img,lmax)
    return hp.sphtfunc.alm2map (alm,Nside)


def grid2healpix(theta,phi,img, lmax, Nside):
    dphi = phi[1]-phi[0]
    dtheta = theta[1]-theta[0]
    img_flat = img.flatten()
    dA_theta = np.sin(theta)*dtheta*dphi
    #alm = np.zeros((lmax,lmax),complex)
    ell = np.arange(lmax)
    theta_list, phi_list = np.meshgrid(theta,2*np.pi-phi)
    mmax = lmax
    alm = []
    for m in range(lmax):
        for l in range(m,lmax):        
            harm = sph_harm (m,l, phi_list, theta_list) #yes idiotic convention
            assert(not np.any(np.isnan(harm)))
            alm.append((img*harm.T*dA_theta[:,None]).sum())
    alm = np.array(alm)
    return hp.sphtfunc.alm2map (alm,Nside)



class LBeam:
    def __init__ (self, fname):
        header = fitsio.read_header(fname)
        fits = fitsio.FITS(fname,'r')
        self.E = fits[0].read() + 1j*fits[1].read()
        self.freq_start = header['freq_start']
        self.freq_end = header['freq_end']
        self.freq_step = header['freq_step']
        self.phi_start = header['phi_start']
        self.phi_end = header['phi_end']
        self.phi_step = header['phi_step']
        self.theta_start = header['theta_start']
        self.theta_end = header['theta_end']
        self.theta_step = header['theta_step']
        self.Nfreq = int((self.freq_end - self.freq_start)/self.freq_step) + 1
        self.Ntheta = int((self.theta_end - self.theta_start)/self.theta_step) + 1
        self.Nphi = int((self.phi_end - self.phi_start)/self.phi_step) + 1
        self.freq = np.linspace(self.freq_start, self.freq_end,self.Nfreq)
        self.theta_deg = np.linspace(self.theta_start, self.theta_end,self.Ntheta)
        self.phi_deg = np.linspace(self.phi_start, self.phi_end,self.Nphi)
        self.theta = self.theta_deg/180*np.pi
        self.phi = self.phi_deg/180*np.pi
        self.direction= np.array([np.sin(self.theta[None,:])*np.cos(self.phi[:,None]),
                     np.sin(self.theta[None,:])*np.sin(self.phi[:,None]),
                     np.cos(self.theta[None,:])*np.ones(self.Nphi)[:,None]]).T
    
    def rotate(self,deg):
        assert (deg in [0,45,-45,90,-90,135,-135,270,-270,180,-180])
        if deg==0:
            return self.copy()
        rad = deg/180*np.pi
        cosrad = np.cos(rad)
        sinrad = np.sin(rad)
        assert (deg%self.phi_step==0)
        m = int(deg // self.phi_step)
        print (m,self.E.shape,'X')
        if (m<0):
            E = np.concatenate ((self.E[:,:,m-1:,:],self.E[:,:,1:m,:]),axis=2)
        else:
            E = np.concatenate ((self.E[:,:,m:,:],self.E[:,:,1:m+1,:]),axis=2)
        #print (self.phi_deg,'A')
        #print (self.phi_deg[m-1:],self.phi_deg[1:m])
        #print (E.shape)
        rotmat = np.array(([[cosrad, +sinrad, 0],[-sinrad,cosrad,0],[0,0,1]]))
        E = np.einsum('fabj,ij->fabi',E,rotmat)
        return self.copy (E=E)
     
    def flip_over_yz(self):
        m = int(90 // self.phi_step)
        n = int(180 // self.phi_step)
        o = int(270 // self.phi_step)
        E = np.concatenate ((self.E[:,:,n:0:-1,:],self.E[:,:,self.Nphi:n-1:-1,:]),axis=2)
        E[:,:,:,0]*=-1 ## X flips over
        return self.copy (E=E)

    def power(self):
        """ return power in the beam """
        P = np.sum(np.abs(self.E**2),axis=3)
        return P

    def power_hp(self, ellmax, Nside, freq_ndx=None):
        """ returns healpix rendering of the power """
        P = self.power()
        take_zero = False
        
        flist = range(self.Nfreq) if freq_ndx is None else np.atleast_1d(freq_ndx)
        result =  [grid2healpix(self.theta,self.phi[:-1], P[i,:,:-1], ellmax, Nside) for i in flist]
        if type(freq_ndx)==int:
            result = result[0]
        return result
    
    def copy(self,E=None):
        ret = copy.deepcopy(self)
        if E is not None:
            ret.E = E
        return ret

    
    def plotE(self, freqndx, toplot = None, noabs=False):
        plt.figure(figsize=(15,10))
        for i in range(3):
            plt.subplot(1,3,i+1)
            ax = plt.gca()
            plt.title ('XYZ'[i])
            toshow = toplot if toplot is not None else self.E
            toshow = np.real(toshow[freqndx,:,:,i]) if noabs else np.abs(toshow[freqndx,:,:,i]) 
            im=ax.imshow(toshow,interpolation='nearest',extent=[0,360,180,0],origin='upper')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    def project_to_phi_theta(self):
        #create projection matrices
        theta = self.theta
        phi= self.phi
        sin = np.sin
        cos = np.cos
        rad = np.array([ sin(theta[:,None])*cos(phi[None,:]), sin(theta[:,None])*sin(phi[None,:]),
                         -cos(theta[:,None])*np.ones(self.Nphi)[None,:]])
        tphi =  np.array([-sin(phi), +cos(phi)])
        ttheta = np.array([ cos(theta[:,None])*cos(phi[None,:]), cos(theta[:,None])*sin(phi[None,:]),
                         +sin(theta[:,None])*np.ones(self.Nphi)[None,:]])

        Erad = np.einsum('fijk,kij->fij',self.E,rad)
        Etheta = np.einsum('fijk,kij->fij',self.E,ttheta)
        Ephi = np.einsum('fijk,kj->fij',self.E[:,:,:,:2],tphi)
        Emag2 = (np.abs(self.E)**2).sum(axis=3)
        assert(abs(Emag2-np.abs(Erad)**2-np.abs(Etheta)**2-np.abs(Ephi)**2).max()<1e-4)
        #print ((np.abs(Erad)/np.sqrt(Emag2)).max())
        assert(np.all(np.abs(Erad)/np.sqrt(Emag2)<1e-4))
        Eout = np.array([Ephi,Etheta])
        Eout = np.moveaxis(Eout,0,-1)
        return Eout
