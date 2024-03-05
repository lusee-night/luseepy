#!/usr/bin/env python
import argparse
import numpy as np
import sys, glob
from converter_base import BeamConverter


try:
    import lusee
    have_lusee = True
except:
    have_lusee = False
    
class HFSS2LBeam(BeamConverter):

    def __init__ (self):
        BeamConverter.__init__(self, 'Convert HFSS Beam to LBEAM.', 'hfss_converted.fits')

    def add_options(self):
        pass

    def process_options(self,args):
        pass
        
    def load(self):
        Edir = self.root+"/ElectricField/"
        Efiles = glob.glob(Edir+'/*.csv')
        if (len(Efiles)==0):
            print (f"Did not find any files in {Edir}. Giving up!")
            sys.exit(1)
        Gdir = self.root+"/Gain/"
        Gfiles = glob.glob(Gdir+'/*.csv')
        if (len(Gfiles)==0):
            print (f"Did not find any files in {Gdir}. Giving up!")
            sys.exit(1)
        freq = []
        freqfname = {}
        def get_freq(fname):
            have = False
            for field in fname.split('_'):
                if "MHz" in field:
                    cfreq=float(field.replace('MHz',''))
                    have = True
                if "kHz" in field:
                    cfreq=float(field.replace('kHz',''))/1000
                    have = True
            
            if have and (cfreq>self.freqmax or cfreq<self.freqmin):
                have = False
            return have, cfreq

        for fname in Efiles:
            have, cfreq = get_freq(fname)
            if have:
                if cfreq in freqfname:
                    print ("We seem to have two files with the same frequency! ")
                    print(freqfname[cfreq])
                    print(fname)
                    print ("Exiting")
                    sys.exit(1)
                freqfname[cfreq]=fname
                freq.append(cfreq)

        freqgain = {}
        for fname in Gfiles:
            have, cfreq = get_freq(fname)
            if have:
                assert (cfreq in freq)
            if cfreq in freqgain:
                print ("We seem to have two files with the same frequency! ")
                print(freqgain[cfreq])
                print(fname)
                print ("Exiting")
                sys.exit(1)

            freqgain[cfreq]=fname

        

        freq = np.array(sorted(freq))
        Nfreq = len(freq)
        
        freq_min, freq_max = freq[0], freq[-1]
        print ("Loading frequencies: ", end = "")
        have_size = False
        for i,cfreq in enumerate(freq):
            print (f"{cfreq} ... ", end = "")
            sys.stdout.flush()
            #Efield
            #print (f"REading freq,{freqfname[cfreq]}, {freqgain[cfreq]}")
            data = open(freqfname[cfreq]).readlines()[1:]
            data = np.array([[float(x) for x in d.split(',')] for d in data])
            phi, theta, ExR, ExI, EyR, EyI, EzR, EzI = data.T
            #gain
            data = open(freqgain[cfreq]).readlines()[1:]
            data = np.array([[float(x) for x in d.split(',')] for d in data])
            gphi, gtheta, gain = data.T
            assert(np.all(gphi==phi))
            assert(np.all(gtheta==theta))
            
            E = np.array([ExR+1j*ExI,  EyR+1j*EyI,  EzR+1j*EzI])
            sin = np.sin
            cos = np.cos
            thetarad = theta/180*np.pi
            phirad = phi/180*np.pi
            trad = np.array([ sin(thetarad)*cos(phirad), sin(thetarad)*sin(phirad),
                             +cos(thetarad)])
            tphi =  np.array([-sin(phirad), +cos(phirad),np.zeros_like(thetarad)])
            ttheta = np.array([ cos(thetarad)*cos(phirad), cos(thetarad)*sin(phirad),
                     -sin(thetarad)])

            Erad = (trad*E).sum(axis=0)
            Ephi = (tphi*E).sum(axis=0)
            Etheta = (ttheta*E).sum(axis=0)
            assert(np.abs((tphi*ttheta).sum(axis=0)).max()<1e-10)
            assert(np.abs((tphi*trad).sum(axis=0)).max()<1e-10)
            assert(np.abs((ttheta*trad).sum(axis=0)).max()<1e-10)
            assert(np.abs(((tphi*tphi).sum(axis=0))-1).max()<1e-10)
            assert(np.abs(((ttheta*ttheta).sum(axis=0))-1).max()<1e-10)
            assert(np.abs(((trad*trad).sum(axis=0))-1).max()<1e-10)

            E = np.sqrt(np.abs(Ephi**2)+np.abs(Erad**2)+np.abs(Etheta**2))+1e-90
            check = np.where((np.abs(Erad)/E>0.01) & (E>E.max()/10))
            if len(check[0]>0):
                print (" [ Warning, Erad exceeds 1% E at places! ]")
            check = np.where((np.abs(Erad)/E>0.1) & (E>E.max()/10))
            if len(check[0]>0):
                print (" [ Warning, Erad exceeds 10% E at places! Stopping! ]",end="")
                sys.exit(1)
            if not (have_size):
                thetag= sorted(list(set(theta)))
                phig = sorted(list(set(phi)))
                theta_min, theta_max, Ntheta = thetag[0], thetag[-1], len(thetag)
                phi_min, phi_max, Nphi = phig[0], phig[-1], len(phig)
                dtheta = (theta_max - theta_min)/(Ntheta-1)
                dphi = (phi_max - phi_min)/(Nphi-1)
                gEtheta = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
                gEphi = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
                ggain = np.zeros((Nfreq,Ntheta,Nphi),float)+np.nan
                
                thetaL = ((theta-theta_min)/dtheta+1e-6).astype(int)
                phiL = ((phi-phi_min)/dphi+1e-6).astype(int)
                gEtheta[i,thetaL, phiL] = Etheta
                gEphi[i,thetaL, phiL] = Ephi
                ggain[i,thetaL, phiL] = gain
                have_size = True
            else:
                gEtheta[i,thetaL, phiL] = Etheta
                gEphi[i,thetaL, phiL] = Ephi
                ggain[i,thetaL, phiL] = gain
                
        Etheta = gEtheta
        Ephi = gEphi
        gain = ggain
        print ("\n")
        print ("Data loaded:")
        print (f"Freq: {freq_min} ... {freq_max} MHz  ({Nfreq} bins)")
        print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        print (f"Phi: {phi_min}, {phi_min+dphi} ... {phi_max} deg ({Nphi} bins)")

        newNtheta = int((self.thetamax-theta_min)/dtheta)+1
        newtheta_max = theta_min+dtheta*(newNtheta-1)
        if newNtheta<Ntheta:
            print ("Applying theta cut...")
            Etheta = Etheta [:, :newNtheta, :]
            Ephi = Ephi [:, :newNtheta, :]
            gain = gain [:, :newNtheta, :]
            Ntheta = newNtheta
            theta_max = newtheta_max
            print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        
        ## now assert all nans are gone
        print (np.where(np.isnan(Etheta)))
        assert(not np.any(np.isnan(Etheta)))
        assert(not np.any(np.isnan(Ephi)))
        print ("Beam loading successful.")

        #f_ground = np.array([f_ground[f] for f in freq])
        #print("f_ground = ", f_ground)
        #print ("f_ground parsing successful.")
        print ("Finding gain conversion factors")
        mygain = np.abs(Etheta**2) + np.abs(Ephi**2)
        #db2fact = lambda dB: 10**(dB/10)
        # in hfss we have directity, which is already in gain units
        ratio = gain/mygain
        gainmax = gain.max(axis=(1,2))
        gainmin = gain.min(axis=(1,2))
        gainconv = []
        for i,f in enumerate(freq):
            r = ratio[i,:,:]
            w = np.where(gain[i,:,:]>gainmax[i]/100)
            meanconv = r[w].mean()
            rms = np.sqrt(r[w].var())
            print (f"    {f} MHz    {meanconv:0.3g} ({rms/meanconv*100:0.3f}% err) (gain {gainmin[i]:0.3g}...{gainmax[i]:0.3g}))")
            #assert (rms/meanconv<1e-2)
            gainconv.append(meanconv)
        
        flist = glob.glob(self.root+"/Impedance/*.csv")
        if (len(flist)>1 and not self.split_impedance):
            print ("Don't know which impedance file to use. Candidates:")
            for f in flist:
                print ("       ",f)
            sys.exit(1)
        elif (len(flist)==0):
            print ("Can't find impedance info.")
            sys.exit(1)
        
        def verbose_read (fname):
            print ("  Reading impedance file",fname)
            data= open(fname).readlines()[1:]
            return data
        data = sum ([verbose_read(fname) for fname in flist],[])
        data = np.array(sorted([[float(x) for x in d.split(',')] for d in data]))
        ## now check that we don't have duplicates
        while True:
            exit = True
            for i in range(len(data)-1):
                if data[i,0]==data[i+1,0]:
                    data[i,:] = (data[i,:]+data[i+1,:])/2
                    data = np.delete(data,i+1,axis=0)
                    exit = False
                    break
            if exit:
                break


        cfreq, ZRe, ZIm = data.T


        print ("\n\n")
        if len(cfreq)!=len(freq) or (not np.all(cfreq==freq)):
            print ("Error: Frequencies in impedance file do not match those in the beam file!")
            print (f'Beam file: {freq[0]}..{freq[-1]} in {len(freq)} bins.')
            print (f'Impedance file: {cfreq[0]}..{cfreq[-1]} in {len(cfreq)} bins.')
            print (cfreq)
            stop()
        print ("Impedance loading successful.")

        
        self.Etheta = Etheta
        self.Ephi = Ephi
        self.ZRe = ZRe
        self.ZIm = ZIm
        self.freq_min, self.freq_max, self.Nfreq = freq_min, freq_max, Nfreq
        self.theta_min, self.theta_max, self.Ntheta = theta_min, theta_max, Ntheta
        self.phi_min, self.phi_max, self.Nphi = phi_min, phi_max, Nphi
        self.gainconv = np.array(gainconv)
        self.gain = gain
        self.freq = freq



if __name__=="__main__":
    H2B = HFSS2LBeam()
    print (f"  HFSS beam converter  ")
    print (f"-----------------------")
    print (f" Loading: {H2B.root}\n")
    H2B.load()
    H2B.save_fits()
    if have_lusee:
        print ("Attempting to reread the file ... ",end="")
        sys.stdout.flush()
        B = lusee.Beam(H2B.output_file)
        print ("OK.")
    else:
        print ("No lusee module so no check.")
