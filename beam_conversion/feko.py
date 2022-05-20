#!/usr/bin/env python
import argparse
import numpy as np
import sys
from converter_base import BeamConverter


try:
    import lusee
    have_lusee = True
except:
    have_lusee = False


def loadSData(data_file, z_load):     
    n_ports = len(z_load)
      
    with open(data_file) as file:
        #Skip header
        lines = file.readlines()[6:]
        n_lines = len(lines)
        line_len = len(np.fromstring(lines[1], dtype=float, sep=' '))
        data = np.zeros((n_lines, line_len))
        #Remove extra leading column from every forth line (frequency)
        for n, line in enumerate(lines):
            if n%n_ports == 0:
                data[n,:] += np.fromstring(line, dtype=float, sep=' ')[1:]
            else:
                data[n,:] += np.fromstring(line, dtype=float, sep=' ')
    
    n_freqs = np.size(data,0)//n_ports
    
    #Convert magnitude and angle to Re+Im
    complex_array = np.empty((n_lines, n_ports), dtype=np.complex128)
    for col in range(n_ports):
        complex_array[:, col] = data[:, 2*col]*np.exp(1.0j*data[:, (2*col)+1]*np.pi/180.0)
    
    #Put into S matrix for each freq
    s_data_array = np.empty((n_freqs, n_ports, n_ports), dtype=np.complex128)
    for i in range(n_freqs):
        s_data_array[i, :, :] = complex_array[i*n_ports:(i+1)*n_ports,:] 
            
    return s_data_array

def convertS2Z(s_data_array, z_load):
    n_ports = len(z_load)
    n_freqs = np.size(s_data_array,0)
    identity = np.identity(n_ports)
    sqrt_z_load = np.matmul(identity, np.sqrt(z_load).T)
    
    z_data_array =  np.empty((n_freqs, n_ports, n_ports), dtype=np.complex128)
    for freq in range(n_freqs):
        s = s_data_array[freq, :, :]
        # z = sqrt(z_load) * (1+s) * (1-s)^-1 *  sqrt(z)
        z = np.matmul(np.matmul( np.matmul(sqrt_z_load, (identity + s)), np.linalg.inv(identity - s)), sqrt_z_load)
        z_data_array[freq, :, :] = z
    
    return z_data_array


    
class Feko2LBeam(BeamConverter):

    def __init__ (self, root, farfield, thetamax = 90):
        BeamConverter.__init__(self,root,thetamax)
        self.farfield = farfield

        
    def load(self):
        fname = self.find_single_file("*.out")
        data = open(fname).readlines()
        skip = True
        beam_data = []
        freq = -1.0
        farfields = []
        f_ground = {}  # store a map for freq → f_ground
        reading_power_radiated = False  # toggle whether we look for radiated power
        farfield = ""
        freq_list = []
        print ("Loading frequencies: ", end = "")
        for line in data:
            if skip:
                ## ignoring radiated power for a while
                # if we find a header with radiated power, start looking
                #if "The directivity/gain is based on an active power of" in line:
                #    reading_power_radiated = True
                #    radiated_power = float(line.split()[-2]) #normalization
                #    skip = False

                if ("   THETA    PHI      magn.    phase  " in line) and (farfield == self.farfield):
                    skip = False
                    freq_list.append(freq)

                if "FREQ =" in line:
                    cfreq=float(line.split()[-1])/1e6
                    if cfreq!=freq:
                        print (f"{cfreq} ... ", end = "")
                        sys.stdout.flush()
                    freq=cfreq
                if "Far field request with name:" in line:
                    farfield = line[:-1].split()[-1]
                    if farfield not in farfields:
                        farfields.append(farfield)
                    
            else:
                if reading_power_radiated:
                    if "0.00 .. 180.00 deg.       0.00 .. 360.00 deg.   " in line:
                        reading_power_radiated = False
                        skip = True
                        f_ground[freq] = 1 - float(line.split()[-2])/radiated_power  # f_ground = 1 - f_sky
                else:
                    if line == "\n":
                        skip = True
                    else:
                        line  = line.split()
                        if len(line)==12:
                            line = [float(x) for x in line[:9]]
                            if line[0]<0:
                                print (line,freq)
                            beam_data.append([freq]+line)
        print()
        print ("Farfields seen:",farfields)
        beam = np.array(beam_data)
        print (f"{beam.shape[0]} rows loaded.")
        plist = []
        for i in range(3):
            plist.append(sorted(list(set(beam[:,i]))))
        freq, theta, phi = plist
        #freq_min, freq_max, Nfreq = freq[0], freq[-1], len(freq)
        assert(set(freq)==set(freq_list))
        freq = np.array(freq_list)
        theta_min, theta_max, Ntheta = theta[0], theta[-1], len(theta)
        phi_min, phi_max, Nphi = phi[0], phi[-1], len(phi)
        #dfreq = (freq_max - freq_min)/(Nfreq-1)
        dtheta = (theta_max - theta_min)/(Ntheta-1)
        dphi = (phi_max - phi_min)/(Nphi-1)
        freq_min = freq.min()
        freq_max = freq.max()
        Nfreq = len(freq)
        #np.testing.assert_almost_equal(freq[1]-freq[0], dfreq)
        np.testing.assert_almost_equal(theta[1]-theta[0], dtheta)
        np.testing.assert_almost_equal(phi[1]-phi[0], dphi)
        print ("Data loaded:")
        print (f"Freq: {freq_min} ... {freq_max} MHz  ({Nfreq} bins)")
        print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        print (f"Phi: {phi_min}, {phi_min+dphi} ... {phi_max} deg ({Nphi} bins)")
        Etheta = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
        Ephi = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
        gain = np.zeros((Nfreq,Ntheta,Nphi),float)+np.nan
        freqL = np.zeros(beam.shape[0],int)-1
        for i,f in enumerate(freq):
            freqL[f==beam[:,0]] = i
        assert (np.all(freqL>=0))
        thetaL = ((beam[:,1]-theta_min)/dtheta+1e-6).astype(int)
        phiL = ((beam[:,2]-phi_min)/dphi+1e-6).astype(int)
        EthetaL = beam[:,3]*np.exp(1j*2*np.pi/360*beam[:,4])
        EphiL = beam[:,5]*np.exp(1j*2*np.pi/360*beam[:,6])

        print (freqL[-10:],thetaL[-10:], phiL[-10:], EthetaL[-10:])
        Etheta[freqL, thetaL, phiL] = EthetaL
        Ephi[freqL, thetaL, phiL] = EphiL
        gain[freqL, thetaL, phiL] = beam[:,9]

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
        db2fact = lambda dB: 10**(dB/10)
        ratio = db2fact(gain)/(mygain+1e-100)
        gainmax = gain.max()
        gainconv = []
        for i,f in enumerate(freq):
            r = ratio[i,:,:]
            w = np.where(gain[i,:,:]>gainmax-20)
            meanconv = r[w].mean()
            rms = np.sqrt(r[w].var())
            assert (rms/meanconv<1e-3)
            print (f"    {f} MHz    {meanconv:0.3f} ({rms/meanconv*100:0.3f}% err)")
            gainconv.append(meanconv)
        
        ZRe = np.zeros(Nfreq)+np.nan
        ZIm =np.zeros(Nfreq)+np.nan

        data = np.loadtxt(self.find_single_file("*_Z_Re.dat"), skiprows=2)

        freqL = np.zeros(data.shape[0],int)-1
        for i,f in enumerate(freq):
            freqL[f==data[:,0]/1e6] = i
        ZRe [freqL] = data[:,1]

        data = np.loadtxt(self.find_single_file("*_Z_Im.dat"), skiprows=2)
        ZIm [freqL] = data[:,1]

        assert(not np.any(np.isnan(ZRe)))
        assert(not np.any(np.isnan(ZIm)))
        print ("Impedance loading successful.")


        
        self.Etheta = Etheta
        self.Ephi = Ephi
        self.ZRe = ZRe
        self.ZIm = ZIm
        self.freq_min, self.freq_max, self.Nfreq = freq_min, freq_max, Nfreq
        self.theta_min, self.theta_max, self.Ntheta = theta_min, theta_max, Ntheta
        self.phi_min, self.phi_max, self.Nphi = phi_min, phi_max, Nphi
        self.gainconv = np.array(gainconv)
        self.freq = freq


def parse_args():
    parser = argparse.ArgumentParser(description='Convert FEKO Beam to LBEAM.')
    parser.add_argument('root_name', nargs=1, help='root name, ')
    parser.add_argument('--farfield', default = "FarField1", help='farfield to pick')
    parser.add_argument('--thetamax', default = 90, type=float, help='do not include data beyond this theta')
    parser.add_argument('-o', '--output_file', default = "feko_converted.fits", help='output filename')
    args = parser.parse_args()
    O = Feko2LBeam(args.root_name[0],farfield = args.farfield, thetamax = args.thetamax)
    return O, args


if __name__=="__main__":
    F2B, args = parse_args()
    F2B.load()
    F2B.save_fits(args.output_file)
    if have_lusee:
        print ("Attempting to reread the file ... ",end="")
        sys.stdout.flush()
        B = lusee.LBeam(args.output_file)
        print ("OK.")
    else:
        print ("No lusee module so no check.")
