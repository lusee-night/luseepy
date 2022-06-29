#!/usr/bin/env python
import lusee
import numpy  as np
import healpy as hp
import pickle
import os,sys
import yaml
from yaml.loader import SafeLoader

class SimDriver(dict):
    def __init__ (self,yaml):
        self.update(yaml)
        self.lmax = self['observation']['lmax'] ## common lmax
        self.root = self['paths']['lusee_drive_dir']
        if self.root[0]=='$':
            self.root = os.environ[self.root[1:]]
        self._parse_sky()
        self._parse_beams()
        

    def _parse_sky(self):
        sky_type = self['sky'].get('type','file')
        if sky_type == 'file':
            fname = os.path.join(self.root,self['paths']['sky_dir'],self['sky']['file'])
            print ("Loading sky: ",fname)
            self.sky = lusee.sky.FitsSky (fname, lmax = self.lmax)
        elif sky_type == 'CMB':
            # make sure if lmax matters here
            print ("Using CMB sky")
            self.sky = lusee.sky.ConstSky(self.lmax,lmax=self.lmax,T=2.73, freq=np.arange(1,51)) 
        elif sky_type == 'Cane1979':
            # make sure if lmax matters here
            print ("Using Cane1979 sky")
            self.sky = lusee.sky.ConstSkyCane1979(self.lmax, lmax=self.lmax, freq=np.arange(1,51))  

    def _parse_beams(self):
        broot = os.path.join(self.root,self['paths']['beam_dir'])
        beams = []
        bd = self['beams']
        bdc = self['beam_config']
        couplings = bdc.get('couplings')
        beam_type = bdc.get('type','fits')
        self.beam_smooth = bdc.get('beam_smooth')
        
        if beam_type=='Gaussian': #similar to sky_type above
            print('Creating Gaussian beams!')
            for b in self['observation']['beams']:
                cbeam=bd[b]
                print ("Creating gaussian beam",b,":")
                B = lusee.LBeam_Gauss(dec_deg=cbeam['declination'],
                                      sigma_deg=cbeam['sigma'],
                                      one_over_freq_scaling=cbeam['one_over_freq_scaling'], id = b)
                angle = bdc['common_beam_angle']+cbeam['angle']
                print ("  rotating: ",angle)
                B = B.rotate(angle)
                beams.append(B)
        elif beam_type == 'fits':
            for b in self['observation']['beams']:
                print ("Loading beam",b,":")
                cbeam = bd[b]
                filename = cbeam.get('file')
                if filename is None:
                    default_file = bdc.get('default_file')
                    filename = default_file
                    if filename is None:
                        print ("Neither default not special file declare for beam",b)
                fname = os.path.join(broot,filename)
                print ("  loading file: ",fname)
                B = lusee.LBeam (fname, id = b)
                angle = bdc['common_beam_angle']+cbeam.get('angle',0)
                print ("  rotating: ",angle)
                B=B.rotate(angle)
                beams.append(B)
        else:
            print ("Beam type unrecognized")
            raise Exception('NotImplementedError')
        
        self.beams = beams
        self.Nbeams = len(self.beams)
        if couplings is not None:
            for c in couplings:
                couplings[c]['two_port'] = os.path.join(broot,couplings[c]['two_port'])
            self.couplings=lusee.LBeamCouplings(beams, from_yaml_dict = couplings)
        else:
            self.couplings = None

    def run(self):
        print ("Starting simulation :")
        od = self['observation']
        dt = od['dt']
        if type(dt)==str:
            dt = eval(dt)
        O=lusee.LObservation(od['lunar_day'],deltaT_sec=dt,
                    lun_lat_deg=od['lat'], lun_long_deg = od['long'])
        freq = np.arange(od['freq']['start'],od['freq']['end'],od['freq']['step'])
        print ("  setting up combinations...")
        combs = od['combinations']
        if type(combs)==str:
            if combs=='all':
                combs = []
                for i in range(self.Nbeams):
                    for j in range(i,self.Nbeams):
                        combs.append((i,j))
    
        print ("  setting up Simulation object...")
        S = lusee.Simulator (O,self.beams, self.sky, freq=freq, lmax = self.lmax,
                             combinations=combs, Tground = od['Tground'],
                             cross_power = self.couplings, beam_smooth = self.beam_smooth,
                             extra_opts = self['simulation'] )
        print ("  Simulating...")
        S.simulate(times=O.times)
        fname = self['simulation']['output']
        print ("Writing to",fname)
        S.write(fname)



if __name__ == "__main__":
    if len(sys.argv)<2:
        print ("Specify yaml config file command line parameter.")
        sys.exit(0)
    yaml_file = sys.argv[1]
    with open(yaml_file) as f:
        config = yaml.load(f,Loader=SafeLoader)
    S=SimDriver(config)
    S.run()
    
