#!/usr/bin/env python3

if __name__ == "__main__":
    import  lusee
    import  numpy  as np
    import  healpy as hp
    import  pickle
    import  os,sys
    import  yaml
    from    yaml.loader import SafeLoader

#######################
class SimDriver(dict):
    def __init__ (self,yaml):
        self.update(yaml)
        self._parse_base()
        self._parse_sky()
        self._parse_beams()
        
    def _parse_base(self):
        self.lmax = self['observation']['lmax'] ## common lmax
        self.root = self['paths']['lusee_drive_dir']
        self.outdir = self['paths']['output_dir']

        if self.root[0]=='$':
            self.root = os.environ[self.root[1:]]
        if self.outdir[0]=='$':
            self.outdir = os.environ[self.outdir[1:]]

        od = self['observation']
        self.dt = od['dt']
        if type(self.dt)==str:
            self.dt = eval(od['dt'])
        self.freq = np.arange(od['freq']['start'],od['freq']['end'],od['freq']['step'])

    def _parse_sky(self):
        sky_type = self['sky'].get('type','file')
        if sky_type == 'file':
            fname = os.path.join(self.root,self['paths']['sky_dir'],self['sky']['file'])
            print ("Loading sky: ",fname)
            self.sky = lusee.sky.FitsSky (fname, lmax = self.lmax)
        elif sky_type == 'CMB':
            # make sure if lmax matters here
            print ("Using CMB sky")
            self.sky = lusee.sky.ConstSky(self.lmax,lmax=self.lmax,T=2.73, freq=self.freq) 
        elif sky_type == 'Cane1979':
            # make sure if lmax matters here
            print ("Using Cane1979 sky")
            self.sky = lusee.sky.ConstSkyCane1979(self.lmax, lmax=self.lmax, freq=self.freq)  
        elif sky_type == 'DarkAges':
            d = self['sky']
            scaled = d.get('scaled',True)
            nu_min = d.get('nu_min',16.4) 
            nu_rms = d.get('nu_rms',14.0)
            A      = d.get('A',0.04)
            
            print (f"Using Dark Ages Monopole sky scaled={scaled}, min={nu_min} MHz, rms={nu_rms}MHz,A={A}K")
            self.sky = lusee.sky.DarkAgesMonopole(self.lmax, lmax=self.lmax, freq = self.freq,
                                                  nu_min=nu_min, nu_rms=nu_rms, A=A)
            
    def _parse_beams(self):
        broot = os.path.join(self.root,self['paths']['beam_dir'])
        beams = []
        bd = self['beams']
        bdc = self['beam_config']
        couplings = bdc.get('couplings')
        beam_type = bdc.get('type','fits')
        beam_smooth = bdc.get('beam_smooth')
        taper = bdc.get('taper', self.get('simulation', {}).get('taper', 0.03))
        
        if beam_type=='Gaussian': #similar to sky_type above
            print('Creating Gaussian beams!')
            for b in self['observation']['beams']:
                cbeam=bd[b]
                print ("Creating gaussian beam",b,":")
                B = lusee.BeamGauss(dec_deg=cbeam['declination'],
                                      sigma_deg=cbeam['sigma'],
                                      one_over_freq_scaling=cbeam['one_over_freq_scaling'], id = b)
                angle = bdc['common_beam_angle']+cbeam['angle']
                print ("  rotating: ",angle)
                B = B.rotate(angle)
                B.taper_and_smooth(taper=taper, beam_smooth=beam_smooth)
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

                B = lusee.Beam(fname, id = b)

                angle = bdc['common_beam_angle']+cbeam.get('angle',0)
                print ("  rotating: ",angle)
                B=B.rotate(angle)
                B.taper_and_smooth(taper=taper, beam_smooth=beam_smooth)
                beams.append(B)
        else:
            print ("Beam type unrecognized")
            raise Exception('NotImplementedError')
        
        self.beams = beams
        self.Nbeams = len(self.beams)
        if couplings is not None:
            for c in couplings:
                couplings[c]['two_port'] = os.path.join(broot,couplings[c]['two_port'])
            self.couplings=lusee.BeamCouplings(beams, from_yaml_dict = couplings)
        else:
            self.couplings = None

    def run(self):
        print("Starting simulation:")
        od = self["observation"]
        # Location and time come only from the observation object
        O = lusee.Observation(
            od["lunar_day"],
            deltaT_sec=self.dt,
            lun_lat_deg=od["lat"],
            lun_long_deg=od["long"],
        )
        print(f"  Using observation: lat={O.lun_lat_deg} deg, lon={O.lun_long_deg} deg, time_range={O.time_range}, N_times={len(O.times)}")
        print("  setting up combinations...")
        combs = od["combinations"]
        if type(combs) == str:
            if combs == "all":
                combs = []
                for i in range(self.Nbeams):
                    for j in range(i, self.Nbeams):
                        combs.append((i, j))

        engine = self["simulation"].get("engine")
        engine = str(engine).strip().lower()
        if engine == "croissant":
            if lusee.CroSimulator is None:
                raise RuntimeError(
                    "CroSimulator requires optional dependency 'croissant' (and s2fft). "
                    "Install with: pip install croissant s2fft"
                )
            print("  setting up Croissant Simulation object...")
            S = lusee.CroSimulator(
                O,
                self.beams,
                self.sky,
                Tground=od["Tground"],
                combinations=combs,
                freq=self.freq,
                lmax=self.lmax,
                cross_power=self.couplings,
                extra_opts=self["simulation"],
            )
        elif engine == "default":
            print("  setting up Default Simulation object...")
            S = lusee.DefaultSimulator(
                O,
                self.beams,
                self.sky,
                Tground=od["Tground"],
                combinations=combs,
                freq=self.freq,
                lmax=self.lmax,
                cross_power=self.couplings,
                extra_opts=self["simulation"],
            )
        else:
            raise ValueError(f"simulation.engine must be 'Default' or 'Croissant', got: {engine}")

        print(f"  Simulating {len(O.times)} timesteps (from observation) x {len(combs)} data products x {len(self.freq)} frequency bins...")
        print("  Simulating...")
        S.simulate(times=O.times)

        out_base = self["simulation"].get("output", f"sim_{engine.capitalize()}_output.fits")
        fname = os.path.join(self.outdir, out_base)

        print ("Writing to",fname)
        S.write_fits(fname)



if __name__ == "__main__":
    if len(sys.argv)<2:
        print ("Specify yaml config file command line parameter.")
        sys.exit(0)
    yaml_file = sys.argv[1]
    with open(yaml_file) as f:
        config = yaml.load(f,Loader=SafeLoader)
    S=SimDriver(config)
    S.run()
    
