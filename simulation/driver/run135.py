
import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver
import copy

yaml_file = 'config/pdr_run.yaml'
raw_config = yaml.load(open(yaml_file),Loader=SafeLoader)


for l in [2,3]:
  for angle in [75,45,15]:
    for bangle in range(10,95,10):

      config = copy.deepcopy(raw_config)
      config['observation']['dt'] = 60
      config['beam_config']['common_beam_angle'] = bangle

      if l>0:
        config['beam_config']['default_file'] = f"hfss_lbl_{l}m_{angle}deg.fits"
        config['beam_config']['couplings']['opposite']['two_port'] = f"hfss_lbl_{l}m_{angle}deg.2port.fits"
      else:
        config['beam_config']['default_file'] = f"feko_bnl_{-l}m_{angle}deg.fits"
        config['beam_config']['couplings']['opposite']['two_port'] = f"feko_bnl_{-l}m_{angle}deg.2port.fits"

      config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg_R{bangle}.fits"
      config['simulation']['cache_transform'] = f"lunar_day_pdr_{bangle}.pickle"
      #S=SimDriver(config)
      #S.run()

    config['observation']['dt'] = 3600*24*4
    config['sky']['type'] = 'CMB'
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg_CMB.fits"
    config['simulation']['cache_transform'] = None
    #S=SimDriver(config)
    #S.run()

    config['sky']['type'] = 'DarkAges'
    config['sky']['scaled'] = True
    for nu_rms in range(1,15):
      config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg_{nu_rms}MHz_DA.fits"
      config['simulation']['cache_transform'] = None
      config['sky']['nu_rms'] = nu_rms
      S=SimDriver(config)
      S.run()
        
        
    


