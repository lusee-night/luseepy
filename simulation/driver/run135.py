
import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver
import copy

yaml_file = 'config/pdr_run.yaml'
raw_config = yaml.load(open(yaml_file),Loader=SafeLoader)


for l in [3]:
  for angle in [75]:
    config = copy.deepcopy(raw_config)
    config['observation']['dt'] = 60
    config['beam_config']['default_file'] = f"hfss_lbl_{l}m_{angle}deg.fits"
    config['beam_config']['couplings']['opposite']['two_port'] = f"hfss_lbl_{l}m_{angle}deg.2port.fits"
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg.fits"
    config['simulation']['cache_transform'] = "lunar_day_pdr.pickle"

    S=SimDriver(config)
    S.run()
    config['observation']['dt'] = 3600*24*4
    config['sky']['type'] = 'CMB'
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg_CMB.fits"
    config['simulation']['cache_transform'] = None
    S=SimDriver(config)
    S.run()
    



