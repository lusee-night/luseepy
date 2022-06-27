
import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver
import copy

yaml_file = 'config/pdr_run.yaml'
raw_config = yaml.load(open(yaml_file),Loader=SafeLoader)


for l in [2,3]:
  for angle in [75,15]:
    config = copy.deepcopy(raw_config)
    config['beam_config']['default_file'] = f"hfss_lbl_{l}m_{angle}deg.fits"
    config['beam_config']['couplings']['opposite']['two_port'] = f"hfss_lbl_{l}m_{angle}deg.2port.fits"
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg.fits"
    S=SimDriver(config)
    S.run()
    config['sky']['type'] = 'CMB'
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg_CMB.fits"
    S=SimDriver(config)
    S.run()
    



