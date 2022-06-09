
import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver
import copy

yaml_file = 'config/lbl_1m_0523.yaml'
raw_config = yaml.load(open(yaml_file),Loader=SafeLoader)


for l in [3,6]:
  for angle in [75,15]:
    config = copy.deepcopy(raw_config)
    config['beams']['default_file'] = f"hfss_lbl_{l}m_{angle}deg.fits"
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg.fits"
    S=SimDriver(config)
    S.run()
    config['sky']['type'] = 'CMB'
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_{angle}deg_CMB.fits"
    S=SimDriver(config)
    S.run()
    



