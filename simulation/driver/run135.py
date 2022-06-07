
import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver

yaml_file = 'config/lbl_1m_0523.yaml'
config = yaml.load(open(yaml_file),Loader=SafeLoader)

for l in [1,3,6]:
    config['beams']['default_file'] = f"hfss_lbl_{l}m_75deg.fits"
    config['simulation']['output'] = f"output/hfss_lbl_{l}m_0523.fits"
    S=SimDriver(config)
    S.run()



