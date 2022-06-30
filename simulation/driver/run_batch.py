import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver
import sys

if len(sys.argv)!=4:
  print ("Needs to be supplied with 3 parameters: yaml_config_file options_mod_file line_num")
  sys.exit(0)

yaml_file = sys.argv[1]
config = yaml.load(open(yaml_file),Loader=SafeLoader)
line = int(sys.argv[3])
mod = open(sys.argv[2]).readlines()[line]

for entry in mod.split(" "):
  print (f"Setting: {entry}")
  keys, value = entry.split('=')
  try:
    value = int(value)
  except:
    try:
      value = float(value)
    except:
      pass
  keys = keys.split(':')
  if len(keys)==1:
    d[keys[0]] = value
  else:
    d = config[keys[0]]
    for k in keys[1:-1]:
      d=d[k]
    d[keys[-1]]=value

S=SimDriver(config)
S.run()
    
