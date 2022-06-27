
import yaml
from yaml.loader import SafeLoader
from  run_sim import SimDriver
import copy
import lusee

yaml_file = 'config/pdr_kaja_run.yaml'
raw_config = yaml.load(open(yaml_file),Loader=SafeLoader)


for l in [2,3]:
  for angle in [75,15]:
    config = copy.deepcopy(raw_config)
    config['beams']['P']['file'] = f"hfss_lbl_{l}m_{angle}deg.2port.fits"
    outname = f"output/kaja_{l}m_{angle}deg.fits"
    config['simulation']['output'] = outname
    S=SimDriver(config)
    S.run()
    print ("Reading output...")
    D = lusee.LData(outname)
    f=open(outname.replace('.fits','.txt'),'w')
    freq = S.sky.freq
    TC = lusee.sky_models.T_C(freq).value
    Tsky = D[0,'00R',:]
    Vsky = D[0,'00RV',:]
    Gamma_VD = D.Gamma_VD[0]
    ZRe = D.ZRe[0]
    ZIm = D.ZIm[0]
    T2VSq = D.T2Vsq[0]
    TDA = lusee.mono_sky_models.T_DarkAges(freq)
    TDA_ant = Tsky/TC*TDA
    f.write("# f TCane TSky Vsky Gamma_VD ZRe ZIm T2VSq T_DA T_DA_ant\n")
    for line in zip (freq,TC,Tsky, Vsky, Gamma_VD, ZRe, ZIm, T2VSq, TDA, TDA_ant):
      f.write (" %f %f %f %g %f %f %f %g %f %f\n"%line)
    f.close()
    



