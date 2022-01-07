import sys
sys.path.append('.')
import lusee
O = lusee.LObservation()
S = lusee.LSatellite()
S.predict_position_mcmf(O.times)
