#
# LuSEE Beam Couplings
#
# Ability to calculate beams couplings based on 1 and 2 ports beams
#
from .LBeam import LBeam
import numpy as np


class LBeamCouplings:

    def __init__(self, beams = [], from_yaml_dict = None):
        self.beamd = {}
        for b in beams:
            self.beamd[b.id] = b
        self.cross_powers = {}
        if from_yaml_dict is not None:
            self._yaml_init(from_yaml_dict)

            
    def _yaml_init(self, d):
        self.cross_powers = {}
        for n,sd in d.items():
            print (f"Initializing copupling '{n}'...")
            two_port_beam = LBeam(sd['two_port'])
            sign = sd['sign']
            combs = sd['combinations']
            b1, b2 = combs[0]
            gain_conv = np.sqrt(self.beamd[b1].gain_conv * self.beamd[b1].gain_conv)
            dgain_conv = two_port_beam.gain_conv
            cross_power = -sign + sign*gain_conv/(2*dgain_conv)
            print (f"  cross_power: {cross_power[0]} ... {cross_power[-1]}")
            for c in combs:
                self.cross_powers[(c[0],c[1])] = cross_power
                self.cross_powers[(c[1],c[0])] = cross_power

                
    def Ex_coupling (self, b1, b2, freq_ndx):
        cross_power = self.cross_powers.get((b1.id,b2.id))
        if cross_power is None:
            cross_power = np.zeros_like(freq_ndx)
        else:
            cross_power = cross_power[freq_ndx]
        return cross_power
    
        


