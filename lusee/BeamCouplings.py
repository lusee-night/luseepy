#
# LuSEE Beam Couplings
#
# Ability to calculate beams couplings based on 1 and 2 ports beams
#
from .Beam import Beam
import numpy as np


class BeamCouplings:
    """
    Class that defines beam couplings for one and two port simulated beams

    :param Beam: Beam object
    :type Beam: class
    :param from_yaml_dict: Beam items to load from yaml dictionary
    :type from_yaml_dict: class
    """

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
            print (f"Initializing coupling '{n}'...")
            two_port_beam = Beam(sd['two_port'])
            sign = sd['sign']
            combs = sd['combinations']
            b1, b2 = combs[0]
            gain_conv = np.sqrt(self.beamd[b1].gain_conv * self.beamd[b2].gain_conv)
            dgain_conv = two_port_beam.gain_conv
            cross_power = -sign + sign*gain_conv/(2*dgain_conv)
            print (f"  cross_power: {cross_power[0]} ... {cross_power[-1]}")
            for c in combs:
                self.cross_powers[(c[0],c[1])] = cross_power
                self.cross_powers[(c[1],c[0])] = cross_power

                
    def Ex_coupling (self, b1, b2, freq_ndx):
        """
        Function that obtains E field cross coupling power for two input beams, b1 and b2

        :param b1: Beam one
        :type b1: class
        :param b2: Beam two
        :type b2: class
        :param freq_ndx: Frequency bin index at which to find cross power.
        :type freq_ndx: int

        :returns: Cross power between two input beams 
        :rtype: float
        """
        
        cross_power = self.cross_powers.get((b1.id,b2.id))
        if cross_power is None:
            cross_power = np.zeros_like(freq_ndx)
        else:
            cross_power = cross_power[freq_ndx]
        return cross_power
    
        


