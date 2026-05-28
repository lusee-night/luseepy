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

                
    def Ex_coupling (self, b1, b2, freq_map):
        """
        Function that obtains E field cross coupling power for two input beams, b1 and b2

        :param b1: Beam one
        :type b1: class
        :param b2: Beam two
        :type b2: class
        :param freq_map: Interpolation map (see :func:`lusee.frequencies.interpolation_weights`)
            from the simulator target frequencies to the beams' native frequency grid.
        :type freq_map: lusee.frequencies.FrequencyMap

        :returns: Cross power between two input beams at the target frequencies.
        :rtype: numpy array of shape ``(N_target,)``
        """
        from .frequencies import interp1d

        cross_power = self.cross_powers.get((b1.id, b2.id))
        ntarget = int(freq_map.alpha.shape[0])
        if cross_power is None:
            return np.zeros(ntarget, dtype=float)
        return interp1d(freq_map, np.asarray(cross_power))
    
        

