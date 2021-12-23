#
# A set of utilities to perform calendar calculations
#

import astropy as ap
import astropy.time as apt


def get_lunar_nights (year=2025):
    """ 
      Gets a list of lunar noon-noon cycles in year.
      Returns a list of tuples.
      Each tuple is (time_start, time_end)
      Where times are astropy times at lunar "noon" (Sun transits)
      time_end corresponds to time_start of the next day.
    """

    raise NotImplementedError


def get_lunar_start_end(lunar_night=2500):
    yr = lunar_night // 100
    cycle = lunar_night % 100
    nights = get_lunar_nights(2000+yr)
    if len(nights) >= cycle:
        raise IndexError
    return nights[cycle]

        






