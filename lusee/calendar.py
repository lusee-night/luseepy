#
# A set of utilities to perform calendar calculations
#
import numpy as np

import astropy as ap
import astropy.coordinates as coord
from astropy.time import Time
import lunarsky.time as LTime
import astropy.units as u
from datetime import datetime
from datetime import timedelta
import lunarsky


def get_sun_alt(loc, times):
    """ returns Sun altitude at a given Moon location 
        for a set of times """

    loc = lunarsky.MoonLocation.from_selenodetic (180.,0)
    #altaz = lunarsky.LunarTopo(location=loc, obstime=lt)
    #sun = coord.get_sun(lt)
    #print(sun[0].transform_to(altaz))

    # This is really inefficient -- there should be a better way, but thing above crashes

    alts = [float(coord.get_sun(time_).transform_to(lunarsky.LunarTopo(location=loc, obstime=time_)).alt/u.deg)
                                      for time_ in times]
    return alts
    


def get_lunar_nights (year=2025):
    """ 
      Gets a list of lunar noon-noon cycles in year.
      Returns a list of tuples.
      Each tuple is (time_start, time_end)
      Where times are astropy times at lunar "noon" (Sun transits) on the farside
      time_end corresponds to time_start of the next day.
    """

    # we do this from a point at the center of far side,
    # as a matter of nigth cycle definition, regardless of where our observatory
    # is


    # let's do the rough resolution first
    loc = lunarsky.MoonLocation.from_selenodetic (180.,0)
    t = np.arange(datetime(year, 1, 1, 0), datetime(year+1, 1, 31, 0),timedelta(days=1)).astype(datetime)
    lt = LTime.Time(t,location=loc)
    alts = get_sun_alt(loc, lt)
    # now count transits and finetune them
    transits = []
    cvlast=0
    print (" Fix this maximizer. You cannot bisect to maximum!!")
    stop()
    for i in range(1,len(alts)-1):
        if ((alts[i]>alts[i-1]) and (alts[i]>alts[i+1])):
            # we have a local maximum, let's do bisection to max
            a = lt[i-1]
            av = alts[i-1]
            b = lt[i+1]
            bv = alts[i+1]
            #if  lt[i-1]>lt[i+1]:
            #     a=lt[i-1]
            #     av=alts[i-1]
            #     b=lt[i]
            #     bv=alts[i]
            #else:
            #     a=lt[i]
            #     av=alts[i]
            #     b=lt[i+1]
            #     bv=alts[i+1]
            print ('----------')
            while True:
                c=a+0.5*(b-a)
                print (a,c,b)
                cv = (get_sun_alt(loc,[c]))[0]
                print (av,cv,bv)
                assert(cv>min(av,bv))
                assert(max(av,bv)<cv)
                assert(b>a)
                
                if (b-a<1/(24*60)): ## get it to a minute
                    break
                if av<bv:
                    a=c
                    av=cv
                else:
                    b=c
                    bv=cv
            print (c,cv)
            transits.append(c)

    result = []
    for st,en in zip(transits[:-1], transits[1:]):
        # only use if start time in this year
        if st < datetime(year+1, 1, 1, 0): 
            result.append((st,en))
    return result
    
            
def get_lunar_start_end(lunar_night=2500):
    yr = lunar_night // 100
    cycle = lunar_night % 100
    nights = get_lunar_nights(2000+yr)
    print (len(nights),cycle,yr)
    if cycle>=len(nights):
        raise IndexError
    return nights[cycle]

        






