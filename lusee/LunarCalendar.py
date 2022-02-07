# Lunar calendar class, based on the module "lunar_calendar" by A.Slosar
# Adding cache management functionality

import  numpy as np

import  astropy             as ap
import  astropy.coordinates as coord
import  astropy.units       as u
from    astropy.time import Time


from datetime import datetime
from datetime import timedelta

import lunarsky
import lunarsky.time as LTime

from scipy.optimize import minimize_scalar

import shelve
import atexit
import os

class LunarCalendar:
    '''
    The LunarCalendar class which allows to define the cache
    file name and whether it is to be used in the firstplace
    '''
    #######################################
    def close_db(self):
        if self.db:
            self.db.close()
            if self.cleanup:
                try:
                    os.remove(self.cache)
                except:
                    if self.verbose:
                        print(f'''Could not remove cache file "{self.cache}"''')
    
    #######################################
    def __init__(self, cache='', cleanup=True, verbose=False):
        self.db         = None
        self.cache      = cache
        self.cleanup    = cleanup
        self.verbose    = verbose

        if(self.cache!=''):
            self.db = shelve.open(cache, writeback=True)
            atexit.register(self.close_db)

    #######################################
    def get_sun_alt(self, times):
        '''
        Returns Sun altitude at a given Moon location 
        for a set of times
        '''

        loc = lunarsky.MoonLocation.from_selenodetic(180.0, 0)

        # how I think it should be
        # coord.get_sun(times).transform_to(lunarsky.LunarTopo(location=loc, obstime=times))
        # This is really inefficient -- there should be a better way, but thing above crashes

        alts = [
            float(
                coord.get_sun(time_)
                .transform_to(lunarsky.LunarTopo(location=loc, obstime=time_))
                .alt
                / u.deg
            )
            for time_ in times
            ]
        return alts


    #######################################
    def get_lunar_nights(self, year=2025):
        ''' 
        Gets a list of lunar noon-noon cycles in year.
        Returns a list of tuples.
        Each tuple is (time_start, time_end)
        Where times are astropy times at lunar "noon" (Sun transits) on the farside
        time_end corresponds to time_start of the next day.
        '''

        # check if we use cache

        if self.db!=None:
            if self.verbose:
                print(f'''Using cache file "{self.cache}"''')
            cache_key = f"lunar_{year}"
            if cache_key in self.db:
                if self.verbose: print(f'''Cache key "{cache_key}" found''')
                return self.db[cache_key]

        # we do this from a point at the center of far side, a matter of nigth cycle definition,
        # regardless of where our observatory is

        # let's do the rough resolution first
        t = np.arange(datetime(year, 1, 1, 0), datetime(year + 1, 1, 31, 0), timedelta(days=1)).astype(datetime)
        lt = LTime.Time(t)
        alts = self.get_sun_alt(lt)
        # now count transits and finetune them
        transits = []
        cvlast = 0
        for i in range(1, len(alts) - 1):
            if (alts[i] > alts[i - 1]) and (alts[i] > alts[i + 1]):
                ## need to convert to a float func we can minimize
                bracket_l = lt[i - 1]
                delta_t = lt[i + 1] - lt[i - 1]
                objf = lambda t_: 90.0 - self.get_sun_alt([bracket_l + t_ * delta_t])[0]
                res = minimize_scalar(objf, bracket=(0, 0.5, 1.0), tol=1e-4)
                assert res.success
                transits.append(bracket_l + res.x * delta_t)

        result = []
        for st, en in zip(transits[:-1], transits[1:]):
            # only use if start time in this year
            if st < datetime(year + 1, 1, 1, 0):
                result.append((st, en))

        if self.db!=None:
            self.db[cache_key] = result
        
        return result

    #######################################
    def get_lunar_start_end(self, lunar_night=2500):
        '''
        Returns noon-noon dates for the lunar date
        '''
        
        if self.verbose:
            print('Verbose mode activated')
        yr = lunar_night // 100
        cycle = lunar_night % 100
        
        nights = self.get_lunar_nights(2000 + yr)
        
        if cycle >= len(nights):
            raise IndexError
    
        return nights[cycle]