import numpy as np
import astropy as ap
import astropy.time as apt
from . import calendar


class LObservation:

    def __init__ (self, lunar_day = 2500, lun_lat_deg = -10., lun_long_deg = 180., lun_height_m = 0,  deltaT_sec = 15*60):
        """ Initializes a basic Lunar Observation object for
            an observatory in selenographic coordinates. 
            deltaT specifies the time resolution of observations

        """


        self.lunar_day = lunar_day
        self.lun_lat = lun_lat_deg/180*np.pi
        self.lun_long = lun_long_deg/180*np.pi
        self.lun_heigh = lun_height_m

        self.time_start, self.time_end = calendar.lunar_start_end(lunar_day)
        self.deltaT = apt.timedelta(seconds = deltaT_sec)
        self.times = np.arange(self.time_start, self_timend+self.deltaT, self.deltaT).astype(datetime)

