import numpy as np
import astropy as ap
import astropy.time as apt
from astropy.time import Time
import lunarsky.time as LTime
import astropy.units as u
import lunarsky

from . import calendar


class LObservation:
    def __init__(
        self,
        lunar_day=2500,
        lun_lat_deg=-10.0,
        lun_long_deg=180.0,
        lun_height_m=0,
        deltaT_sec=15 * 60,
    ):
        """ Initializes a basic Lunar Observation object for
            an observatory in selenographic coordinates. 
            deltaT specifies the time resolution of observations

        """

        self.lunar_day = lunar_day
        self.lun_lat = lun_lat_deg / 180 * np.pi
        self.lun_long = lun_long_deg / 180 * np.pi
        self.lun_heigh = lun_height_m
        self.loc = lunarsky.MoonLocation.from_selenodetic(
            lon=lun_long_deg, lat=lun_lat_deg, heigh=lun_height_m
        )

        self.time_start, self.time_end = calendar.lunar_start_end(lunar_day)
        self.deltaT = apt.timedelta(seconds=deltaT_sec)
        self.times = np.arange(
            self.time_start, self_timend + self.deltaT, self.deltaT
        ).astype(datetime)

    def get_track_solar(self, objid):
        """ get a track in alt,az coordinates for an object in the solar system
            on the self.times time stamps.
            objid can be 'sun', 'moon' (as debug, should be alt=-90),
            or plantes id (jupyter, etc)
        """
        raise NotImplemented

    def get_track_ra_dec(self, objid):
        """ get a track in alt,az coordinates for an object with celecstial coordinates
            in ra,dec on the self.times time stamps.
            objid can be 'sun', 'moon' (as debug, should be alt=-90),
            or plantes id (jupyter, etc)
        """
        raise NotImplemented
