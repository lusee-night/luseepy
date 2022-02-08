import  numpy           as np

import  astropy         as ap
import  astropy.time    as apt
import  astropy.units   as u
import  astropy.coordinates as coord
from    astropy.time    import TimeDelta



import  lunarsky
from    lunarsky.time   import Time
from    lunarsky        import MoonLocation


from    datetime        import datetime
from    datetime        import timedelta


from    .LunarCalendar  import LunarCalendar

class LObservation:
    def __init__(
        self,
        lunar_day       =   2500,
        lun_lat_deg     =   -10.0,
        lun_long_deg    =   180.0,
        lun_height_m    =   0,
        deltaT_sec      =   15*60,
    ):
        """
        Initializes a basic Lunar Observation object for
        an observatory in selenographic coordinates. 
        deltaT specifies the time resolution of observations
        """
        self.lunar_day  = lunar_day
        self.lun_lat    = lun_lat_deg   / 180*np.pi
        self.lun_long   = lun_long_deg  / 180*np.pi
        self.lun_heigh  = lun_height_m

        self.loc = MoonLocation.from_selenodetic(lon=lun_long_deg, lat=lun_lat_deg, height=lun_height_m)

        lc = LunarCalendar()

        self.time_start, self.time_end = lc.get_lunar_start_end(lunar_day)
        self.deltaT = TimeDelta(deltaT_sec * u.s)
        self.times = np.arange(
            self.time_start, self.time_end + self.deltaT, self.deltaT
        ).astype(Time)

    def get_track_solar(self, objid):
        """ get a track in alt,az coordinates for an object in the solar system
            on the self.times time stamps.
            objid can be 'sun', 'moon' (as debug, should be alt=-90),
            or plantes id (jupyter, etc)
        """
#        cache_key = f"track_solar_{objid}"
#        if cache_key in self.cache:
#            return self.cache[cache_key]

        valid_bodies = coord.solar_system_ephemeris.bodies
        if objid not in valid_bodies:
            print(f"{objid} not a valid body name. Use :", valid_bodies)
            raise ValueError

        altaz = [
            coord.get_body(objid, time_).transform_to(
                lunarsky.LunarTopo(location=self.loc, obstime=time_)
            )
            for time_ in self.times
        ]

        alt = np.array([np.float(altaz_.alt / u.rad) for altaz_ in altaz])
        az = np.array([np.float(altaz_.az / u.rad) for altaz_ in altaz])
        track = (alt, az)
#        self.cache[cache_key] = track
        return track

    def get_track_ra_dec(self, ra, dec):
        """ get a track in alt,az coordinates for an object with celecstial coordinates
            in ra,dec on the self.times time stamps.
            objid can be 'sun', 'moon' (as debug, should be alt=-90),
            or plantes id (jupyter, etc)
        """
#        cache_key = f"track_ra_dec_{ra}_{dec}"
#        if cache_key in self.cache:
#            return self.cache[cache_key]

        if type(ra) == float:
            c = coord.SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")
        elif type(ra) == str:
            c = coord.SkyCoord(ra=ra, dec=dec, frame="icrs")

        altaz = [
            c.transform_to(lunarsky.LunarTopo(location=self.loc, obstime=time_))
            for time_ in self.times
        ]

        alt = np.array([np.float(altaz_.alt / u.rad) for altaz_ in altaz])
        az = np.array([np.float(altaz_.az / u.rad) for altaz_ in altaz])
        track = (alt, az)
#           self.cache[cache_key] = track
        return track
