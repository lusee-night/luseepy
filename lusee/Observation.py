import  numpy           as np

import  astropy         as ap
import  astropy.time    as apt
import  astropy.units   as u
import  astropy.coordinates as coord

from    astropy.time    import TimeDelta
from    astropy.coordinates.builtin_frames import icrs

import  lunarsky
from    lunarsky.time   import Time
from    lunarsky        import MoonLocation
from    lunarsky        import SkyCoord
from    lunarsky        import LunarTopo

from    datetime        import datetime
from    datetime        import timedelta

from    .LunarCalendar  import LunarCalendar


# ---

class Observation:
    def __init__(
        self,
        time       =   2500,
        lun_lat_deg     =   -23.814,
        lun_long_deg    =   182.258,
        lun_height_m    =   0,
        deltaT_sec      =   15*60,
    ):
        """
        Initializes a basic Lunar Observation object for
        an observatory in selenographic coordinates. 
        deltaT specifies the time resolution of observations

        lunar day can be specified as:
           int = lunar day as per LunarCalendar
           "CY##" or "CY####"  = full calendar year 1/1 to 12/31
           "FY##" or "FY####"  = full fiscale year  10/1 to 9/30
           "UTC to UTC", i.e. '2025-02-01 13:00:00 to 2025-04-01 16:00:00'

        """
        self.time      = time
    
        self.lun_lat_deg    = lun_lat_deg  
        self.lun_long_deg   = lun_long_deg 
        self.lun_height_m   = lun_height_m
        self.deltaT_sec     = deltaT_sec

        self.lun_lat        = lun_lat_deg   / 180*np.pi
        self.lun_long       = lun_long_deg  / 180*np.pi
        
        self.loc = MoonLocation.from_selenodetic(lon=lun_long_deg, lat=lun_lat_deg, height=lun_height_m)

        if type(time) == int:
            lc = LunarCalendar()
            self.time_start, self.time_end = lc.get_lunar_start_end(time)
        elif(type(time)==str):
            assert(type(time)==str)
            ## parse the string to determine syntax
            if time[0:2]=='CY':
                year = int(time[2:])
                if year<100: year+=2000
                self.time_start = Time(datetime(year,   1, 1, 1))
                self.time_end   = Time(datetime(year+1, 1, 1))
            elif time[0:2]=='FY':
                year = int(time[2:])
                if year<100:
                    year+=2000
                self.time_start = Time(datetime(year-1, 10, 1))
                self.time_end   = Time(datetime(year , 10, 1))
            elif " to " in time:
                start, end      = time.split(" to ")
                self.time_start = Time(start)
                self.time_end   = Time(end)
            else:
                raise NotImplementedError
        elif(type(time)==tuple):
            self.time_from_range(time)
        else:
            raise NotImplementedError                    

        self.deltaT = TimeDelta(deltaT_sec * u.s) # NB units
        self.times  = np.arange(self.time_start, self.time_end + self.deltaT, self.deltaT).astype(Time)


    # ---
    def time_from_range(self, tpl):
        self.time_start = Time(tpl[0])
        self.time_end   = Time(tpl[1])
    
    # ---
    def get_track_solar(self, objid):
        """
        Get a track an object in the solar system, in (alt,az) coordinates as a time series on [self.times].
        objid can be 'sun', 'moon' (as debug, should be alt=-90), or planets id (jupyter, etc)
        """
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

        alt = np.array([float(altaz_.alt / u.rad) for altaz_ in altaz])
        az = np.array([float(altaz_.az / u.rad) for altaz_ in altaz])
        track = (alt, az)

        return track

    # ---
    def get_track_ra_dec(self, ra, dec, times = None):
        """ get a track in alt,az coordinates for an object with celecstial coordinates
            in ra,dec on the self.times time stamps.
            ra,dec are in degrees
        """
        if times is None:
            times = self.times
        if type(ra) == float:
            c = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")
        elif type(ra) == str:
            c = SkyCoord(ra=ra, dec=dec, frame="icrs")

        altaz = [
            c.transform_to(lunarsky.LunarTopo(location=self.loc, obstime=time_))
            for time_ in times
        ]

        alt = np.array([float(altaz_.alt / u.rad) for altaz_ in altaz])
        az = np.array([float(altaz_.az / u.rad) for altaz_ in altaz])
        track = (alt, az)
        return track

    # ---
    def get_track_l_b(self, l, b, times= None):
        """ get a track in l,b coordinates for an object with celecstial coordinates
            in l,b galactic on the self.times time stamps.
            ra,dec are in degrees
        """
        if times is None: times = self.times
    
        if type(l) == float:
            c = coord.SkyCoord(l=l, b=b, frame="galactic", unit="deg")
        elif type(l) == str:
            c = coord.SkyCoord(l=l, b=b, frame="galactic")
            
        return self.get_track_ra_dec(float(c.icrs.ra/u.deg), float(c.icrs.dec/u.deg), times=times)

    # ---
    def get_ra_dec_from_alt_az (self, alt, az, times = None):
        """ get a track in ra_dec given alt, az.
            alt, az are astro convention (from N towards E)
            returns ra dec in *radians*
        """

        if times is None: times = self.times

        icrs = [LunarTopo(alt = alt*u.rad, az = az*u.rad, obstime = time_, 
                 location = self.loc).transform_to(coord.ICRS()) for time_ in times]
        
        ra = np.array([float(x.ra/u.rad) for x in icrs])
        dec = np.array([float(x.dec/u.rad) for x in icrs])
        return ra, dec

    # ---
    def get_l_b_from_alt_az (self, alt, az, times = None):
        """ get a track in l b given alt, az.
            alt, az are astro convention (from N towards E)
            returns ra dec in *radians*
        """

        if times is None: times = self.times

        galactic = [LunarTopo(alt = alt*u.rad, az = az*u.rad, obstime = time_, 
                 location = self.loc).transform_to(coord.Galactic()) for time_ in times]
        l = np.array([float(x.l/u.rad) for x in galactic])
        b = np.array([float(x.b/u.rad) for x in galactic])
        return l,b

        
