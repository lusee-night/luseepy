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
    """
    Class that initializes a basic Lunar Observation object for
    an observatory in selenographic coordinates. 

    Lunar day can be specified as:
|       int = lunar day as per LunarCalendar
|       "CY##" or "CY####"  = full calendar year 1/1 to 12/31
|       "FY##" or "FY####"  = full fiscale year  10/1 to 9/30
|       "UTC to UTC", i.e. '2025-02-01 13:00:00 to 2025-04-01 16:00:00'

    :param lunar_day: Lunar day on which observation takes place
    :type lunar_day: int or str, see above
    :param lun_lat_deg: Lunar latitude of observatory in degrees
    :type lun_lat_deg: float
    :param lun_long_deg: Lunar longitude of observatory in degrees
    :type lun_long_deg: float
    :param lun_height_m: Height of observatory above lunar surface in meters
    :type lun_height_m: float
    :param deltaT_sec: Time resolution of observations
    :type deltaT_sec: float
    """
    
    default_time_range      = 2500 # default to lunar day number 2500
    default_lun_lat_deg     = -23.814
    default_lun_long_deg    = 182.258
    default_lun_height_m    = 0

    def __init__(
        self,
        time_range      =   default_time_range,
        lun_lat_deg     =   default_lun_lat_deg,
        lun_long_deg    =   default_lun_long_deg,
        lun_height_m    =   default_lun_height_m,
        deltaT_sec      =   15*60,
    ):
        self.time_range     = time_range
        self.lun_lat_deg    = lun_lat_deg  
        self.lun_long_deg   = lun_long_deg

        self.lun_height_m   = lun_height_m
        self.deltaT_sec     = deltaT_sec

        self.lun_lat        = lun_lat_deg   / 180*np.pi
        self.lun_long       = lun_long_deg  / 180*np.pi
        
        # Locattion on the Moon:
        self.loc = MoonLocation.from_selenodetic(lon=self.lun_long_deg, lat=self.lun_lat_deg, height=self.lun_height_m)

        if type(time_range) == int:
            lc = LunarCalendar()
            self.time_start, self.time_end = lc.get_lunar_start_end(time_range)
        elif(type(time_range)==str):
            assert(type(time_range)==str)
            ## parse the string to determine syntax
            if time_range[0:2]=='CY':
                year = int(time_range[2:])
                if year<100: year+=2000
                self.time_start = Time(datetime(year,   1, 1, 1))
                self.time_end   = Time(datetime(year+1, 1, 1))
            elif time_range[0:2]=='FY':
                year = int(time_range[2:])
                if year<100:
                    year+=2000
                self.time_start = Time(datetime(year-1, 10, 1))
                self.time_end   = Time(datetime(year , 10, 1))
            elif " to " in time_range:
                start, end      = time_range.split(" to ")
                self.time_start = Time(start)
                self.time_end   = Time(end)
            else:
                raise NotImplementedError
        elif(type(time_range)==tuple):
            self.set_time_range(time_range)
        else:
            raise NotImplementedError                    

        self.deltaT = TimeDelta(deltaT_sec * u.s) # NB units
        self.times  = np.arange(self.time_start, self.time_end + self.deltaT, self.deltaT).astype(Time)


    # ---
    def set_time_range(self, tpl):
        self.time_start = Time(tpl[0])
        self.time_end   = Time(tpl[1])
    
    # ---
    def get_track_solar(self, objid):
        """
        Function that calculates a track in (Alt, Az) coordinates for an object in the solar system at the self.times time stamps.
        objid can be 'sun', 'moon' (as debug, should be alt=-90), or planet id (jupiter, etc)

        :param objid: Object ID ('sun', 'moon', 'jupiter', etc.)
        :type objid: str

        :returns: (Alt, Az) coordinates of object at self.times 
        :rtype: numpy array
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
        """ 
        Function that calculates a track in (Alt, Az) coordinates for an object with celestial coordinates in (RA, Dec) at the self.times time stamps.
        (RA, Dec) are given in degrees

        :param ra: Right Ascension
        :type ra: numpy array
        :param dec: Declination
        :type dec: numpy array
        :param times: Times for observation. Defaults to self.times if not specified
        :type times: numpy array

        :returns: (Alt, Az) coordinates of object at self.times 
        :rtype: numpy array
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
        """ 
        Function that calculates a track in (Alt, Az) coordinates for an object with celestial coordinates (l, b) in the galactic coordinate system, at the self.times time stamps.
        (l, b) are given in degrees

        :param l: Galactic longitude, zero angle is from sun to galactic center
        :type l: numpy array
        :param b: Galactic latitude, measures angle above the galactic plane 
        :type b: numpy array
        :param times: Times for observation. Defaults to self.times if not specified
        :type times: numpy array
        
        :returns: (Alt, Az) coordinates of object at self.times 
        :rtype: numpy array
        """
        if times is None: times = self.times
    
        if type(l) == float:
            c = coord.SkyCoord(l=l, b=b, frame="galactic", unit="deg")
        elif type(l) == str:
            c = coord.SkyCoord(l=l, b=b, frame="galactic")
            
        return self.get_track_ra_dec(float(c.icrs.ra/u.deg), float(c.icrs.dec/u.deg), times=times)

    # ---
    def get_ra_dec_from_alt_az (self, alt, az, times = None):
        """ 
        Function that calculates a track in (RA, Dec) given coordinates in lunarcentric (Alt, Az).
        Alt and Az are given following the astronomical convention (angles measured from N towards E).
        Returns (RA, Dec) in *radians*

        :param alt: Altitude
        :type alt: numpy array
        :param az: Azimuth
        :type az: numpy array
        :param times: Times for observation. Defaults to self.times if not specified
        :type times: numpy array

        :returns: (RA, Dec) coordinates of object at self.times 
        :rtype: numpy array        
        """

        if times is None: times = self.times

        icrs = [LunarTopo(alt = alt*u.rad, az = az*u.rad, obstime = time_, 
                 location = self.loc).transform_to(coord.ICRS()) for time_ in times]
        
        ra = np.array([float(x.ra/u.rad) for x in icrs])
        dec = np.array([float(x.dec/u.rad) for x in icrs])
        return ra, dec

    # ---
    def get_l_b_from_alt_az (self, alt, az, times = None):
        """ 
        Function that calculates a track in (l, b) galactic coordinates given coordinates in lunarcentric (Alt, Az).
        Alt and Az are given following the astronomical convention (angles measured from N towards E).
        Returns (l, b) in *radians*

        :param alt: Altitude
        :type alt: numpy array
        :param az: Azimuth
        :type az: numpy array
        :param times: Times for observation. Defaults to self.times if not specified
        :type times: numpy array

        :returns: (l, b) coordinates of object at self.times 
        :rtype: numpy array   
        """

        if times is None: times = self.times

        galactic = [LunarTopo(alt = alt*u.rad, az = az*u.rad, obstime = time_, 
                 location = self.loc).transform_to(coord.Galactic()) for time_ in times]
        l = np.array([float(x.l/u.rad) for x in galactic])
        b = np.array([float(x.b/u.rad) for x in galactic])
        return l,b

        
