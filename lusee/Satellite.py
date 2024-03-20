# class for dealing with orbiting relay / calibrator satelite
import  numpy as np

from    lunarsky.time       import Time
from    astropy.time        import TimeDelta
from    scipy.interpolate   import interp1d
import  astropy.constants   as ac
import  astropy.units       as u
from    scipy.spatial.transform import Rotation as R
from    lunarsky            import MCMF, SkyCoord, LunarTopo


class Satellite:
    """
    Class that defines satellite parameters and position
    
    :param semi_major_km: Semi-major axis of body in km
    :type semi_major_km: float
    :param eccentricity: Eccentricity of orbit
    :type eccentricity: float
    :param inclination_deg: Inclination of orbit
    :type inclination_deg: float
    :param raan_deg: Right-ascension angle in degrees
    :type raan_deg: float
    :param argument_of_pericenter_deg: Argument of pericenter of orbit in degrees
    :type argument_of_pericenter_deg: float
    :param aposelene_ref_time: Aposelene Reference Time
    :type aposelene_ref_time: lunarsky.time
    """
    ### ------------
    def __init__(self,
        semi_major_km               =5738,
        eccentricity                =0.56489,
        inclination_deg             =57.097,
        raan_deg                    =0,
        argument_of_pericenter_deg  =72.625,
        aposelene_ref_time          =Time("2024-05-01T00:00:00"),
    ):
        ## first period
        M_moon = 7.34767309e22 * u.kg
        self.semi_major = semi_major_km
        self.period = float(
            np.sqrt((4 * np.pi ** 2 * (semi_major_km * u.km) ** 3) / (ac.G * M_moon))
            / (1 * u.day)
        )
        self.moon_sidereal_period = 27.322  # in days
        self.e = eccentricity

        ## first need to get the mean anomaly to eccentric anomaly interpolator
        E_ = np.linspace(0, 2 * np.pi, 50000)
        M_ = E_ - eccentricity * np.sin(E_)
        self.M2E = interp1d(M_, E_)

        ## now we need to defined two nominal vectors: one pointing at percenter
        ## and the other perpendicular to it

        ## Ok, this is some hocus pocus that might or might not be right
        r = R.from_euler(
            "zxz", [argument_of_pericenter_deg, inclination_deg, raan_deg], degrees=True
        )
        self.pericent_norm = r.apply(np.array([1.0, 0.0, 0.0]))
        self.periperp_norm = r.apply(np.array([0.0, 1.0, 0.0]))
        self.t0 = aposelene_ref_time

    ### ------------
    def predict_position_mcmf(self, times):
        """    
        Function that returns an array of satellite positions for input times
        
        :param times: Array of lunarsky.time times at which to evaluate satellite position
        :type times: array[lunarsky.time]
        
        :returns: Position of satellite body
        :rtype: Numpy array
        """
        
        ## neeed to do this like this
        dt = np.array([float((t - self.t0) / TimeDelta(1 * u.d)) for t in times])

        mean_anomaly    = 2 * np.pi * ((dt / self.period) % 1.0)
        phi_moon        =-2 * np.pi * ((dt / self.moon_sidereal_period) % 1.0)

        E = self.M2E(mean_anomaly)
        r = self.semi_major * (1 - self.e * np.cos(E))

        true_anomaly = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(E / 2), np.sqrt(1 - self.e) * np.cos(E / 2)
        )

        pos = np.outer(r * np.cos(true_anomaly), self.pericent_norm) + np.outer(
            r * np.sin(true_anomaly), self.periperp_norm
        )

        # this is in the axis where moon has rotated since aposelene_ref_time
        pos = np.array(
            [
                R.from_euler("z", phi_moon_, degrees=False).apply(v)
                for phi_moon_, v in zip(phi_moon, pos)
            ]
        )
        return pos


##############################################
class ObservedSatellite:
    """
    Class that calculates satellite observables
    
    :param observation: Observation parameters, from lusee.observation class
    :type observation: class
    :param satellite: Satellite parameters, from lusee.satellite class
    :type satellite: class
    
    """    
    def __init__(self, observation, satellite):
        self.observation = observation
        self.satelite = satellite
        self.posxyz = satellite.predict_position_mcmf(observation.times)
        self.sky_coords = SkyCoord(MCMF(*(self.posxyz.T*u.km)))
        self.satpos = self.sky_coords.transform_to(LunarTopo(location=observation.loc))

    def alt_rad(self):
        """
        Function that returns satellite altitude in radians
        
        :returns: Satellite altitude
        :rtype: numpy array
        """
        return np.array(self.satpos.alt).astype(float) / 180.0 * np.pi

    def az_rad(self):
        """
        Function that returns satellite azimuth in radians
        
        :returns: Satellite azimuth
        :rtype: numpy array
        """
        
        return np.array(self.satpos.az).astype(float) / 180.0 * np.pi

    def dist_km(self):
        """
        Function that returns distance to satellite in km
        
        :returns: Satellite distance
        :rtype: numpy array
        """
        return np.array(self.satpos.distance / u.km).astype(float)

    def get_transit_indices(self):
        """
        Function that returns an array of transit indices for satellite and observation times specified in ObservedSatellite class

        :returns: Transit Indices
        :rtype: array

        """

        visible = self.alt_rad() > 0
        passes = []
        if visible[0]:
            si = 0
            tostate = False
        else:
            tostate = True

        for i, v in enumerate(visible):
            if v == tostate:
                if not tostate:
                    passes.append((si, i))
                else:
                    si = i
                tostate = not tostate
        return passes

    ### ------------
    def plot_tracks(self, ax, lin_map = False):
        """
        Function that plots satellite trajectories
        
        :param ax: Plot axis object
        :param lin_map: Use linear approximation for small altitudes? 
        :type lin_map: bool
        """
        transits = self.get_transit_indices()
        az = self.az_rad()
        alt = self.alt_rad()
        if lin_map:
            X = np.sin(az) * (1-alt/(np.pi/2))
            Y = np.cos(az) * (1-alt/(np.pi/2))
        else:
            X = np.sin(az) * np.cos(alt)
            Y = np.cos(az) * np.cos(alt)
        for s, e in transits:
            ax.plot(X[s:e], Y[s:e])

    def get_track_coverage(self, Nphi=10, Nmu=10):
        """
        Function that returns array of alt-az bins showing where satellite transit passed. Bins are 1 if satellite passed through bin, 0 if not.
        
        :param Nphi: Number of az bins
        :type Nphi: int
        :param Nmu: Number of alt bins
        :type Nmu: int
        
        :returns: Binned satellite transits
        :rtype: numpy array
        
        """
        transits = self.get_transit_indices()
        altbin = (np.sin(self.alt_rad())*Nmu).astype(int)
        azbin = (self.az_rad()/(2*np.pi)*Nphi).astype(int)
        
        m = np.zeros((Nmu,Nphi))
        for s,e in transits:
            #print (azbin[s:e].min(), azbin[s:e].max(), altbin[s:e].min(), altbin[s:e].max(), self.alt_rad()[s:e].min(), self.alt_rad()[s:e].max())
            m[altbin[s:e],azbin[s:e]] = 1

        return m
    
    
