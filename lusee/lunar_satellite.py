# class for dealing with orbiting relay / calibrator satelite
from lunarsky.time import Time
from astropy.time import TimeDelta
import numpy as np
from scipy.interpolate import interp1d
import astropy.constants as ac
import astropy.units as u
from scipy.spatial.transform import Rotation as R
from lunarsky import MCMF, SkyCoord, LunarTopo


class LSatellite:
    def __init__(
        self,
        semi_major_km=5740,
        eccentricity=0.58,
        inclination_deg=54.856,
        raan_deg=0,
        argument_of_pericenter_deg=86.322,
        aposelene_ref_time=Time("2024-05-01T00:00:00"),
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

    def predict_position_mcmf(self, times):
        ## neeed to do this like this
        dt = np.array([float((t - self.t0) / TimeDelta(1 * u.d)) for t in times])
        mean_anomaly = 2 * np.pi * ((dt / self.period) % 1.0)
        phi_moon = -2 * np.pi * ((dt / self.moon_sidereal_period) % 1.0)
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


class ObservedSatellite:
    def __init__(self, observation, satellite):
        self.observation = observation
        self.satelite = satellite
        self.posxyz = satellite.predict_position_mcmf(observation.times)
        self.sky_coords = SkyCoord(MCMF(*(self.posxyz.T)))
        self.satpos = self.sky_coords.transform_to(LunarTopo(location=observation.loc))

    def alt_rad(self):
        return np.array(self.satpos.alt).astype(float) / 180.0 * np.pi

    def az_rad(self):
        return np.array(self.satpos.az).astype(float) / 180.0 * np.pi

    def dist_km(self):
        return np.array(self.satpos.distance / u.km).astype(float)

    def get_transit_indices(self):
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

    def plot_tracks(self, ax):
        transits = self.get_transit_indices()
        az = self.az_rad()
        alt = self.alt_rad()
        X = np.sin(az) * np.cos(alt)
        Y = np.cos(az) * np.cos(alt)
        for s, e in transits:
            ax.plot(X[s:e], Y[s:e])
