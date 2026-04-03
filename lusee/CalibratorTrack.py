import numpy as np


class CalibratorTrack:
    """
    Contains the information about a calibrator satellite overpass.

    Arrays alt, az, and polarization must have the same length as times
    (NTime = number of samples in the overpass). tone_freqs and tone_amplitude
    are NFreq arrays that are independent of NTime; tone_amplitude must match
    the length of tone_freqs (one amplitude per frequency).

    :param times: Observation times (astropy Time array, same format as Observation.times)
    :param alt: Altitude above horizon in radians
    :param az: Azimuth in radians (astronomical convention, from N towards E)
    :param polarization: Spin-2 polarization angle in radians
        (0 = pure E_theta, pi/2 = pure E_phi)
    :param tone_freqs: Tone frequencies in MHz — NFreq values, independent of NTime
    :param tone_amplitude: Tone amplitudes (same units as sky temperature),
        one per frequency (must match len(tone_freqs))
    """

    def __init__(self, times, alt, az, polarization, tone_freqs, tone_amplitude):
        times          = np.asarray(times)
        alt            = np.asarray(alt,            dtype=float)
        az             = np.asarray(az,             dtype=float)
        polarization   = np.asarray(polarization,   dtype=float)
        tone_freqs     = np.asarray(tone_freqs,     dtype=float)
        tone_amplitude = np.asarray(tone_amplitude, dtype=float)

        n = len(times)
        for name, arr in [
            ("alt",          alt),
            ("az",           az),
            ("polarization", polarization),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"All arrays must have the same length as times ({n}); "
                    f"got len({name})={len(arr)}"
                )

        if len(tone_amplitude) != len(tone_freqs):
            raise ValueError(
                f"tone_amplitude must have the same length as tone_freqs "
                f"({len(tone_freqs)}); got len(tone_amplitude)={len(tone_amplitude)}"
            )

        self.times          = times
        self.alt            = alt
        self.az             = az
        self.polarization   = polarization
        self.tone_freqs     = tone_freqs
        self.tone_amplitude = tone_amplitude

    def __len__(self):
        return len(self.times)
