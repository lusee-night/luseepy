import numpy as np


class CalibratorTrack:
    """
    Contains the information about a calibrator satellite overpass.

    All arrays must have the same length (number of samples in the overpass).

    :param times: Observation times (astropy Time array, same format as Observation.times)
    :param alt: Altitude above horizon in radians
    :param az: Azimuth in radians (astronomical convention, from N towards E)
    :param polarization: Spin-2 polarization angle in radians
        (0 = pure E_theta, pi/2 = pure E_phi)
    :param tone_freqs: Tone frequencies in MHz, one per time sample
    :param tone_amplitude: Tone amplitudes (same units as sky temperature), one per sample
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
            ("alt",            alt),
            ("az",             az),
            ("polarization",   polarization),
            ("tone_freqs",     tone_freqs),
            ("tone_amplitude", tone_amplitude),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"All arrays must have the same length as times ({n}); "
                    f"got len({name})={len(arr)}"
                )

        self.times          = times
        self.alt            = alt
        self.az             = az
        self.polarization   = polarization
        self.tone_freqs     = tone_freqs
        self.tone_amplitude = tone_amplitude

    def __len__(self):
        return len(self.times)
