import numpy as np
from astropy.io import fits as astrofits


class CalibratorSimulator:
    """
    Simulates calibrator satellite observations.

    Takes an Observation (with calibrator_tracks populated) and a list of Beam
    objects. For each CalibratorTrack in obs.calibrator_tracks, evaluates the
    complex E-field response of each beam along the satellite's trajectory.

    Result: list of complex arrays, one per pass, shape (NTime, NBeam, NFreq).

    Signal formula for pass p, time t, beam b, frequency f::

        E_theta = beam_b.interp_Etheta(alt_t, az_t, tone_freqs[f])
        E_phi   = beam_b.interp_Ephi  (alt_t, az_t, tone_freqs[f])
        result[p][t, b, f] = tone_amplitude[t] * (E_theta * cos(pol_t) + E_phi * sin(pol_t))

    :param obs: Observation object with calibrator_tracks populated
    :param beams: List of Beam (or BeamGauss) objects
    """

    def __init__(self, obs, beams):
        self.obs   = obs
        self.beams = beams
        self.result = None
        # Build interpolators once — beam patterns are constant across all passes
        self.interpolators = [b.get_Efield_interpolator() for b in beams]

    def simulate(self):
        """
        Run the simulation over all calibrator passes.

        :returns: List of complex arrays, one per pass, shape (NTime_p, NBeam, NFreq_p)
        :rtype: list[np.ndarray]
        """
        NBeam = len(self.beams)

        self.result = []
        for track in self.obs.calibrator_tracks:
            NTime = len(track)
            NFreq = len(track.tone_freqs)
            pass_result = np.zeros((NTime, NBeam, NFreq), dtype=complex)

            for ti in range(NTime):
                alt = track.alt[ti]
                az  = track.az[ti]
                pol = track.polarization[ti]

                for bi, (iEt, iEp) in enumerate(self.interpolators):
                    # Vectorize over all tone freqs at once
                    Et = iEt(alt, az, track.tone_freqs)   # shape (NFreq,)
                    Ep = iEp(alt, az, track.tone_freqs)   # shape (NFreq,)
                    pass_result[ti, bi, :] = (
                        track.tone_amplitude * (Et * np.cos(pol) + Ep * np.sin(pol))
                    )

            self.result.append(pass_result)
        return self.result

    def write_fits(self, out_file):
        """
        Write simulation results to a FITS file.

        Structure:

        - Primary HDU: header with obs metadata (NBEAMS, NPASSES, LONG, LAT)
        - For each pass p:

          - ``PASS{p}_FREQS``: 1-D array of NFreq_p tone frequencies (MHz)
          - ``PASS{p}_TIMES``: 1-D array of NTime_p times as MJD floats
          - ``PASS{p}_RE_{b}`` / ``PASS{p}_IM_{b}``: real and imaginary parts
            of shape (NTime_p, NFreq_p) for each beam b

        :param out_file: Output FITS path
        :type out_file: str
        """
        if self.result is None:
            raise RuntimeError("Call simulate() before write_fits()")

        primary = astrofits.PrimaryHDU()
        hdr = primary.header
        hdr['NBEAMS']  = len(self.beams)
        hdr['NPASSES'] = len(self.result)
        hdr['LONG']    = self.obs.lun_long_deg
        hdr['LAT']     = self.obs.lun_lat_deg

        hdulist = [primary]

        for p, (pass_result, track) in enumerate(zip(self.result, self.obs.calibrator_tracks)):
            NTime, NBeam, NFreq = pass_result.shape

            freq_hdu = astrofits.ImageHDU(track.tone_freqs, name=f'PASS{p}_FREQS')
            hdulist.append(freq_hdu)

            times_mjd = np.array([t.mjd for t in track.times])
            times_hdu = astrofits.ImageHDU(times_mjd, name=f'PASS{p}_TIMES')
            hdulist.append(times_hdu)

            for b in range(NBeam):
                re_hdu = astrofits.ImageHDU(
                    pass_result[:, b, :].real, name=f'PASS{p}_RE_{b}'
                )
                im_hdu = astrofits.ImageHDU(
                    pass_result[:, b, :].imag, name=f'PASS{p}_IM_{b}'
                )
                hdulist.append(re_hdu)
                hdulist.append(im_hdu)

        astrofits.HDUList(hdulist).writeto(out_file, overwrite=True)
