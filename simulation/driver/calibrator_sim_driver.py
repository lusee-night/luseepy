#!/usr/bin/env python3
import os

import numpy as np


class CalibratorSimDriver(dict):
    def __init__(self, cfg):
        import lusee

        self._lusee = lusee
        self.update(cfg)
        self._parse_base()
        self._parse_beams()

    def _parse_base(self):
        self.root   = self["paths"]["lusee_drive_dir"]
        self.outdir = self["paths"].get("output_dir", ".")

        if isinstance(self.root, str) and self.root.startswith("$"):
            self.root = os.environ[self.root[1:]]
        if isinstance(self.outdir, str) and self.outdir.startswith("$"):
            self.outdir = os.environ[self.outdir[1:]]

        od = self["observation"]
        self.dt = od["dt"]
        if isinstance(self.dt, str):
            self.dt = eval(self.dt)

    def _parse_beams(self):
        lusee = self._lusee
        bdc        = self["beam_config"]
        beam_type  = bdc.get("type", "fits")
        beam_smooth = bdc.get("beam_smooth")
        taper      = bdc.get("taper", 0.03)

        beams = []

        if beam_type == "Gaussian":
            print("Creating Gaussian beams!")
            for b, cbeam in self["beams"].items():
                angle = bdc.get("common_beam_angle", 0) + cbeam.get("angle", 0)
                print(f"  Creating Gaussian beam {b}, angle={angle}")
                B = lusee.BeamGauss(
                    dec_deg=cbeam["declination"],
                    sigma_deg=cbeam["sigma"],
                    one_over_freq_scaling=cbeam.get("one_over_freq_scaling", False),
                    id=b,
                )
                B = B.rotate(angle)
                B.taper_and_smooth(taper=taper, beam_smooth=beam_smooth)
                beams.append(B)

        elif beam_type == "fits":
            broot = os.path.join(self.root, self["paths"]["beam_dir"])
            for b, cbeam in self["beams"].items():
                print(f"  Loading beam {b}:")
                filename = cbeam.get("file") or bdc.get("default_file")
                if filename is None:
                    raise ValueError(f"No beam file configured for beam {b}")
                fname = os.path.join(broot, filename)
                print(f"    {fname}")
                B = lusee.Beam(fname, id=b)
                angle = bdc.get("common_beam_angle", 0) + cbeam.get("angle", 0)
                print(f"    rotating: {angle}")
                B = B.rotate(angle)
                B.taper_and_smooth(taper=taper, beam_smooth=beam_smooth)
                beams.append(B)

        else:
            raise ValueError(f"Unknown beam_config.type={beam_type!r}")

        self.beams = beams

    def _build_calibrator_tracks(self, obs):
        lusee    = self._lusee
        sat_conf = self["satellite"]

        tone_freqs = np.array(sat_conf["tone_freqs"], dtype=float)
        amp_cfg    = sat_conf["tone_amplitude"]
        if np.isscalar(amp_cfg):
            tone_amplitude = np.full(len(tone_freqs), float(amp_cfg))
        else:
            tone_amplitude = np.array(amp_cfg, dtype=float)

        sat_kwargs = {}
        for key in ("semi_major_km", "eccentricity", "inclination_deg",
                    "raan_deg", "argument_of_pericenter_deg"):
            if key in sat_conf:
                sat_kwargs[key] = sat_conf[key]

        sat    = lusee.Satellite(**sat_kwargs)
        os_    = lusee.ObservedSatellite(obs, sat)
        passes = os_.get_transit_indices()

        tracks = []
        for si, ei in passes:
            n = ei - si
            if n == 0:
                continue
            tracks.append(lusee.CalibratorTrack(
                times         = obs.times[si:ei],
                alt           = os_.alt[si:ei],
                az            = os_.az[si:ei],
                polarization  = np.zeros(n),
                tone_freqs    = tone_freqs,
                tone_amplitude= tone_amplitude,
            ))

        return tracks

    def run(self):
        lusee = self._lusee
        od    = self["observation"]

        print("Creating Observation...")
        obs = lusee.Observation(
            od["lunar_day"],
            deltaT_sec   = self.dt,
            lun_lat_deg  = od["lat"],
            lun_long_deg = od["long"],
        )

        print("Computing satellite tracks...")
        obs.calibrator_tracks = self._build_calibrator_tracks(obs)
        print(f"  Found {len(obs.calibrator_tracks)} overpasses.")

        print("Running CalibratorSimulator...")
        sim = lusee.CalibratorSimulator(obs, self.beams)
        sim.simulate()

        out_base = self["simulation"].get("output", "calibrator_sim.fits")
        fname    = os.path.join(self.outdir, out_base)
        print(f"Writing to {fname}")
        sim.write_fits(fname)
