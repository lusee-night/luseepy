#!/usr/bin/env python3
import os

import numpy as np


class SimDriver(dict):
    def __init__(self, cfg):
        self.update(cfg)
        self._configure_jax_precision()
        import lusee
        self._lusee = lusee
        self._parse_base()
        self._parse_sky()
        self._parse_beams()

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _configure_jax_precision(self):
        sim_cfg = self.get("simulation", {})
        if "jax_enable_x64" not in sim_cfg:
            return
        enabled = self._to_bool(sim_cfg.get("jax_enable_x64", False))
        os.environ["JAX_ENABLE_X64"] = "1" if enabled else "0"
        import jax
        jax.config.update("jax_enable_x64", enabled)
        print(f"JAX x64 precision enabled: {enabled}")

    def _parse_base(self):
        self.lmax = self["observation"]["lmax"]
        self.root = self["paths"]["lusee_drive_dir"]
        self.outdir = self["paths"].get("output_dir", ".")
        if isinstance(self.root, str) and self.root.startswith("$"):
            self.root = os.environ[self.root[1:]]
        if isinstance(self.outdir, str) and self.outdir.startswith("$"):
            self.outdir = os.environ[self.outdir[1:]]

        od = self["observation"]
        self.dt = od["dt"]
        if isinstance(self.dt, str):
            self.dt = eval(self.dt)
        self.freq = np.arange(od["freq"]["start"], od["freq"]["end"], od["freq"]["step"])

    def _parse_sky(self):
        lusee = self._lusee
        sky_type = self["sky"].get("type", "file")
        if sky_type == "file":
            fname = os.path.join(self.root, self["paths"]["sky_dir"], self["sky"]["file"])
            print("Loading sky: ", fname)
            self.sky = lusee.sky.FitsSky(fname, lmax=self.lmax)
        elif sky_type == "CMB":
            print("Using CMB sky")
            self.sky = lusee.sky.ConstSky(self.lmax, lmax=self.lmax, T=2.73, freq=self.freq)
        elif sky_type == "Cane1979":
            print("Using Cane1979 sky")
            self.sky = lusee.sky.ConstSkyCane1979(self.lmax, lmax=self.lmax, freq=self.freq)
        elif sky_type == "DarkAges":
            d = self["sky"]
            scaled = d.get("scaled", True)
            nu_min = d.get("nu_min", 16.4)
            nu_rms = d.get("nu_rms", 14.0)
            A = d.get("A", 0.04)
            print(
                f"Using Dark Ages Monopole sky scaled={scaled}, min={nu_min} MHz, "
                f"rms={nu_rms}MHz,A={A}K"
            )
            self.sky = lusee.sky.DarkAgesMonopole(
                self.lmax,
                lmax=self.lmax,
                freq=self.freq,
                nu_min=nu_min,
                nu_rms=nu_rms,
                A=A,
            )
        else:
            raise ValueError(f"Unknown sky.type={sky_type!r}")

    def _parse_beams(self):
        lusee = self._lusee
        broot = os.path.join(self.root, self["paths"]["beam_dir"])
        beams = []
        bd = self["beams"]
        bdc = self["beam_config"]
        couplings = bdc.get("couplings")
        beam_type = bdc.get("type", "fits")
        beam_smooth = bdc.get("beam_smooth")
        taper = bdc.get("taper", self.get("simulation", {}).get("taper", 0.03))

        if beam_type == "Gaussian":
            print("Creating Gaussian beams!")
            for b in self["observation"]["beams"]:
                cbeam = bd[b]
                print("Creating gaussian beam", b, ":")
                B = lusee.BeamGauss(
                    dec_deg=cbeam["declination"],
                    sigma_deg=cbeam["sigma"],
                    one_over_freq_scaling=cbeam["one_over_freq_scaling"],
                    id=b,
                )
                angle = bdc["common_beam_angle"] + cbeam["angle"]
                print("  rotating: ", angle)
                B = B.rotate(angle)
                B.taper_and_smooth(taper=taper, beam_smooth=beam_smooth)
                beams.append(B)
        elif beam_type == "fits":
            for b in self["observation"]["beams"]:
                print("Loading beam", b, ":")
                cbeam = bd[b]
                filename = cbeam.get("file")
                if filename is None:
                    filename = bdc.get("default_file")
                    if filename is None:
                        raise ValueError(f"No beam file configured for beam {b}")
                fname = os.path.join(broot, filename)
                print("  loading file: ", fname)
                B = lusee.Beam(fname, id=b)
                angle = bdc["common_beam_angle"] + cbeam.get("angle", 0)
                print("  rotating: ", angle)
                B = B.rotate(angle)
                B.taper_and_smooth(taper=taper, beam_smooth=beam_smooth)
                beams.append(B)
        else:
            raise ValueError(f"Unknown beam_config.type={beam_type!r}")

        self.beams = beams
        self.Nbeams = len(self.beams)
        if couplings is not None:
            for c in couplings:
                couplings[c]["two_port"] = os.path.join(broot, couplings[c]["two_port"])
            self.couplings = lusee.BeamCouplings(beams, from_yaml_dict=couplings)
        else:
            self.couplings = None

    @staticmethod
    def _normalize_engine(cfg):
        engine = cfg.get("engine")
        if engine is None:
            engine = cfg.get("simulation", {}).get("engine", "luseepy")
        e = str(engine).strip().lower()
        aliases = {
            "default": "luseepy",
            "lusee": "luseepy",
            "jax": "luseepy",
            "numpy": "numpy",
            "croissant": "croissant",
        }
        return aliases.get(e, e)

    def run(self):
        lusee = self._lusee
        print("Starting simulation:")
        od = self["observation"]
        O = lusee.Observation(
            od["lunar_day"],
            deltaT_sec=self.dt,
            lun_lat_deg=od["lat"],
            lun_long_deg=od["long"],
        )
        print(
            f"  Using observation: lat={O.lun_lat_deg} deg, lon={O.lun_long_deg} deg, "
            f"time_range={O.time_range}, N_times={len(O.times)}"
        )
        print("  setting up combinations...")
        combs = od["combinations"]
        if isinstance(combs, str) and combs == "all":
            combs = []
            for i in range(self.Nbeams):
                for j in range(i, self.Nbeams):
                    combs.append((i, j))

        engine = self._normalize_engine(self)
        if engine == "croissant":
            if lusee.CroSimulator is None:
                raise RuntimeError(
                    "CroSimulator requires optional dependency 'croissant' (and s2fft). "
                    "Install with: pip install croissant s2fft"
                )
            print("  setting up Croissant Simulation object...")
            S = lusee.CroSimulator(
                O,
                self.beams,
                self.sky,
                Tground=od["Tground"],
                combinations=combs,
                freq=self.freq,
                lmax=self.lmax,
                cross_power=self.couplings,
                extra_opts=self["simulation"],
            )
        elif engine == "luseepy":
            print("  setting up Default Simulation object...")
            S = lusee.DefaultSimulator(
                O,
                self.beams,
                self.sky,
                Tground=od["Tground"],
                combinations=combs,
                freq=self.freq,
                lmax=self.lmax,
                cross_power=self.couplings,
                extra_opts=self["simulation"],
            )
        elif engine == "numpy":
            print("  setting up NumPy Simulation object...")
            S = lusee.NumpySimulator(
                O,
                self.beams,
                self.sky,
                Tground=od["Tground"],
                combinations=combs,
                freq=self.freq,
                lmax=self.lmax,
                cross_power=self.couplings,
                extra_opts=self["simulation"],
            )
        else:
            raise ValueError(
                "engine must be one of {luseepy, numpy, croissant} (or aliases default/lusee/jax), "
                f"got: {engine}"
            )

        print(
            f"  Simulating {len(O.times)} timesteps (from observation) x {len(combs)} "
            f"data products x {len(self.freq)} frequency bins..."
        )
        print("  Simulating...")
        S.simulate(times=O.times)

        out_base = self["simulation"].get("output", f"sim_{engine}_output.fits")
        fname = os.path.join(self.outdir, out_base)
        print("Writing to", fname)
        S.write_fits(fname)
