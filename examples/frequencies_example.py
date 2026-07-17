"""
Working with arbitrary frequency grids.

Since the off-grid interpolation update, simulators accept any target
frequencies inside the beams' native range -- not just the canonical
1-50 MHz integer grid.  Off-grid targets are linearly interpolated from
the bracketing native bins; on-grid targets snap exactly (alpha = 0.0,
bit-identical to plain indexing).  Closed-form monopole skies evaluate
their spectra exactly at any frequency, with no interpolation at all.

Runs without LUSEE_DRIVE_DIR (analytic BeamGauss + point-source sky).
"""

import os
# for double-precision simulation values; frequency snapping itself does
# not need x64 (native grids are host-side numpy float64 regardless)
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import lusee
from lusee.frequencies import FrequencyMap, frequencies_from_config

# ── 1. YAML 'freq:' config forms ─────────────────────────────────────

# explicit values, any spacing
print(frequencies_from_config({"values": [10.0, 17.5, 30.25]}))

# arange semantics: end is EXCLUSIVE (1, 2, ..., 49)
print(frequencies_from_config({"start": 1.0, "end": 50.0, "step": 1.0}))

# linspace semantics: end inclusive, n points (the full canonical grid)
print(frequencies_from_config({"start": 1.0, "end": 50.0, "n": 50}))

# ── 2. FrequencyMap: snapping and interpolation weights ──────────────

native = np.linspace(1.0, 50.0, 50)          # a beam's native grid
target = np.asarray([12.0, 12.5, 30.0 + 1e-10])
fmap = FrequencyMap.build(target, native)

print(fmap)
print(fmap.alpha)          # [0.0, 0.5, 0.0]: on-grid targets snap exactly

# cheap per-native-bin arrays (gains, impedances) interpolate directly
gain = np.cos(native)
print(fmap.from_native(gain))

# expensive products (beam/sky alms) are evaluated only at the unique
# bracketing native bins and expanded with fmap.from_unique(...)
print(fmap.source_indices)

# out-of-range targets fail loudly
try:
    FrequencyMap.build([55.0], native)
except ValueError as e:
    print(f"rejected: {e}")

# ── 3. Simulating at off-grid frequencies ────────────────────────────

obs = lusee.Observation("2025-03-01 00:00:00 to 2025-03-01 02:00:00",
                        deltaT_sec=3600.0, lun_lat_deg=0.0, lun_long_deg=0.0)
lmax = 8

# BeamGauss's native grid defaults to 1-50 MHz / 50 bins; use the
# freq_min/freq_max/Nfreq kwargs for an extended band (e.g. up to 75 MHz)
beam = lusee.BeamGauss(alt_deg=90.0, az_deg=0.0, sigma_deg=20.0,
                       one_over_freq_scaling=False, id="ex")

# the sky carries its own native grid; it need not match the beam's
sky = lusee.sky.HarmonicPointSourceSky(
    lmax=lmax, l_deg=0.0, b_deg=0.0,
    freq=np.asarray([5.0, 15.0, 25.0]), T=np.asarray([1.0, 3.0, 5.0]))

freq = np.asarray([10.0, 12.5, 20.0])        # 12.5 MHz is off-grid for the beam
sim = lusee.TopoNumpySimulator(
    obs, [lusee.NpWrapper(beam)], lusee.NpWrapper(sky),
    Tground=0.0, combinations=[(0, 0)], freq=freq, lmax=lmax)
wf = np.asarray(sim.simulate(times=obs.times))
print(wf.shape)                               # (n_times, n_combinations, n_freq)

# ── 4. Closed-form monopole skies: exact at any frequency ────────────

# ConstSkyCane1979 and DarkAgesMonopole implement get_alm_at_freq, so
# simulators evaluate their spectra exactly instead of interpolating
cane = lusee.sky.ConstSkyCane1979(16, lmax=lmax)
alm = np.asarray(cane.get_alm_at_freq(np.asarray([13.37, 42.0])))
print(alm.shape)                              # (n_freq, n_alm)
