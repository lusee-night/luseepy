# Old-vs-new comparison: results (2026-08-04)

Sky-only ULSA runs, matched times/frequencies/lmax. The full analysis
apparatus — run scripts, the matched-configuration README, every figure
(PDF+PNG), the cached `.npz` simulation outputs the figures regenerate
from, and the status deck — is archived as `old_vs_new.tar.gz` on the
[2.0 GitHub release](https://github.com/lusee-night/luseepy/releases/tag/2.0).
The legacy side of the comparison ran `main @ 20a92e9` with the
now-deprecated scalar beams, so the archive is the durable record; the
old half is not expected to remain rerunnable.

## Headline: net change of the forecasting quantity (autos, V^2/Hz)

Over 5-45 MHz, time-averaged, mean over the four monopoles:

    median ratio new/old = 0.79   (-21%)
    range                = 0.35 .. 1.22

    5 MHz: 0.93   10 MHz: 0.87   20 MHz: 0.65
   30 MHz: 1.11   40 MHz: 0.87   50 MHz: 1.05

This sits inside the expected 20-50% band. The extreme excursions are
NOT broadband: they are confined to the ~30 MHz resonance flanks. The
new instrument's resonance is visibly sharper (higher Q) than the old
one; on the steep flanks a ~1 MHz change of resonance shape swings the
pointwise ratio from 1.11 (at the 30 MHz peak) to 0.38 (at 32-33 MHz)
and back to ~1.1 by 44 MHz. Away from the resonance the change is a
smooth -10..-40%.

## Decomposition: antenna/formalism vs receiver chain

The temperature-domain ratio (new T_equiv vs old T_ant, which cancels
each side's receiver conversion) shows the antenna-side change:

  - below ~15 MHz the new response couples only ~45% as much sky per
    unit dissipated power: the BGL_V16 regolith/lander model absorbs
    roughly twice the power into the Moon at low frequency compared to
    the old 8-layer-regolith single-antenna beams (see fsky.*: at
    10 MHz f_sky drops from ~0.22 to ~0.10; the curves converge above
    ~32 MHz and agree to ~10% there);
  - around 30-45 MHz the new response couples 5-17% MORE sky.

In the voltage domain these are partially compensated by the receiver
chain: the measured-JFET loading M = ZL(ZA+ZL)^-1 transfers about
twice the open-circuit power at low frequency compared to the legacy
Gamma_VD (35 pF + 1 MOhm) throughput model, pulling the low-frequency
net ratio back up to ~0.9.

Consequence worth noting for dark-ages forecasts: in the 10-25 MHz
window the sky-signal coupling (K-domain) is about HALF the old
prediction even though the V-domain totals moved only ~10-35%. Whether
science S/N moves accordingly depends on where the noise budget sits;
this deserves a dedicated pass with the moon term switched on.

## Crosses (expected to change - the old model was approximate)

The old pipeline modeled only opposite-pair couplings via 2-port files
(broadband negative Re for N-S/E-W, small L-pair signal). The full
four-port network instead concentrates cross power at the resonance:
L-pair Re flips sign and reaches ~-1.1e-16 V^2/Hz at 30 MHz (~10x the
old amplitude), while opposite pairs show a narrow bipolar feature.
Cross-based analyses calibrated on the old model should be revisited.

## Internal consistency checks

- In BOTH cases the Novaco-Brown 1978 monopole passed through the same
  instrument model tracks the simulated ULSA auto band across the full
  50 MHz (gal_sens_old/new) - each pipeline is self-consistent; the
  difference between them is instrument physics, not bookkeeping.
- Waterfall time structure (galaxy transit) matches between old and new
  at matched timestamps, supporting the coordinate/azimuth conventions
  of the new response ingestion.

## Caveats

- Sky-only (Tground = T_moon = 0), matching the legacy forecasting
  config. With emission on, the new pipeline attributes ALL non-sky
  dissipation (including low-frequency regolith absorption, i.e. most
  of the power below 15 MHz) to Rmoon at T_moon - the old code had no
  equivalent term with Tground=0.
- Old crosses used sign=-1 two-port couplings for opposite pairs only.
- lmax=32 both; dt=3600; single lunar day (2025-02).

## C4 symmetry breaking (actual vs group-averaged response, 2026-08-04)

Same ULSA sky-only run with the actual BGL_V16 response vs its C4
group average (circulant ZA, rotation-averaged patterns; see
beam_conversion/symmetrize_response.py and c4_* figures).

Autos, 5-45 MHz: median |dP/P| = 6.4%, 90th pct 21%, max 41%.
Crosses:         median |dC|/sqrt(PaPb) = 6.1%, 90th pct 16%, max 19%.

Structure, not noise: the difference waterfalls are nearly constant in
time (pure frequency bands) - the asymmetry is a stable instrument
signature, in principle calibratable. It is largest at 15-28 MHz
(port-to-port spread peaks at 60% near 21 MHz) and SMALLEST in the
resonance region (median 3.3% at 27-33 MHz): the 30 MHz resonance is a
property of the rods, while off-resonance the lander scattering shapes
the patterns. The breaking is strongly N-S anti-correlated (N +40%
where S -25% around 20 MHz) with mild E/W deviations, pointing at a
dominant asymmetry along the N-S axis of the lander layout.
