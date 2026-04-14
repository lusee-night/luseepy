"""Diagnose why SVD K=4 underperforms single-freq CG at low frequencies.

Loads bench_multifreq_full_scale.npz (truth, alm_svd K=3), bench_svd_ksweep.npz
(alm K=4), bench_singlefreq_allbands.npz (10 independent SF solves).

For each freq, compares rho_ell(single-freq) vs rho_ell(SVD K=4) to locate
the ell range where SVD loses. If the gap is at high ell -> SVD has a
high-ell regularization advantage that single-freq lacks (weird given low band
SVD loses). If the gap is at low ell -> SVD prior is over-regularizing the
bright modes (makes sense at bright foreground bands).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dmf = np.load("/home/zack/luseepy/tests/bench_multifreq_full_scale.npz")
dks = np.load("/home/zack/luseepy/tests/bench_svd_ksweep.npz")
dsf = np.load("/home/zack/luseepy/tests/bench_singlefreq_allbands.npz")

FREQS = dmf["freqs"]
LMAX = int(dmf["lmax"])
truth = dmf["truth_alm"]
alm_svd_K4 = dks["rec_K4"]
alm_sf = dsf["alm_sf"]


def rho(a, b):
    return hp.alm2cl(np.asarray(a), np.asarray(b)) / np.sqrt(
        hp.alm2cl(np.asarray(a)) * hp.alm2cl(np.asarray(b)) + 1e-30)


print(f"{'MHz':>5}  {'SF 1..5':>7}  {'SF 6..15':>8}  {'SF 16..32':>9}  "
      f"{'SVD 1..5':>8}  {'SVD 6..15':>9}  {'SVD 16..32':>10}  "
      f"{'del lo':>7}  {'del md':>7}  {'del hi':>7}")
for fi, f in enumerate(FREQS):
    rs = rho(truth[fi], alm_sf[fi])
    rv = rho(truth[fi], alm_svd_K4[fi])
    s_lo = float(np.nanmean(rs[1:6]))
    s_md = float(np.nanmean(rs[6:16]))
    s_hi = float(np.nanmean(rs[16:LMAX + 1]))
    v_lo = float(np.nanmean(rv[1:6]))
    v_md = float(np.nanmean(rv[6:16]))
    v_hi = float(np.nanmean(rv[16:LMAX + 1]))
    print(f"{f:>5.1f}  {s_lo:>7.4f}  {s_md:>8.4f}  {s_hi:>9.4f}  "
          f"{v_lo:>8.4f}  {v_md:>9.4f}  {v_hi:>10.4f}  "
          f"{v_lo - s_lo:>+7.4f}  {v_md - s_md:>+7.4f}  {v_hi - s_hi:>+7.4f}")

# Plot per-freq
fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True, sharey=True)
ells = np.arange(LMAX + 1)
for fi, ax in enumerate(axes.flat):
    rs = rho(truth[fi], alm_sf[fi])
    rv = rho(truth[fi], alm_svd_K4[fi])
    ax.plot(ells[1:LMAX + 1], rs[1:LMAX + 1], 'k-', lw=1.5, label='single-freq CG')
    ax.plot(ells[1:LMAX + 1], rv[1:LMAX + 1], 'r-', lw=1.5, label='SVD K=4')
    ax.axhline(1, color='k', lw=0.5, ls=':')
    ax.set_title(f"{FREQS[fi]:.0f} MHz")
    ax.set_ylim(0, 1.05)
    if fi == 0:
        ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel(r'$\ell$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\rho_\ell$')
plt.tight_layout()
plt.savefig("/home/zack/luseepy/tests/diagnose_svd_lowband.png", dpi=120)
print("saved tests/diagnose_svd_lowband.png")
