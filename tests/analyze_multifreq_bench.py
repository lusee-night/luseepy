"""Analyze bench_multifreq_full_scale.npz and produce the comparison story:
  - per-frequency rho_l curves (SVD vs full CG)
  - single-freq reference comparison at 25 MHz
  - summary metrics table
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp

d = np.load("/home/zack/luseepy/tests/bench_multifreq_full_scale.npz")
FREQS = d["freqs"]; LMAX = int(d["lmax"])
truth = d["truth_alm"]; alm_full = d["alm_full"]; alm_svd = d["alm_svd"]
alm_sf = d["alm_sf"]; ref_fi = int(d["ref_fi"])


def rho(true, rec):
    return hp.alm2cl(true, rec) / np.sqrt(hp.alm2cl(true) * hp.alm2cl(rec) + 1e-30)


def mr(r, a, b): return float(np.nanmean(r[a:b+1]))


print("=" * 72)
print(f"Timing: full CG {float(d['t_full']):.1f}s | SVD {float(d['t_svd']):.1f}s | single-freq {float(d['t_sf']):.1f}s")
print(f"SVD singular value ratios: {d['sv'][1]/d['sv'][0]:.3e}, {d['sv'][2]/d['sv'][0]:.3e}")
print("=" * 72)
print(f"\n-- At reference freq {FREQS[ref_fi]:.0f} MHz, mean rho over ell ranges --\n")
print(f"{'solver':<22}  {'1..10':>8}  {'1..20':>8}  {'1..32':>8}")
rho_sf  = rho(truth[ref_fi], alm_sf)
rho_ful = rho(truth[ref_fi], alm_full[ref_fi])
rho_svd = rho(truth[ref_fi], alm_svd[ref_fi])
for name, r in [("single-freq CG", rho_sf), ("10-freq full CG", rho_ful), ("10-freq SVD K=3", rho_svd)]:
    print(f"{name:<22}  {mr(r,1,10):>8.4f}  {mr(r,1,20):>8.4f}  {mr(r,1,LMAX):>8.4f}")

# Per-freq mean rho
print("\n-- Per-frequency mean rho(1..20): full CG vs SVD --\n")
print(f"{'MHz':>6}  {'full CG':>10}  {'SVD K=3':>10}  {'delta':>8}")
for fi, f in enumerate(FREQS):
    rf = rho(truth[fi], alm_full[fi]); rs = rho(truth[fi], alm_svd[fi])
    m_f = mr(rf, 1, 20); m_s = mr(rs, 1, 20)
    print(f"{f:>6.1f}  {m_f:>10.4f}  {m_s:>10.4f}  {m_s-m_f:>+8.4f}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(np.arange(LMAX+1)[1:], rho_sf[1:LMAX+1],  'k-',  lw=2, label=f'single-freq CG ({FREQS[ref_fi]:.0f} MHz)')
ax.plot(np.arange(LMAX+1)[1:], rho_ful[1:LMAX+1], 'b--', lw=2, label=f'10-freq full CG at {FREQS[ref_fi]:.0f} MHz')
ax.plot(np.arange(LMAX+1)[1:], rho_svd[1:LMAX+1], 'r-',  lw=2, label=f'10-freq SVD K=3 at {FREQS[ref_fi]:.0f} MHz')
ax.axhline(1, color='k', lw=0.5, ls=':')
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\rho_\ell$")
ax.set_title(f"Reconstruction fidelity at {FREQS[ref_fi]:.0f} MHz")
ax.legend(); ax.set_ylim(0.5, 1.02)

ax = axes[1]
for fi, f in enumerate(FREQS):
    r = rho(truth[fi], alm_svd[fi])
    ax.plot(np.arange(LMAX+1)[1:], r[1:LMAX+1], lw=1.5,
            color=plt.cm.viridis(fi/(len(FREQS)-1)), label=f"{f:.0f} MHz")
ax.axhline(1, color='k', lw=0.5, ls=':')
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\rho_\ell$")
ax.set_title("SVD K=3 per-frequency reconstruction")
ax.legend(ncol=2, fontsize=8); ax.set_ylim(0.5, 1.02)

plt.tight_layout()
plt.savefig("/home/zack/luseepy/tests/bench_multifreq_full_scale.png", dpi=120)
print("\nSaved plot: tests/bench_multifreq_full_scale.png")
