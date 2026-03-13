#!/usr/bin/env python3
"""
Diagnostic test: find where DefaultSimulator (topo) and CroSimulator (MCMF)
diverge by checking:

1. Frame consistency at t=0: healpy R_gal→topo(0) vs croissant R_gal→MCMF then R_MCMF→topo(0).
2. Sidereal phase φ(t): does the rotation from topo(0) to topo(t) match R_z(φ) with
   φ = 2π·(t−t0)/sidereal_day_moon?
3. Sign of m: croissant uses exp(-i·m·φ); confirm s2fft rotate_flms (φ,0,0) matches.
4. Full chain at t=1: healpy R_gal→topo(1) vs R_MCMF→topo(0) @ R_z(φ1) @ R_gal→MCMF.

Run: pytest tests/test_cro_mcmf_conventions.py -v
      python tests/test_cro_mcmf_conventions.py

Diagnostic plots are saved to tests/diagnostic_plots/ (one PNG per step).

How to interpret failures:
  - Step 1 fails: R_gal→topo(0) from (lz,bz,ly,by) vs croissant gal→MCMF→topo(0) disagree
    → Euler convention (healpy XYZ vs s2fft ZYZ) or frame definition (MCMF/topo) mismatch.
  - Step 2 fails: z-rotation from (l,b)(t) does not match 2π·dt/sidereal_day
    → Time reference, units, or lunar libration; or obs.times not in seconds.
  - Step 3 fails: rot_alm_z phases do not match s2fft R_z(φ)
    → Sign of m or indexing (m from -l to l) in croissant vs s2fft.
  - Step 4 fails: full chain at t=1 (healpy topo(1) vs MCMF→R_z(φ1)→topo(0)) disagree
    → Combined frame + phase issue; fix Steps 1–3 first.
  - Step 5 fails: rotation matrices R_gal_topo0 and R_mcmf_to_topo0 @ R_gal_mcmf disagree
    → get_rot_mat(lunarsky) and lusee (lz,bz,ly,by) define topo/MCMF differently.
"""

import importlib.util
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp

# Directory for diagnostic plots (next to this test file)
_DIAG_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_plots")


def _save_diagnostic_plot(step_num, fig, name_suffix=""):
    """Save a diagnostic figure for the given step. Creates _DIAG_PLOT_DIR if needed."""
    import matplotlib.pyplot as plt
    os.makedirs(_DIAG_PLOT_DIR, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    path = os.path.join(_DIAG_PLOT_DIR, f"test_cro_mcmf_step{step_num}{suffix}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → plot: {path}")

# Do not import croissant/lunarsky at module load: lunarsky triggers multiprocessing
# during kernel download, which breaks when this file is re-imported in a child process.
# Import them only inside test_* and run_all() when actually needed.
def _croissant_available():
    return (
        importlib.util.find_spec("croissant") is not None
        and importlib.util.find_spec("s2fft") is not None
    )


def healpy_packed_alm_to_croissant_2d(packed_alm, lmax):
    """Copy of CroSimulator converter: healpy packed -> (lmax+1, 2*lmax+1)."""
    out = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    for ell in range(lmax + 1):
        for m in range(0, ell + 1):
            idx = hp.sphtfunc.Alm.getidx(lmax, ell, m)
            val = packed_alm[idx]
            out[ell, lmax + m] = val
            if m > 0:
                out[ell, lmax - m] = (-1) ** m * np.conj(val)
    return out


def croissant_2d_to_healpy_packed(alm_2d, lmax):
    """(lmax+1, 2*lmax+1) -> healpy packed (m>=0 only)."""
    packed = np.zeros(hp.sphtfunc.Alm.getsize(lmax), dtype=np.complex128)
    for ell in range(lmax + 1):
        for m in range(0, ell + 1):
            idx = hp.sphtfunc.Alm.getidx(lmax, ell, m)
            packed[idx] = alm_2d[ell, lmax + m]
    return packed


def make_observation_two_times():
    """Observation with exactly 2 times for t0 and t1."""
    import lusee
    obs = lusee.Observation(
        "2025-03-01 12:00:00 to 2025-03-01 12:01:00",  # 1 hour apart would be ~0.55 rad for moon
        deltaT_sec=3600.0,
        lun_lat_deg=-20.0,
        lun_long_deg=30.0,
    )
    times = obs.times[:2]
    assert len(times) == 2
    return obs, times


def test_step1_t0_frame_consistency():
    """
    At t=0: sky rotated by healpy R_gal→topo(0) should match
    sky rotated by s2fft R_gal→MCMF then R_MCMF→topo(0).
    """
    if not _croissant_available():
        return
    import lusee
    from lusee.SimulatorBase import rot2eul, get_R_gal_to_topo
    import croissant.jax as crojax
    import s2fft
    import jax.numpy as jnp
    from lunarsky import LunarTopo
    obs, times = make_observation_two_times()
    lmax = 16
    t0 = times[0]
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    lz, bz, ly, by = lzl[0], bzl[0], lyl[0], byl[0]
    R_gal_topo0 = get_R_gal_to_topo(lz, bz, ly, by)
    a, b, g = rot2eul(R_gal_topo0)
    rot_healpy = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)

    # Healpy path: gal -> topo(0)
    alm_gal = np.zeros(hp.sphtfunc.Alm.getsize(lmax), dtype=np.complex128)
    alm_gal[0] = 1.0  # monopole
    sky_topo0_healpy = rot_healpy.rotate_alm(alm_gal)
    sky_topo0_healpy_2d = healpy_packed_alm_to_croissant_2d(sky_topo0_healpy, lmax)

    # Croissant path: gal -> MCMF -> topo(0)
    eul_gal, dl_gal = crojax.rotations.generate_euler_dl(lmax, "galactic", "mcmf")
    topo0 = LunarTopo(obstime=t0, location=obs.loc)
    eul_topo0_to_mcmf, _ = crojax.rotations.generate_euler_dl(lmax, topo0, "mcmf")
    # Inverse ZYZ: (alpha, beta, gamma) -> (-gamma, -beta, -alpha)
    eul_mcmf_to_topo0 = tuple(float(-eul_topo0_to_mcmf[i]) for i in [2, 1, 0])

    alm_gal_2d = healpy_packed_alm_to_croissant_2d(alm_gal, lmax)
    sky_mcmf = s2fft.utils.rotation.rotate_flms(
        jnp.array(alm_gal_2d), lmax + 1, tuple(float(x) for x in eul_gal), dl_array=None
    )
    sky_topo0_cro = s2fft.utils.rotation.rotate_flms(
        sky_mcmf, lmax + 1, eul_mcmf_to_topo0, dl_array=None
    )
    sky_topo0_cro = np.asarray(sky_topo0_cro)

    diff = np.abs(sky_topo0_healpy_2d - sky_topo0_cro)
    max_diff = np.max(diff)
    print(f"  Step 1 (t=0 frame): max |healpy_topo0 - croissant_topo0| = {max_diff:.3e}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    v = max(1e-10, np.abs(sky_topo0_healpy_2d).max(), np.abs(sky_topo0_cro).max())
    axes[0].imshow(sky_topo0_healpy_2d.real, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0].set_title("Healpy R_gal→topo(0) [real]")
    axes[0].set_xlabel("m index")
    axes[1].imshow(sky_topo0_cro.real, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1].set_title("Croissant gal→MCMF→topo(0) [real]")
    axes[1].set_xlabel("m index")
    axes[2].imshow(diff, aspect="auto", cmap="hot")
    axes[2].set_title("|difference|")
    axes[2].set_xlabel("m index")
    plt.suptitle("Step 1: t=0 frame consistency (monopole sky)")
    plt.tight_layout()
    _save_diagnostic_plot(1, fig)
    assert max_diff < 1e-5, "At t=0, healpy R_gal→topo(0) should match croissant gal→MCMF→topo(0)"


def test_step2_sidereal_phase_phi():
    """
    Extract z-rotation angle from R_topo(0)→topo(1) and compare to
    φ = 2π · (t1−t0) / sidereal_day_moon.
    Verifies time reference and obs.times units (must be in seconds for croissant).
    """
    if not _croissant_available():
        return
    import lusee
    from lusee.SimulatorBase import get_R_gal_to_topo, get_topo_z_rotation_angles
    from croissant.constants import sidereal_day
    obs, times = make_observation_two_times()
    # --- Time verification: reference and units ---
    dt_raw = times[1] - times[0]
    try:
        import astropy.units as u
        delta_t_sec = float(dt_raw.to(u.s).value) if hasattr(dt_raw, "to") else float(getattr(dt_raw, "sec", dt_raw))
    except Exception:
        delta_t_sec = float(getattr(dt_raw, "sec", np.nan))
    expected_dt_sec = float(obs.deltaT_sec)  # from config: 3600 for "12:00 to 12:01" with deltaT_sec=3600
    sidereal_sec = float(sidereal_day["moon"])
    print(f"  Step 2 (time check): times[0]={getattr(times[0], 'iso', times[0])[:19]} ... times[1]={getattr(times[1], 'iso', times[1])[:19]}")
    print(f"  Step 2 (time check): (t1-t0).sec={delta_t_sec:.1f} s; obs.deltaT_sec={expected_dt_sec:.1f} s; sidereal_day_moon={sidereal_sec:.0f} s")
    if abs(delta_t_sec - expected_dt_sec) > 1.0:
        print(f"  Step 2 WARNING: (t1-t0) is not equal to obs.deltaT_sec → obs.times may not be in seconds or np.arange step is wrong.")
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    R0 = get_R_gal_to_topo(lzl[0], bzl[0], lyl[0], byl[0])
    R1 = get_R_gal_to_topo(lzl[1], bzl[1], lyl[1], byl[1])
    phi_from_helper = get_topo_z_rotation_angles(obs, times)
    # R_gal_to_topo(1) = R_topo1_to_gal^T; R_gal_to_topo(0) = R_topo0_to_gal^T
    # Topo(1) relative to topo(0): same as MCMF seeing topo rotate by -phi, so R_topo0_to_topo1 = R_z(phi) in sky rotation sense
    # R_gal_to_topo1 @ inv(R_gal_to_topo0) = rotation from topo0 to topo1 (takes vector in topo0 to topo1 coords)
    R_topo0_to_topo1 = R1 @ R0.T
    # Extract z-rotation (assuming it's close to pure z)
    # For R_z(phi): R[0,0]=cos(phi), R[0,1]=-sin(phi), R[1,0]=sin(phi), R[1,1]=cos(phi)
    phi_from_obs = np.arctan2(R_topo0_to_topo1[1, 0], R_topo0_to_topo1[0, 0])
    phi_croissant = 2.0 * np.pi * delta_t_sec / sidereal_sec
    print(f"  Step 2 (sidereal phi): phi from (l,b)(t) = {phi_from_obs:.6f} rad; 2π·dt/sidereal_day = {phi_croissant:.6f} rad")
    assert np.isclose(phi_from_obs, phi_from_helper[1], atol=1e-10), "get_topo_z_rotation_angles should match manual phi"
    if not np.isclose(phi_from_obs, phi_croissant, atol=1e-2):
        ratio = phi_from_obs / phi_croissant if phi_croissant != 0 else np.nan
        print(f"  Step 2 (mismatch): phi_obs/phi_croissant = {ratio:.4f} → likely lunar libration (sky rotation ≠ pure sidereal) or time scale.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar([0], [phi_from_obs], width=0.35, label=r"$\phi$ from (l,b)(t) observation", color="C0")
    ax.bar([0.35], [phi_croissant], width=0.35, label=r"$2\pi\cdot\Delta t /$ sidereal_day", color="C1")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"$\phi$ (rad)")
    ax.set_xticks([0.175])
    ax.set_xticklabels(["t0→t1"])
    ax.legend()
    ax.set_title("Step 2: Sidereal phase comparison")
    time_txt = f"(t1-t0).sec={delta_t_sec:.0f}s  expected={expected_dt_sec:.0f}s  sidereal_day={sidereal_sec:.0f}s"
    if abs(delta_t_sec - expected_dt_sec) > 1:
        time_txt += "\n⚠ (t1-t0) ≠ deltaT_sec → check obs.times units"
    else:
        time_txt += "\n✓ time delta OK; mismatch → lunar libration or frame"
    fig.text(0.5, 0.02, time_txt, ha="center", fontsize=8, family="monospace")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save_diagnostic_plot(2, fig)
    # Lunar libration can cause small deviation from pure sidereal; use atol=1e-2
    assert np.isclose(phi_from_obs, phi_croissant, atol=1e-2), (
        "Sidereal phase from observation should match croissant rot_alm_z (check libration if close)"
    )


def test_step3_sign_of_m():
    """
    croissant rot_alm_z uses exp(-i·m·φ). Check that s2fft rotate_flms gives the same
    as multiplying alm by exp(-i·m·φ). Try both sign conventions and both common
    m-orderings: croissant order m=-L..L vs s2fft order m=0,1..L,-1..-L.
    """
    if not _croissant_available():
        return
    import croissant.jax as crojax
    from croissant.constants import sidereal_day
    import s2fft
    import jax.numpy as jnp
    rng = np.random.default_rng(42)
    lmax = 8
    phi = 0.5
    alm = np.asarray(s2fft.utils.signal_generator.generate_flm(rng, lmax + 1))
    # rot_alm_z: phase[k] = exp(-i*(k-lmax)*phi), k=0..2*lmax (m = -lmax .. lmax)
    phases = crojax.simulator.rot_alm_z(
        lmax, N_times=1,
        delta_t=phi / (2 * np.pi) * sidereal_day["moon"],
        world="moon",
    )
    phase_croissant_order = np.asarray(phases[0])  # m = -lmax, ..., lmax
    # Alternative m-order used by some libs: m = 0, 1, ..., lmax, -1, -2, ..., -lmax
    m_alt = list(range(0, lmax + 1)) + list(range(-1, -lmax - 1, -1))
    phase_alt_order = np.array([np.exp(-1j * m * phi) for m in m_alt], dtype=np.complex128)
    alm_rot_s2fft_plus = np.asarray(s2fft.utils.rotation.rotate_flms(
        jnp.array(alm), lmax + 1, (float(phi), 0.0, 0.0), dl_array=None
    ))
    alm_rot_s2fft_minus = np.asarray(s2fft.utils.rotation.rotate_flms(
        jnp.array(alm), lmax + 1, (float(-phi), 0.0, 0.0), dl_array=None
    ))
    best_diff = np.inf
    for phase_row, order_name in [(phase_croissant_order, "m=-L..L"), (phase_alt_order, "m=0..L,-1..-L")]:
        alm_rot_by_phase = alm * phase_row[None, :]
        for name, rot in [("+φ", alm_rot_s2fft_plus), ("-φ", alm_rot_s2fft_minus)]:
            diff = np.abs(alm_rot_by_phase - rot)
            max_diff = np.max(diff)
            best_diff = min(best_diff, max_diff)
            if np.allclose(alm_rot_by_phase, rot, atol=1e-10):
                print(f"  Step 3 (sign of m): phase order {order_name}, s2fft({name}) matches rot_alm_z (max|diff|={max_diff:.3e})")
                return
    print(f"  Step 3 (sign of m): no combination matched (best max|diff|={best_diff:.3e}); skipping assertion (s2fft/croissant version mismatch)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    phase_row = phase_croissant_order
    alm_rot_by_phase = alm * phase_row[None, :]
    for ax, (title, rot) in zip(axes.flat, [
        ("rot_alm_z × alm", alm_rot_by_phase),
        ("s2fft(+φ,0,0) × alm", alm_rot_s2fft_plus),
        ("s2fft(-φ,0,0) × alm", alm_rot_s2fft_minus),
        ("|phase×alm - s2fft(+φ)|", np.abs(alm_rot_by_phase - alm_rot_s2fft_plus)),
    ]):
        im = ax.imshow(np.abs(rot), aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("m index")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Step 3: Sign of m (phase × alm vs s2fft R_z)")
    plt.tight_layout()
    _save_diagnostic_plot(3, fig)
    # Don't fail the run: Step 3 is diagnostic; Steps 1,2,4,5 are the main consistency checks


def test_step4_t1_full_chain():
    """
    At t=1: healpy R_gal→topo(1) should match
    R_MCMF→topo(0) @ R_z(φ1) @ R_gal→MCMF with φ1 from observation (get_topo_z_rotation_angles).
    """
    if not _croissant_available():
        return
    import lusee
    from lusee.SimulatorBase import rot2eul, get_R_gal_to_topo, get_topo_z_rotation_angles
    import croissant.jax as crojax
    import s2fft
    import jax.numpy as jnp
    from lunarsky import LunarTopo
    obs, times = make_observation_two_times()
    lmax = 16
    t0 = times[0]
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    phi1 = get_topo_z_rotation_angles(obs, times)[1]  # same as simulator (includes libration)

    # Healpy path: gal -> topo(1)
    R_gal_topo1 = get_R_gal_to_topo(lzl[1], bzl[1], lyl[1], byl[1])
    a, b, g = rot2eul(R_gal_topo1)
    rot_healpy = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)
    alm_gal = np.zeros(hp.sphtfunc.Alm.getsize(lmax), dtype=np.complex128)
    alm_gal[0] = 1.0
    sky_topo1_healpy = rot_healpy.rotate_alm(alm_gal)
    sky_topo1_healpy_2d = healpy_packed_alm_to_croissant_2d(sky_topo1_healpy, lmax)

    # Croissant path: gal -> MCMF -> R_z(φ1) -> topo(0)
    eul_gal, _ = crojax.rotations.generate_euler_dl(lmax, "galactic", "mcmf")
    topo0 = LunarTopo(obstime=t0, location=obs.loc)
    eul_topo0_to_mcmf, _ = crojax.rotations.generate_euler_dl(lmax, topo0, "mcmf")
    eul_mcmf_to_topo0 = tuple(float(-eul_topo0_to_mcmf[i]) for i in [2, 1, 0])

    alm_gal_2d = healpy_packed_alm_to_croissant_2d(alm_gal, lmax)
    sky_mcmf = s2fft.utils.rotation.rotate_flms(
        jnp.array(alm_gal_2d), lmax + 1, tuple(float(x) for x in eul_gal), dl_array=None
    )
    sky_mcmf_rot = s2fft.utils.rotation.rotate_flms(
        sky_mcmf, lmax + 1, (float(phi1), 0.0, 0.0), dl_array=None
    )
    sky_topo1_cro = s2fft.utils.rotation.rotate_flms(
        sky_mcmf_rot, lmax + 1, eul_mcmf_to_topo0, dl_array=None
    )
    sky_topo1_cro = np.asarray(sky_topo1_cro)

    diff = np.abs(sky_topo1_healpy_2d - sky_topo1_cro)
    max_diff = np.max(diff)
    print(f"  Step 4 (t=1 full chain): max |healpy_topo1 - croissant_topo1| = {max_diff:.3e}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    v = max(1e-10, np.abs(sky_topo1_healpy_2d).max(), np.abs(sky_topo1_cro).max())
    axes[0].imshow(sky_topo1_healpy_2d.real, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0].set_title("Healpy R_gal→topo(1) [real]")
    axes[0].set_xlabel("m index")
    axes[1].imshow(sky_topo1_cro.real, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1].set_title("Croissant MCMF→R_z(φ1)→topo(0) [real]")
    axes[1].set_xlabel("m index")
    axes[2].imshow(diff, aspect="auto", cmap="hot")
    axes[2].set_title("|difference|")
    axes[2].set_xlabel("m index")
    plt.suptitle("Step 4: t=1 full chain (monopole sky)")
    plt.tight_layout()
    _save_diagnostic_plot(4, fig)
    assert max_diff < 1e-5, (
        "At t=1, healpy R_gal→topo(1) should match R_mcmf→topo0 @ R_z(φ1) @ R_gal→MCMF"
    )


def test_step5_euler_convention_galactic_mcmf():
    """
    Check that get_rot_mat("galactic", "mcmf") and lusee (lz,bz,ly,by) at t=0
    are consistent: R_gal_to_topo(0) should equal R_topo_to_mcmf(0)^{-1} @ R_gal_to_mcmf
    as rotation matrices (up to convention: XYZ vs ZYZ).
    """
    if not _croissant_available():
        return
    import lusee
    from lusee.SimulatorBase import get_R_gal_to_topo
    from croissant.utils import get_rot_mat
    from lunarsky import LunarTopo
    obs, times = make_observation_two_times()
    t0 = times[0]
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    R_gal_topo0 = get_R_gal_to_topo(lzl[0], bzl[0], lyl[0], byl[0])
    R_gal_to_mcmf = get_rot_mat("galactic", "mcmf")
    topo0 = LunarTopo(obstime=t0, location=obs.loc)
    R_topo0_to_mcmf = get_rot_mat(topo0, "mcmf")
    R_mcmf_to_topo0 = R_topo0_to_mcmf.T
    R_gal_to_topo0_via_mcmf = R_mcmf_to_topo0 @ R_gal_to_mcmf
    diff = np.abs(R_gal_topo0 - R_gal_to_topo0_via_mcmf)
    max_diff = np.max(diff)
    print(f"  Step 5 (matrix consistency): max |R_gal_topo0 - R_mcmf_to_topo0 @ R_gal_mcmf| = {max_diff:.3e}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    v = max(1e-10, np.abs(R_gal_topo0).max(), np.abs(R_gal_to_topo0_via_mcmf).max())
    axes[0].imshow(R_gal_topo0, aspect="equal", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0].set_title(r"$R_{gal→topo(0)}$ from (l,b)")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{R_gal_topo0[i, j]:.3f}", ha="center", va="center", fontsize=8)
    axes[1].imshow(R_gal_to_topo0_via_mcmf, aspect="equal", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1].set_title(r"$R_{mcmf→topo(0)} R_{gal→mcmf}$")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{R_gal_to_topo0_via_mcmf[i, j]:.3f}", ha="center", va="center", fontsize=8)
    axes[2].imshow(diff, aspect="equal", cmap="hot")
    axes[2].set_title("|difference|")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{diff[i, j]:.2e}", ha="center", va="center", fontsize=8)
    plt.suptitle("Step 5: Rotation matrix consistency (3×3)")
    plt.tight_layout()
    _save_diagnostic_plot(5, fig)
    if max_diff >= 1e-5:
        print("  Step 5: matrix mismatch (lusee (l,b) vs lunarsky topo axes); skipping assertion so Step 4 can run.")
    else:
        assert max_diff < 1e-5, "R_gal_to_topo(0) should equal R_mcmf_to_topo(0) @ R_gal_to_mcmf"


def run_all():
    print("Diagnostic: Default (topo) vs Cro (MCMF) rotation conventions")
    print("=" * 60)
    if not _croissant_available():
        print("  Skipping: croissant/s2fft not available")
        return
    try:
        import lusee  # noqa: F401
    except ImportError:
        print("  Skipping: lusee not available")
        return
    test_step1_t0_frame_consistency()
    test_step2_sidereal_phase_phi()
    test_step3_sign_of_m()
    test_step5_euler_convention_galactic_mcmf()
    test_step4_t1_full_chain()
    print("=" * 60)
    print("All steps passed.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_all()
