# %% [markdown]
# # `grid2healpix` demo
#
# Project a (theta, phi) grid image onto a healpix map. Compares the
# slow reference path against the fast variant, then projects a real
# beam's `power_hp` onto a healpix map.
#
# Requires `LUSEE_DRIVE_DIR` for the beam example at the bottom.

# %%
import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

import lusee
from lusee import grid2healpix

# %%
theta_deg = np.arange(91) * 2
phi_deg = np.arange(180) * 2
Ntheta = len(theta_deg)
Nphi = len(phi_deg)
theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)
img = np.zeros((Ntheta, Nphi))

img[51, 44] = 1.0
plt.imshow(img, interpolation="nearest")
plt.title("input grid: single pixel")
Nside = 128

# %%
mp = grid2healpix(theta, phi, img, 60, Nside, fast=False)
hp.mollview(mp, title="grid2healpix(fast=False), single pixel")

# %%
mp = grid2healpix(theta, phi, img, 60, Nside, fast=True)
hp.mollview(mp, title="grid2healpix(fast=True), single pixel")

# %%
img = np.zeros_like(img)
# draw a rectangle outline
img[30:32, 80:150] = 1.0
img[60:62, 80:150] = 1.0
img[30:62, 80:82] = 1.0
img[30:62, 148:150] = 1.0
plt.imshow(img)
plt.title("input grid: rectangle outline")

# %%
hp.mollview(grid2healpix(theta, phi, img, 128, 1024, fast=True),
            title="grid2healpix of rectangle outline")

# %% [markdown]
# ## Real beam example
#
# Load a beam FITS file from the LuSEE drive checkout and visualize its
# power pattern at one frequency on a healpix map.

# %%
drive = os.environ.get("LUSEE_DRIVE_DIR")
beam_path = (
    f"{drive}/Simulations/BeamModels/LanderFreeSpaceComparison/hfss_lbl_1m_75.fits"
    if drive else None
)
if beam_path and os.path.exists(beam_path):
    B = lusee.Beam(beam_path)
    hp.mollview(B.power_hp(50, Nside, freq_ndx=44),
                title=f"Beam power_hp at freq idx 44 ({beam_path.split('/')[-1]})")
else:
    print("LUSEE_DRIVE_DIR or beam file not available; skipping real-beam panel.")
