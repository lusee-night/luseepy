# %% [markdown]
# # Transit demos
#
# Track the Sun, Jupiter and a fixed RA/Dec source through different
# observation windows on the lunar far side using `lusee.Observation`.

# %%
import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import matplotlib.pyplot as plt

import lusee

# %% [markdown]
# ## Single lunar night (night 2500)

# %%
night = 2500
obs = lusee.Observation(night, deltaT_sec=3600)

# %%
alt_sun, _ = obs.get_track_solar("sun")
alt_jup, _ = obs.get_track_solar("jupiter")

plt.figure()
plt.plot(alt_sun, label="Sun")
plt.plot(alt_jup, label="Jupiter")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("hour into lunar night")
plt.ylabel("altitude [deg]")
plt.legend()
plt.title(f"Lunar night {night}")
plt.tight_layout()

# %% [markdown]
# ## Crab pulsar altitude during the same night

# %%
crab_ra, crab_dec = "05h34m31.94s", "+22d00m52.2s"
alt_crab, _ = obs.get_track_ra_dec(ra=crab_ra, dec=crab_dec)

plt.figure()
plt.plot(alt_crab)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("hour into lunar night")
plt.ylabel("altitude [deg]")
plt.title("Crab pulsar")
plt.tight_layout()

# %% [markdown]
# ## Examples of time-range specifications

# %%
obs = lusee.Observation("CY25", deltaT_sec=3600 * 24)
alt, _ = obs.get_track_solar("sun")
plt.figure(); plt.plot(alt)
plt.title("Sun altitude — CY25 (1-day cadence)")
plt.xlabel("day"); plt.ylabel("altitude [deg]")
plt.axhline(0, color="k", lw=0.5)

# %%
obs = lusee.Observation("FY2026", deltaT_sec=3600 * 24)
alt, _ = obs.get_track_solar("sun")
plt.figure(); plt.plot(alt)
plt.title("Sun altitude — FY2026 (1-day cadence)")
plt.xlabel("day"); plt.ylabel("altitude [deg]")
plt.axhline(0, color="k", lw=0.5)

# %%
obs = lusee.Observation(
    "2025-02-01 13:00:00 to 2025-04-01 16:00:00", deltaT_sec=3600 * 24
)
alt, _ = obs.get_track_solar("sun")
plt.figure(); plt.plot(alt)
plt.title("Sun altitude — explicit UTC range")
plt.xlabel("day"); plt.ylabel("altitude [deg]")
plt.axhline(0, color="k", lw=0.5)
