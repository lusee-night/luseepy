# Concordance measurement of preamp properties.
# See lusee-night/impedance_measurements for the raw data and analysis.
#
# Preamp input impedance model parameters.
#
#   Re Z = 1/(w^2 Rp C^2) + Rs + a w^n,   Im Z = w L - 1/(w C)
#
# Units: C_pf [pF], L_nh [nH], Rs_ohm [Ohm], a [Ohm s^n], Rp_ohm [Ohm].

# Measured fit parameters for the four spare flight-model preamps
# (fmpre0/2/5/7, on the BNL flatsat), from the JFET impedance Bode-sweep
# analysis, see BODE500_BNL/2026_06_01_LuSEENight_JFETAnalysis.pdf
# (table "Fitted Parameters - All 4 Preamps").  Re-fit with
# fits/fit_bode_spares.py: C and L reproduce to 0.2 pF / 2 nH; the Re Z
# parameters (Rs, a, n, Rp) are strongly degenerate and weighting-dependent.
#
# Note: fmpre7 L_nh=38.5 not 8.5 as in the HFSS notebook typo.
#
FLATSAT_CHANNEL_TO_PREAMP = {0: "fmpre0", 1: "fmpre2", 2: "fmpre5", 3: "fmpre7"}
SPARE_PREAMP_FITS = {
    "fmpre0": {"C_pf": 33.79, "L_nh": 35.0, "Rs_ohm": 0.21, "n": 1.39,
               "a": 5.2e-12, "Rp_ohm": 1101e3},
    "fmpre2": {"C_pf": 35.32, "L_nh": 36.4, "Rs_ohm": 2.16, "n": 2.26,
               "a": 6.9e-20, "Rp_ohm": 331e3},
    "fmpre5": {"C_pf": 34.89, "L_nh": 34.7, "Rs_ohm": 3.71, "n": 4.35,
               "a": 2.2e-38, "Rp_ohm": 288e3},
    "fmpre7": {"C_pf": 36.85, "L_nh": 38.5, "Rs_ohm": 1.65, "n": 1.88,
               "a": 1.9e-16, "Rp_ohm": 375e3},
}

# Flight preamp id -> ADC channel (see ../README.md, "Flight model connections").
# The LCX200 data file labels preamps 0..3; these are taken to be CH0..CH3.
FLIGHT_CHANNEL_TO_PREAMP = {0: "fmpre6", 1: "fmpre3", 2: "fmpre4", 3: "fmpre1"}

# Flight preamps, in the flight configuration at SDL, measured with an LCX200
# LCR meter at 100/180/320/560/1000 kHz (LCX200_SSL_SDL/data/ZJFET_250930.csv).
# Produced by fits/fit_lcx200_flight.py:
#   * C_pf is from the 1 MHz point only, C = 1/(w |Z|).  The lower-frequency
#     LCX200 points give a steadily larger apparent C (up to 44-47 pF at
#     100 kHz) and a phase of -100 deg at every frequency (negative Re Z),
#     whereas the Bode data on the spares show a flat C and -90 deg phase over
#     the same band; the LCX200 was on its manual 3 kOhm range, so the 1 MHz
#     point (|Z| = 4.8 kOhm) is the only one near range.  Fitting all five
#     frequencies instead gives 33.4 / 37.7 / 35.6 / 35.3 pF for CH0..CH3 with
#     chi^2/dof > 400.  Statistical error on the 1 MHz values is < 0.11 pF.
#   * L, Rs, Rp, a, n are averages over the four spares: L and Rs arithmetic
#     means, Rp the harmonic mean (averages the 1/Rp contribution to Re Z),
#     and (a, n) from a power-law refit to the mean of the four a w^n curves
#     over 1-100 MHz (the individual (a, n) pairs cannot be averaged term by
#     term).
FLIGHT_PREAMP_FITS = {
    "fmpre6": {"C_pf": 33.12, "L_nh": 36.2, "Rs_ohm": 1.93, "n": 1.56,
               "a": 8.1e-14, "Rp_ohm": 397e3},  # CH0 (J4)
    "fmpre3": {"C_pf": 33.29, "L_nh": 36.2, "Rs_ohm": 1.93, "n": 1.56,
               "a": 8.1e-14, "Rp_ohm": 397e3},  # CH1 (J5)
    "fmpre4": {"C_pf": 33.44, "L_nh": 36.2, "Rs_ohm": 1.93, "n": 1.56,
               "a": 8.1e-14, "Rp_ohm": 397e3},  # CH2 (J6)
    "fmpre1": {"C_pf": 33.17, "L_nh": 36.2, "Rs_ohm": 1.93, "n": 1.56,
               "a": 8.1e-14, "Rp_ohm": 397e3},  # CH3 (J7)
}
