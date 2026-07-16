import os


_LUNARSKY_MOON_FRAME_READY = False


def ensure_lunarsky_moon_frame():
    """Load lunarsky's Moon frame kernels into spiceypy's global kernel pool.

    croissant uses spiceypy directly for frame transforms such as ``MOON_ME``,
    while lunarsky manages its own kernel pool internally. A normal Python
    process therefore needs to furnish the Moon frame kernels into spiceypy's
    process-global pool before Croissant simulations can run.
    """
    global _LUNARSKY_MOON_FRAME_READY
    if _LUNARSKY_MOON_FRAME_READY:
        return

    import lunarsky
    import spiceypy as spice

    if spice.namfrm("MOON_ME") != 0:
        _LUNARSKY_MOON_FRAME_READY = True
        return

    data_dir = os.path.join(os.path.dirname(lunarsky.__file__), "data")
    needed = (
        "pck/moon_pa_de421_1900-2050.bpc",
        "fk/satellites/moon_080317.tf",
    )

    missing = []
    for relpath in needed:
        path = os.path.join(data_dir, relpath)
        if os.path.isfile(path):
            spice.furnsh(path)
        else:
            missing.append(path)

    if spice.namfrm("MOON_ME") == 0:
        missing_text = ", ".join(missing) if missing else "unknown"
        raise RuntimeError(
            "CroSimulator could not furnish the MOON_ME frame from lunarsky "
            f"data. Missing kernel paths: {missing_text}"
        )

    _LUNARSKY_MOON_FRAME_READY = True
