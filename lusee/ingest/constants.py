"""Constants and magic numbers for the LuSEE-Night downlink pipeline.

All values come from the CCSDS standard (CCSDS 133.0-B-2) or from the
LuSEE-Night flight-software conventions.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Stage 1: CCSDS framing
# ---------------------------------------------------------------------------

SYNC_WORD = 0xECA0           # 2-byte big-endian, immediately precedes a primary header
PADDING_BYTE = 0xA5          # discarded while seeking sync
PRIMARY_HEADER_LEN = 6       # CCSDS Space Packet primary header
CRC_LEN = 2                  # CRC-16-CCITT trailer appended outside the data field
CRC_INIT = 0xFFFF
CRC_POLY = 0x1021


# ---------------------------------------------------------------------------
# Bank layout
# ---------------------------------------------------------------------------

SCIENCE_BANKS = ("b05", "b06", "b07", "b08", "b09")
TELEMETRY_BANK = "b01"
BANK_FILENAME = "FFFFFFFE"   # on-disk filename inside each bN/ directory


# ---------------------------------------------------------------------------
# Telemetry sub-second tick (used for combining mission_seconds + lusee_subsecs
# into a continuous time axis; this is a pure unit conversion and is not
# DCB-specific).
# ---------------------------------------------------------------------------

LUSEE_SUBSEC_LSB = 1.0 / 65536.0


# ---------------------------------------------------------------------------
# Mission time
# ---------------------------------------------------------------------------

MISSION_TIME_FRACT_SHIFT = 4       # bottom bits dropped from (time_16<<32)|time_32
MISSION_TIME_FRACT_DIVISOR = 4096  # 12 fractional bits


# ---------------------------------------------------------------------------
# Logical-packet identity
# ---------------------------------------------------------------------------

# Class A: blob[0:4] is the unique_packet_id as little-endian uint32.
# Predicate is "appid is a spectrum / TR / zoom / cal-data / cal-pfb / cal-debug
# / Grimm" -- deferred to lusee.ingest.collation, which uses uncrater helpers.
# Class B: typed header carries a unique_packet_id field.
# Class C: no embedded id; inherit from the most recent preceding A/B packet.


# ---------------------------------------------------------------------------
# Output layout (HDF5)
# ---------------------------------------------------------------------------

NPRODUCTS = 16
NCHANNELS = 2048
WAVEFORM_SAMPLES = 16384
ZOOM_BINS = 64
ZOOM_COMPONENTS = 4

HDF5_LAYOUT_VERSION = 3
HDF5_DEFAULT_COMPRESSION = "gzip"
HDF5_DEFAULT_COMPRESSION_OPTS = 1

# Normal spectra are persisted in the same digital units used to fit the
# gain model.  Firmware transmits a bit-sliced accumulator word; restoring
# it to this reference is an ingestion invariant, not a calibration option.
BITSLICE_REFERENCE = 31
SPECTRA_UNITS = "SDU"
SPECTRA_REPRESENTATION = "gain_model_input_sdu"
SPECTRA_NORMALIZATION_VERSION = 1

DEFAULT_LUN_LAT_DEG = -15.0
DEFAULT_LUN_LONG_DEG = 175.0
DEFAULT_LUN_HEIGHT_M = 0.0
DEFAULT_RAW_TIME_SUBTRACT_SECONDS = 0.0
DEFAULT_MJD_EPOCH_OFFSET_DAYS = 0.0

# Time provenance recorded in output files. "unknown" is the honest
# default: the mission clock's relation to UTC/TAI has not been
# established at ingest time. Readers must refuse to guess -- see
# lusee.ingest.obs_factory.IngestData and its ``assume_scale`` argument.
KNOWN_TIME_SCALES = ("utc", "tai", "tt", "tdb", "unknown")
DEFAULT_TIME_SCALE = "unknown"
DEFAULT_CLOCK_SOURCE = "unknown"


# ---------------------------------------------------------------------------
# Filename conventions
# ---------------------------------------------------------------------------

FILENAME_PACKET_INDEX_DEFAULT_WIDTH = 5   # widen to 6 if a session has >= 1e6 packets
FILENAME_PACKET_INDEX_WIDE_THRESHOLD = 1_000_000
FILENAME_APID_HEX_WIDTH = 4

DEFAULT_SESSION_NAME_FMT = "session_{ord:03d}_{ts}"
SESSION_NAME_NO_TIME_FMT = "session_{ord:03d}"
SESSION_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
