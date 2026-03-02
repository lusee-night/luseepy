# CCSDS to HDF5/FITS pipeline overview

This document describes how the `receive/` pipeline converts raw CCSDS telemetry files into:

1. session directories containing reconstructed packet blobs (`cdi_output_*`)
2. final session products (`.h5` and `.fits`)

It focuses on *how CCSDS parsing/reconstruction works* and how to consume output in a user-friendly way.  
For detailed HDF5 layout, see [`receive/hdf5.md`](receive/hdf5.md).

## 1) End-to-end flow

Primary entry point: `receive/reconstruct_packets.py`

There are two operating modes:

- `--flash-path <FLASH_TLMFS>`:
  - decode raw flash files (`b05..b09/FFFFFFFE`) into collated packets
  - split packets into sessions
  - write per-session `.bin` packet files into `cdi_output_<session_start_seconds>/`
  - decode DCB telemetry from `b01/FFFFFFFE` and align telemetry to sessions
  - write session `.h5` and `.fits` files
- `--session-dirs <cdi_output_...> ...`:
  - skip raw CCSDS decode and read existing session directories
  - directly write `.h5` and `.fits`

Core code path in flash mode:

1. `decode_directory()`  
2. `split_into_sessions()`  
3. `decode_telemetry_directory()`  
4. `write_sessions()`  
5. `save_to_hdf5()` and `save_to_fits()`

## 2) Raw CCSDS parsing (`receive/low_level.py`)

### 2.1 Stream framing and separators

`L0_to_ccsds(data)` scans each raw byte stream with a state machine:

- `FINDING_SYNC`: searches for sync word `0xECA0`
- `READING_LEN`: reads the 6-byte CCSDS primary header
- `READING_DATA`: reads packet data until expected length, then validates CRC

Important framing details:

- `0xA5` bytes are treated as padding and ignored while seeking sync.
- Synchronization is on a two-byte marker (`0xEC`, `0xA0`).
- A garbage-byte counter is tracked for diagnostics (`garb`).

### 2.2 CCSDS primary header parsing

`decode_ccsds_header(pkt[:6])` extracts:

- `version` (3 bits)
- `packet_type` (1 bit)
- `secheaderflag` (1 bit)
- `appid` (11 bits)
- `groupflags` (2 bits)
- `sequence_cnt` (14 bits)
- `packetlen` (16 bits)

### 2.3 Length and CRC handling

After reading header, parser reads `packetlen + 3` bytes of data field, then:

- CCSDS convention is used here: `packetlen` is stored as `(remaining packet bytes) - 1`.
  - So `packetlen + 1` is the nominal CCSDS post-header byte count.
  - This stream carries an additional trailing 2-byte CRC, so the parser reads `packetlen + 3` and then splits off CRC.
  - This is intentional and not an off-by-one bug.
- treats the final 2 bytes as transmitted CRC (`pktcrc`)
- computes CRC-16-CCITT over `header + body_without_crc`
  - polynomial `0x1021`
  - initial value `0xFFFF`
  - big-endian bit processing
- aborts on mismatch (`assert False`)

Returned packet tuple is:

`(sequence_cnt, pkt_head, pkt_body_without_crc)`

### 2.4 Multi-packet collation

`collate_packets(pkts)` merges CCSDS fragments into logical packets (`CollatedPacket`) using `groupflags`.

- Payload bytes are reordered in 16-bit lanes via `reorder()`:
  - even/odd bytes swapped (`cdata[::2]=data[1::2]`, `cdata[1::2]=data[::2]`)
  - this is a compatibility workaround for a known spectrometer-firmware byte-order bug
- If `groupflags == 3`, packet is considered complete and appended.
- Otherwise bytes continue accumulating until terminating segment arrives.

`CollatedPacket` stores:

- `start_seq`: first CCSDS sequence count for the logical packet
- `seq`: sequence count of terminating CCSDS segment
- `app_id`
- `blob`: collated payload bytes
- `single_packet`: whether packet was unsegmented
- `unique_packet_id` (filled later)

## 3) Unique ID assignment and session splitting (`reconstruct_packets.py`)

### 3.1 Unique packet IDs

`assign_uids()` does two passes:

1. `extract_unique_id(pkt)` reads UID directly from payload when packet type supports it.
2. For raw waveform and bootloader packets, UID is inherited from nearest preceding packet with a UID.

UID extraction rules depend on AppID:

- some AppIDs use `pycoreloop` ctypes structs (`from_buffer(...)`) and read `header.unique_packet_id`
- some AppIDs store UID as little-endian `uint32` at offset 0
- heartbeat packets are treated as invalid in this pipeline (`RuntimeError`)

After UID extraction, `decode_directory()` keeps only packets with UID and sorts by:

`(unique_packet_id, seq)`

This is the main chronological/causal ordering used downstream.

### 3.2 Session boundaries

`split_into_sessions(pkts)` starts a new session when `AppID_uC_Start` appears after non-start data.

- Consecutive `uC_Start` packets are treated as one session start sequence.
- `get_session_start_seconds()` parses first startup packet and converts mission time via:
  - `unc.utils.Time2Time(header.time_32, header.time_16)`

Session output directory name:

`cdi_output_<session_start_seconds>`

## 4) Intermediate session format (`cdi_output_*`)

`write_sessions()` writes one file per collated packet:

- filename format:
  - `{:05d}_{:04x}.bin` for < 1,000,000 packets
  - `{:06d}_{:04x}.bin` otherwise
- fields encoded in name:
  - left part: packet index in session order
  - right part: AppID in hex
- file content:
  - exactly `CollatedPacket.blob` bytes (already collated/reordered payload)

Optional integrity mode (`check_existing=True`) compares MD5 of existing files to avoid silent mismatch.

This directory is what `uncrater.Collection` consumes.

## 5) Telemetry side channel (`receive/telemetry_utils.py`)

Telemetry is read separately from `b01/FFFFFFFE`:

1. parse CCSDS frames with `L0_to_ccsds()`
2. keep only telemetry AppIDs with `extract_telemetry_packets()`:
   - `0x314` (FPGA/DCB telemetry)
   - `0x325` (encoder telemetry)

Notes:

- telemetry packets are expected to be single CCSDS packets (`assert single_packet`)
- unlike `collate_packets()`, telemetry extraction does **not** call `reorder()`
  - reason: telemetry comes from different firmware that does not have the spectrometer byte-order bug

### 5.1 Packet formats

- `0x314` is bitfield-decoded with `bitstring.BitStream` using `TELEMETRY_FIELDS`
  - includes `mission_seconds`, `lusee_subsecs`, and many 12-bit ADC/count fields
  - engineering unit conversion is applied (voltage/current/temperature formulas)
- `0x325` is parsed with `ENCODER_FIELDS`, retaining key fields:
  - `mission_seconds`, `lusee_subsecs`, `enc_pos`, `enc_status`

### 5.2 Session alignment

`decode_telemetry_directory(path, session_start_seconds, ...)` assigns each telemetry row to a session using:

- `_session_index()` with `bisect_right(session_start_seconds, mission_seconds) - 1`

Result type:

`List[(fpga_dict, encoder_dict)]` where each dict contains NumPy arrays for a single session.

This pair is passed into HDF5/FITS writers.

## 6) Session to HDF5 (`receive/hdf5_writer.py`)

`save_to_hdf5(cdi_dir, output_file, consts, fpga_tel, encoder_tel)` creates `HDF5Writer` and calls `write()`.

High-level behavior:

1. load `unc.Collection(cdi_dir)` (parses `.bin` session directory)
2. assign zoom packet timestamps from nearest preceding spectra metadata (`_assign_zoom_timestamps`)
3. write file-level metadata:
   - `session_invariants` from hello/start packet
   - `constants` (landing site constants)
   - `DCB_telemetry` (session-aligned telemetry arrays)
4. group science data by metadata configuration (`_group_by_metadata`)
5. write each group as `item_###` with sub-products:
   - spectra, TR spectra, waveform, housekeeping, zoom spectra, calibrator data

Metadata serialization uses `metadata_utils.metadata_to_dict()` and handles nested dict/list/array structures recursively.

### 6.1 Missing spectra behavior in HDF5

For data consumers: missing spectra data are represented as `NaN` values in `item_###/spectra/data`.

Developer mechanics (where this happens): `receive/hdf5_writer.py`, method `_write_spectra()`.

- The output cube is preallocated as `float32` filled with `NaN`:
  - shape `(n_time, NPRODUCTS, NCHANNELS)`
- For each time index, only products present in `sp_dict` are written.
- If a product packet is missing at that time, that `[time, product, :]` slice stays `NaN`.
- If a product exists but has fewer than `NCHANNELS` values, only the available prefix is written and the rest stays `NaN`.

If an `item_###` has no spectra at all, `_write_spectra()` returns early and that item simply has no `spectra/*` datasets.

Again: detailed dataset tree is documented in [`receive/hdf5.md`](receive/hdf5.md).

## 7) Metadata conversion (`receive/metadata_utils.py`)

`metadata_to_dict(meta_pkt)` normalizes metadata packets into plain Python/NumPy structures:

- scalar run/config fields (averaging, notch, masks, flags, etc.)
- per-channel arrays (`gain`, `actual_gain`, bitslice arrays, etc.)
- structured ADC stats and route fields (converted into arrays)
- extra runtime fields attached to packet objects (`adc_*`, `telemetry_*`, `time`)

This normalized dict drives both:

- metadata-group equality checks (for grouping in HDF5 writer)
- final storage to HDF5 attributes/datasets

## 8) User-friendly loading (`receive/test_data.py` + `receive/data.py`)

The intended analysis API is `receive/data.py`, for example:

```python
from data import Spectra

data = Spectra(source="session_001_20251105_120504.h5")
```

From `test_data.py`, expected usage is:

- `data.data`: NumPy array with spectra samples
- `data.time`: per-sample time array
- `data.FPGA_temperature`: convenience metadata field (alias from `telemetry_T_FPGA`)
- `data.lun_lat_deg`, `data.lun_long_deg`, `data.lun_height_m`: constants from file

`Data`/`Spectra` also support:

- loading multiple files (`source=[...,...]`) and concatenating rows
- loading by directory or ID pattern via `LUSEE_SESSIONS_DIR`
- exposing metadata entries as attributes (broadcast to data row count)

Minimal validation pattern (mirrors `test_data.py`):

```python
import numpy as np
from data import Spectra

data = Spectra(source="session_001_20251105_120504.h5")
assert isinstance(data.data, np.ndarray)
assert data.data.shape[0] == data.time.shape[0]
assert data.FPGA_temperature.shape[0] == data.data.shape[0]
```

## 9) Practical notes

- CCSDS parse is strict: CRC mismatch aborts decode.
- UID extraction is AppID-specific; unsupported AppIDs may be dropped from final sorted stream if no UID is found.
- Telemetry is sourced from a separate file (`b01`) and matched to science sessions by mission time, not packet order.
- `.fits` writing mirrors the same session inputs as HDF5 and is executed in the same loop in `reconstruct_packets.py`.
