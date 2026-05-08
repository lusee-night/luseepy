# Specification: CCSDS to Uncrater Session to HDF5/FITS Pipeline

## Context

The receive pipeline takes raw binary downlink files produced by the LuSEE-Night
instrument, recovers the on-wire CCSDS packets, reassembles them into logical
science packets, splits them into per-observation sessions, persists each
session as a directory of packet blobs (the "uncrater session" intermediate
format), and finally produces one HDF5 and one FITS file per session.

This specification is intended as the ground truth for a clean re-implementation.
It is purely data-level: it describes what bytes appear where, what
transformations are applied to them, and what the on-disk outputs look like. It
does **not** describe the current code's organization (functions, modules,
classes) — the new implementation is expected to design its own structure from
scratch.

Two important constraints set by the user:

1. **Telemetry preservation.** Real flight data carries DCB-side telemetry
   (FPGA voltages, currents, temperatures, encoder position) inside the same
   raw CCSDS stream as the science packets, but on a different storage bank.
   The uncrater session format does **not** persist this telemetry.
   Therefore telemetry only reaches the final HDF5/FITS products if the
   conversion runs raw → HDF5 in a single process pass; an offline run that
   starts from a saved uncrater session will lose it.

   Some legacy uncrater sessions captured from real flight data carry a
   sibling `DCB_telemetry.json` binary sidecar (despite the misleading name,
   this file is binary). The new pipeline must be able to **read** this
   sidecar when present so telemetry from such sessions still propagates to
   HDF5. It must **not** write a new sidecar of this kind: carrying a binary
   "json" file alongside the session is not desired going forward. The
   intended path for telemetry into HDF5 is the in-memory single-pass
   conversion described in §1.
2. **Corrupted test data.** The test fixtures currently in the repository
   contain corrupted DCB telemetry packets: when the published bitfield
   conversion formulas are applied to them, intermediate quantities go
   non-physical (negative arguments to `log`, etc.) and the final values are
   meaningless. The pipeline must not crash on these inputs. The required
   behavior across all telemetry-decoding code paths is: catch the math
   error, emit `0.0` as a sentinel, and issue a warning (a per-field
   warning is acceptable, including a deduplicated/throttled form). This
   uniform behavior replaces the current state in which only some code
   paths catch the exception.

The pipeline has two operating modes that share most stages:

* **Raw flash mode** — input is a directory tree of raw binary downlinks.
  Produces uncrater sessions and the final HDF5/FITS in a single process
  pass.
* **Existing-session mode** — input is one or more pre-extracted uncrater
  session directories. If a session carries a legacy `DCB_telemetry.json`
  sidecar it is read; otherwise telemetry is absent for that session.
  Produces HDF5/FITS only.

> **Note on FITS.** All FITS-related sections of this spec are explicitly
> **subject to change**. FITS is currently a secondary priority — no real
> consumers exist yet, so the format may be redefined. HDF5 is the
> authoritative output and the FITS sections describe the current
> behavior for reference only.

---

## 1. Pipeline Overview

```
                       (raw flash mode)                (existing-session mode)
                              |                                  |
                              v                                  |
   +------------------------------------+                         |
   |   raw downlink bank files          |                         |
   |   (FLASH_TLMFS/b01,b05..b09/...)   |                         |
   +------------------------------------+                         |
                  |                                                |
                  v                                                |
   +------------------------------------+                          |
   | Stage 1. CCSDS frame recovery      |                          |
   |   - find sync words                |                          |
   |   - parse 6-byte primary headers   |                          |
   |   - validate CRC-16-CCITT          |                          |
   +------------------------------------+                          |
                  |                                                |
                  v                                                |
   +------------------------------------+                          |
   | Stage 2. Logical packet collation  |                          |
   |   - reassemble multi-segment       |                          |
   |     payloads via group flags       |                          |
   |   - 16-bit lane swap on science    |                          |
   |     payloads only                  |                          |
   |   - filter telemetry banks to      |                          |
   |     APIDs 0x314, 0x325 only        |                          |
   +------------------------------------+                          |
                  |                                                |
                  v                                                |
   +------------------------------------+                          |
   | Stage 3. Logical-packet identity   |                          |
   |   - extract or inherit             |                          |
   |     unique_packet_id               |                          |
   |   - sort by (unique_packet_id,     |                          |
   |               sequence_cnt)        |                          |
   +------------------------------------+                          |
                  |                                                |
                  v                                                |
   +------------------------------------+                          |
   | Stage 4. Session splitting         |                          |
   |   - new session at each startup    |                          |
   +------------------------------------+                          |
                  |                                                |
                  v                                                |
   +------------------------------------+      +-----------------+ |
   | Stage 5. Persist uncrater session  |----->| cdi_output_*    |<+
   |   - one .bin per logical packet    |      +-----------------+ |
   |   - no telemetry sidecar written   |               |          |
   |     (legacy DCB_telemetry.json     |               v          |
   |      may be read if present)       |      +-----------------+ |
   +------------------------------------+      | Stage 6. Decode |<+
            |                                  | uncrater session |
            v                                  | + optional       |
   (telemetry held in memory                   |   legacy sidecar |
    and forwarded to HDF5/FITS)                +-----------------+
                                                        |
                                                        v
                                       +-------------------------------+
                                       | Stage 7. HDF5 writer          |
                                       | Stage 8. FITS writer          |
                                       +-------------------------------+
                                                        |
                                                        v
                                       session_NNN_YYYYMMDD_HHMMSS.h5
                                       session_NNN_YYYYMMDD_HHMMSS.fits
```

A few invariants tie the stages together:

* The transport layer (Stages 1-2) is content-agnostic: it produces a stream
  of typed byte blobs identified by an 11-bit application ID (APID).
* The logical layer (Stages 3-4) imposes per-packet identity and per-session
  grouping, but does not yet know the body layout of any APID.
* The product layer (Stages 6-8) interprets the body of each APID against an
  external schema (described in §7) and writes the result as scientific data
  products.

---

## 2. Inputs

### 2.1 Raw flash mode

The input is a directory whose conventional name is `FLASH_TLMFS`. Inside it,
science and telemetry are stored in separate banks:

| Path inside input directory | Contents                                         |
|-----------------------------|--------------------------------------------------|
| `b05/FFFFFFFE`              | Science packets, bank 5                          |
| `b06/FFFFFFFE`              | Science packets, bank 6                          |
| `b07/FFFFFFFE`              | Science packets, bank 7                          |
| `b08/FFFFFFFE`              | Science packets, bank 8                          |
| `b09/FFFFFFFE`              | Science packets, bank 9                          |
| `b01/FFFFFFFE`              | DCB / encoder telemetry packets (restricted APIDs) |

Each `FFFFFFFE` file is an unframed concatenation of CCSDS packets. The literal
filename `FFFFFFFE` (eight hex digits) is the on-disk convention; it has no
extension and no internal magic bytes. Banks may be missing — if a particular
`bNN/FFFFFFFE` file does not exist, that bank is silently skipped and only
the present banks are processed.

### 2.2 Existing-session mode

The input is one or more directories named `session_*` (or any name; only the
internal layout matters). Each such directory contains:

```
session_<something>/
  cdi_output/                 (mandatory)
    NNNNN_XXXX.bin
    NNNNN_XXXX.bin
    ...
  DCB_telemetry.json          (optional, binary despite extension; see §6.2)
```

Some legacy sessions captured from real flight data have a
`DCB_telemetry.json` sidecar. Sessions produced by the coreloop simulator,
and any session created by the new pipeline, do not. The pipeline must
tolerate both — read the sidecar if present, but never require it.

---

## 3. Stage 1 — CCSDS frame recovery

### 3.1 Stream framing

Each bank file is treated as one continuous byte stream. The stream is a
concatenation of CCSDS packets interspersed with arbitrary `0xA5` padding
bytes. Packet boundaries are not length-prefixed at the stream level; they are
discovered with a sync word.

Framing constants:

| Symbol      | Value     | Role                                                  |
|-------------|-----------|-------------------------------------------------------|
| Sync word   | `0xECA0`  | Two-byte big-endian marker placed immediately before each CCSDS primary header |
| Padding byte| `0xA5`    | Discarded while seeking sync; never appears as the high byte of `0xECA0`        |

A state machine consumes the stream byte by byte:

```
state = FINDING_SYNC; sync_window = 0; head = b''; body = b''
for each byte v from the stream:
    if state == FINDING_SYNC:
        if v == 0xA5:                              # padding
            continue
        sync_window = ((sync_window << 8) | v) & 0xFFFF
        if sync_window == 0xECA0:
            state = READING_HEADER
            sync_window = 0
            head = b''; body = b''
    elif state == READING_HEADER:
        head += v
        if len(head) == 6:
            decode primary header; remember packetlen
            state = READING_BODY
    elif state == READING_BODY:
        body += v
        if len(body) >= packetlen + 3:
            split body into payload = body[:-2] and pktcrc = body[-2:]
            verify CRC-16-CCITT over (head + payload)
            on match: emit (sequence_cnt, head, payload)
            on mismatch: abort the whole stream (current behavior)
            state = FINDING_SYNC
```

Notes:

* The `0xECA0` sync word may legitimately appear inside a payload, and that
  is not a problem. CCSDS Space Packets are self-delimited (CCSDS
  133.0-B-2, §2.1.1–2.1.2): once the 6-byte primary header is parsed,
  the body length is known exactly from `packetlen`, and the parser
  consumes precisely `packetlen + 3` bytes for body+CRC without
  rescanning for sync. Sync-word search is performed **only** between
  packets, never inside an in-progress one. There is therefore no need
  for a byte-stuffing or escape mechanism.
* The current pipeline is strict: a single CRC failure aborts processing of
  the whole bank file. A more permissive design that resyncs and discards the
  bad packet is acceptable but must be a conscious choice.

### 3.2 CCSDS primary header

The 6-byte primary header is fixed by the CCSDS Space Packet standard. All
multi-byte fields are big-endian. The header is parsed as three 16-bit
big-endian unsigned integers `H0`, `H1`, `H2`:

| Bits in `H0` (16) | Field name      | Width | Meaning                                                      |
|-------------------|-----------------|-------|--------------------------------------------------------------|
| 15..13            | `version`       | 3     | CCSDS version (always 0 in this system)                       |
| 12                | `packet_type`   | 1     | 0 = telemetry, 1 = command                                    |
| 11                | `secheaderflag` | 1     | Secondary header presence (not used downstream)               |
| 10..0             | `appid`         | 11    | Application ID, the byte-stream-level packet type identifier  |

| Bits in `H1` (16) | Field name      | Width | Meaning                                                      |
|-------------------|-----------------|-------|--------------------------------------------------------------|
| 15..14            | `groupflags`    | 2     | Segmentation flag, see §4.1                                   |
| 13..0             | `sequence_cnt`  | 14    | Mod-16384 transmission counter, increments per CCSDS packet  |

| Bits in `H2` (16) | Field name      | Width | Meaning                                                      |
|-------------------|-----------------|-------|--------------------------------------------------------------|
| 15..0             | `packetlen`     | 16    | (Length of the CCSDS data field) - 1 byte                    |

The CCSDS standard says the data field is `packetlen + 1` bytes long. In this
system the 2-byte CRC trailer is appended **outside** the CCSDS data field,
so the on-stream byte count after the primary header is `packetlen + 1 + 2 =
packetlen + 3`. This is intentional and not an off-by-one error.

### 3.3 CRC-16-CCITT

After collecting `packetlen + 3` bytes after the primary header, the parser
splits them as:

* `payload = bytes[0 : packetlen + 1]`
* `pktcrc  = uint16_be(bytes[packetlen + 1 : packetlen + 3])`

It then computes a CRC-16-CCITT checksum over the concatenation `head ||
payload`. The algorithm is:

```
crc = 0xFFFF
for b in head + payload:
    crc ^= (b << 8)                  # XOR byte into MSB of running register
    for _ in range(8):                # MSB-first
        if (crc & 0x8000) != 0:
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF
        else:
            crc = (crc << 1) & 0xFFFF
```

Parameters:

| Parameter      | Value     |
|----------------|-----------|
| Polynomial     | `0x1021`  |
| Initial value  | `0xFFFF`  |
| Bit order      | MSB-first |
| XorOut         | none      |
| RefIn / RefOut | false     |

The CRC is computed over the 6-byte primary header plus the payload (i.e. the
data field excluding the trailing CRC bytes themselves).

A mismatch is treated as a fatal corruption signal in the current code. A
re-implementation may choose to raise instead of aborting; either way the
packet must not be emitted on mismatch.

### 3.4 Output of Stage 1

The output of Stage 1 is, per bank file, an ordered list of validated
CCSDS-packet records. Each record carries:

* The 14-bit `sequence_cnt` from the primary header.
* The full 6-byte primary header.
* The variable-length payload (with the 2-byte CRC stripped).

---

## 4. Stage 2 — Logical packet collation

### 4.1 Segmentation flags

The 2-bit `groupflags` field of every CCSDS packet describes its role inside a
multi-segment logical packet:

| `groupflags` | Binary | Meaning                                                      |
|--------------|--------|--------------------------------------------------------------|
| 3            | `11`   | Unsegmented — the entire logical packet fits in this CCSDS frame |
| 2            | `10`   | First segment of a multi-segment logical packet              |
| 0            | `00`   | Continuation segment                                         |
| 1            | `01`   | Last segment                                                 |

A logical packet therefore corresponds to either a single CCSDS frame with
`groupflags == 3`, or to a contiguous run of frames with `groupflags`
sequence `2 → (0 …) → 1`. The CCSDS `sequence_cnt` is *not* reset at logical
packet boundaries; it is a global transmission counter.

> **Note on the current behavior.** As implemented today, the science
> reassembler emits a logical packet **only** when it sees `groupflags == 3`;
> it never treats `groupflags == 1` as a terminator on the science path. In
> practice the producer only ever emits multi-frame logical packets that end
> with `groupflags == 3` rather than `1`, so this works for the current data,
> but it is not standard CCSDS. A faithful re-implementation can either
> (a) preserve this behavior verbatim, or (b) terminate on either `1` or `3`,
> which is the conventional CCSDS rule. Whichever path is chosen must be
> applied consistently and stated up front.
[USER COMMENT: let's follow the CCSDS rule to be standard-conformant]


### 4.2 Reassembly algorithm

Reassembly walks the per-bank list in order and accumulates payload bytes into
a buffer. The `start_seq` for a logical packet must be captured at the
**first** contributing CCSDS frame (i.e. the frame with `groupflags == 2`,
or the unsegmented frame itself), not overwritten on subsequent frames:

```
buffer = b''; start_seq = None; current_appid = None
for each (sequence_cnt, head, payload) in stream:
    appid, groupflags = unpack from head
    if start_seq is None:                 # first frame of a logical packet
        start_seq = sequence_cnt
        current_appid = appid
    if appid != current_appid:
        warn (APID changed mid-packet); keep current_appid as is
    buffer += transform(payload)          # see §4.3
    if groupflags == 3:
        emit logical packet (single_packet=True, start_seq, seq=sequence_cnt)
        buffer = b''; start_seq = None; current_appid = None
    elif groupflags == 1:
        emit logical packet (single_packet=False, start_seq, seq=sequence_cnt)
        buffer = b''; start_seq = None; current_appid = None
    # groupflags 2 and 0: keep accumulating
```

(See the note above on whether the science path actually terminates on
`groupflags == 1`. The pseudocode here is the conventional CCSDS form; the
current code emits only on `3`.)

For each emitted logical packet, the following fields are recorded:

* `appid` — the 11-bit application ID.
* `start_seq` — the `sequence_cnt` of the first contributing CCSDS frame.
* `seq` — the `sequence_cnt` of the terminating CCSDS frame (== `start_seq`
  for unsegmented packets).
* `blob` — the concatenated, possibly byte-swapped payload.
* `single_packet` — True iff `groupflags == 3` for the originating frame.
* `unique_packet_id` — populated in Stage 3 (initially absent).

### 4.3 Per-bank payload transform

Science banks (`b05..b09`) carry payloads with a known firmware byte-order
defect: the spectrometer FPGA emits 16-bit lanes byte-swapped relative to the
documented packet layouts. To compensate, every science-bank CCSDS payload is
byte-swapped in place at 16-bit granularity before being appended to the
buffer:

```
def transform_science(payload):
    out = bytearray(len(payload))
    out[0::2] = payload[1::2]
    out[1::2] = payload[0::2]
    return bytes(out)
```

The science payload length is always even in normal operation; an odd-length
payload would cause the slice assignment above to fail (the two halves have
different lengths) and is treated as an error. This is acceptable because
no valid science data is expected to produce odd-length payloads.

The DCB telemetry bank (`b01`) carries payloads from a different firmware
that does **not** have the byte-swap defect. Its payloads must be passed
through untransformed. The byte-swap is a workaround for a specific
science-firmware bug, so it must only be applied to packets coming from
the science banks (and never to telemetry, regardless of any future
addition of telemetry-like content elsewhere).

### 4.4 Per-bank APID filtering

After Stage 1+2 there is one stream per bank. The semantics differ:

* For each science bank (`b05..b09`), the result is a list of logical packets
  carrying any APID. All of these are kept and merged into a single global
  science stream, ordered as in §4.5.
* For the telemetry bank (`b01`), the result is filtered down to the
  recognised telemetry APIDs claimed by the private decoder (see
  [`lusee_telemetry`][lusee-tel]); typically FPGA / DCB engineering
  housekeeping and encoder position. The exact APID values are
  restricted.

  Telemetry packets are required to be unsegmented (`single_packet == True`).
  A segmented telemetry packet is treated as a hard error in the current code
  and should be flagged in any re-implementation.

### 4.5 Global ordering

After per-bank processing, all science logical packets from `b05..b09` are
merged into a single ordered stream. Stage 3 imposes the final sort order;
between Stage 2 and Stage 3 the natural order is each bank in file order, with
banks concatenated. Stage 3 will resort by `(unique_packet_id, seq)` after
identity assignment.

---

## 5. Stage 3 — Logical-packet identity

Each logical science packet is tagged with a 32-bit `unique_packet_id`. This
ID — not the CCSDS `sequence_cnt` — is what binds together a metadata packet,
its associated full-resolution spectra, its time-resolved spectra, and any
calibration products that share an acquisition.

### 5.1 APID classes by ID encoding

There are three classes of logical packets, distinguished by where (or whether)
the ID lives in the blob. The classes are determined by the APID:

#### Class A: explicit ID at offset 0 (little-endian uint32)

For these APIDs the first four bytes of the logical-packet blob are the
`unique_packet_id` as a little-endian unsigned 32-bit integer. The rest of the
blob is the body proper.

This class includes (inclusive of all variants and priority levels):

* All full-resolution spectra APIDs.
* All time-resolved (TR) spectra APIDs.
* All calibrator zoom-spectra APIDs.
* The Calibrator Data and Calibrator Raw-PFB APIDs.
* All calibrator debug APIDs.
* The Grimm spectra APID.

#### Class B: explicit ID inside a typed header

For these APIDs the blob starts with a typed C-style header structure (defined
externally — see §7) that contains a `unique_packet_id` field at a known
offset. The header is parsed and the field read as a native-endian unsigned
32-bit integer.

This class includes:

* The startup ("Hello") packet.
* Generic housekeeping packets.
* Calibrator metadata packets.
* Generic metadata packets.
* Waveform metadata packets.

The heartbeat APID nominally falls in this class but currently raises an
error and is excluded from the science stream. A re-implementation may choose
to keep it but must decide explicitly.

#### Class C: no embedded ID — inherit from preceding packet

For these APIDs the blob has no internal ID, but the packet is logically tied
to an earlier metadata packet that does. The ID is inherited from the most
recently emitted Class-A or Class-B packet that has an ID assigned.

This class includes:

* All raw-ADC waveform APIDs (one waveform per CCSDS packet, one channel each).
* Bootloader-related APIDs.

Inheritance walks backward from the current packet through the per-bank
ordered stream until it finds a packet with `unique_packet_id != None`. If
none exists, the packet is dropped from the science stream.

### 5.2 Identity assignment algorithm

```
# Pass 1: explicit extraction
for pkt in stream:
    if pkt.appid is Class A:
        pkt.unique_packet_id = u32_le(pkt.blob[0:4])
    elif pkt.appid is Class B:
        header = parse_typed_header(pkt.blob, pkt.appid)
        pkt.unique_packet_id = header.unique_packet_id
    # Class C: leave None for now

# Pass 2: inheritance
for i, pkt in enumerate(stream):
    if pkt.appid is Class C and pkt.unique_packet_id is None:
        for j in range(i - 1, -1, -1):
            if stream[j].unique_packet_id is not None:
                pkt.unique_packet_id = stream[j].unique_packet_id
                break

# Filter and sort
stream = [pkt for pkt in stream if pkt.unique_packet_id is not None]
stream.sort(key=lambda p: (p.unique_packet_id, p.seq))
```

The sort by `(unique_packet_id, seq)` is the canonical chronological order
used by all downstream processing.

### 5.3 Identity collisions across APIDs

The producer assigns `unique_packet_id` independently per APID family in some
cases, so two logically unrelated packets with different APIDs can share an
ID. Sessionizing and product reconstruction therefore key on
`(unique_packet_id, appid)` or rely on the sort order above; never on
`unique_packet_id` alone. A re-implementation must preserve this distinction.

---

## 6. Stage 4 — Session splitting and uncrater session persistence

### 6.1 Session boundaries

A session is the run of packets between two startup events. The startup event
is the appearance of a startup packet (Class B "Hello") in the sorted science
stream. The rules:

* A new session begins when a startup packet appears after at least one
  non-startup packet has been emitted in the current session.
* Consecutive startup packets are folded into the same session opening — they
  do **not** create empty sessions.
* If the stream begins with one or more non-startup packets (no startup ever
  seen), they form a "session 0" with no startup-derived metadata.

The session start time, used for filename generation and telemetry alignment,
is decoded from the first startup packet of the session by combining its
`time_32` and `time_16` fields:

```
raw_seconds = ((((time_16 & 0xFFFF) << 32) + time_32) >> 4) / 4096
```

This formula treats the concatenation `(time_16 << 32) | time_32` as a 48-bit
fixed-point number with 12 fractional bits at the bottom (after the
right-shift by 4 and division by 4096), giving a resolution of approximately
244 microseconds and a range of approximately 136 years. If the session lacks
a startup packet, `raw_seconds = 0` is used as a fallback.

### 6.2 Uncrater session directory layout

A session is a directory of `.bin` files, one per logical packet, plus
optional auxiliary files. There are two layouts in current use:

```
# Layout A — bare directory of packet files (raw flash mode)
cdi_output_<session_id>/
  NNNNN_XXXX.bin
  NNNNN_XXXX.bin
  ...

# Layout B — wrapper directory with a cdi_output/ subdirectory
<session_root>/
  cdi_output/
    NNNNN_XXXX.bin
    NNNNN_XXXX.bin
    ...
  DCB_telemetry.json                  # optional, legacy, binary; see §6.2.2
```

The new pipeline must accept both layouts when reading. Whether the
`cdi_output_` prefix carries a session timestamp, an integer ordinal, or
some other identifier is **not fixed by this spec** — session-id naming
is a separate decision that will be made elsewhere. What this spec
guarantees is the per-packet file naming and content (§6.2.1) and the
optional sidecar format (§6.2.2). Everything else about the parent
directory name is informational.

#### 6.2.1 Per-packet files

Inside `cdi_output/`, each surviving logical packet (one per row of the
sorted, sessionized science stream) is written as one binary file:

* **Filename** — `<index>_<apid_hex>.bin`, where:
  * `<index>` is the packet's ordinal within the session (zero-based,
    contiguous integers), printed as a zero-padded decimal string. Five
    digits are used when the session has fewer than 1,000,000 packets, six
    digits otherwise. The same width is used for every file in a given
    session.
  * `<apid_hex>` is the 11-bit APID written as a four-character lowercase
    hexadecimal string with leading zeros (`{:04x}`).
* **Content** — exactly the bytes of the logical-packet blob produced by
  Stage 2, including the byte-swap on science banks. There is no header,
  trailer, length prefix, or magic byte. The CCSDS primary header and the
  CRC are not stored.
* **Size** — variable, determined by the APID's body type. Sizes range from
  a few tens of bytes (housekeeping, hello) to ~32 KB (full-resolution
  spectra).

Filename examples:

| Filename               | Index | APID    | Meaning                          |
|------------------------|-------|---------|----------------------------------|
| `00000_0209.bin`       | 0     | `0x209` | Startup (Hello)                  |
| `00001_02fa.bin`       | 1     | `0x2FA` | Generic metadata                 |
| `00002_02f0.bin`       | 2     | `0x2F0` | Spectra (high priority, prod 0)  |
| `00003_02f1.bin`       | 3     | `0x2F1` | Spectra (high priority, prod 1)  |

The `<index>_<apid_hex>` filename is a *scrambled* identity: it does not
contain `unique_packet_id`. To recover the link between files belonging to
the same acquisition, the consumer must reparse the body and read the
embedded ID. This is the contract the uncrater decoder relies on (see §7).

#### 6.2.2 The `DCB_telemetry.json` sidecar

Despite its `.json` extension, this file is a **binary** stream: a
back-to-back concatenation of fixed-size DCB telemetry records that
match the on-wire `0x314` CCSDS packet body byte-for-byte (CCSDS
primary header and CRC stripped). The detailed byte-level layout, the
list of channel names, and the engineering-unit conversion formulas
are restricted; they live with the [`lusee_telemetry`][lusee-tel]
package and are not reproduced in this public spec.

[lusee-tel]: https://github.com/lusee-night/lusee_telemetry

This sidecar is a **legacy, read-only** artefact in the new pipeline:

* **Read** the sidecar in existing-session mode when it is present, and use
  it as the telemetry source for that session.
* **Tolerate** its absence (most sessions, including all coreloop-simulator
  sessions, will not have it).
* **Do not write** new sidecars. The new raw-flash path keeps decoded
  telemetry in memory and forwards it directly to the HDF5 writer; it must
  not produce a new `DCB_telemetry.json`. Carrying a binary file under a
  `.json` extension is undesirable going forward, and the single-pass
  in-memory route is the only supported way for telemetry to reach HDF5
  in newly produced sessions.

#### 6.2.3 Encoder telemetry is not persisted

The uncrater session format stores no encoder telemetry sidecar.
Encoder data captured from the (restricted) encoder APID in raw flash
mode is held in memory for the duration of the run and passed straight
to the HDF5 writer in single-pass mode; once the run ends and the
uncrater session is the only artefact, encoder telemetry is lost.
Adding a new binary sidecar of any kind is explicitly **not** desired
(see §6.2.2). The single-pass-to-HDF5 route, or the in-session
manifest's flash-source backreference, are the supported ways to retain
encoder data.

### 6.3 What is preserved and what is lost

| Source datum                                  | Preserved in uncrater session?     |
|-----------------------------------------------|------------------------------------|
| CCSDS 6-byte primary header                   | No                                 |
| CCSDS 2-byte CRC trailer                      | No                                 |
| CCSDS sequence_cnt                            | No (the file ordinal replaces it)  |
| Logical-packet body bytes (post lane-swap)    | Yes (one .bin file)                |
| Logical-packet APID                           | Yes (encoded in filename)          |
| Logical-packet `unique_packet_id`             | Yes (inside the body)              |
| Per-session DCB telemetry stream              | Not persisted by the new pipeline; |
|                                               | only read from a legacy sidecar    |
|                                               | when one already exists            |
| Per-session encoder telemetry                 | No                                 |
| Session start time                            | Yes (inside the startup packet)    |

This is the central data-loss point of the pipeline and the reason the user
emphasizes single-pass conversion when telemetry must reach the final HDF5/FITS.

---

## 7. Stage 6 — Decoding the uncrater session into in-memory products

The uncrater session directory is consumed by an external decoder (the
`uncrater` package) that interprets each `.bin` file according to its APID.
This specification does not redefine that decoder; it defines what the new
pipeline must obtain from it before writing HDF5/FITS.

### 7.1 Firmware version selection

The startup ("Hello") packet carries a `SW_version` field. The decoder uses it
to select one of three packet schemas:

| `SW_version` | Schema                |
|--------------|-----------------------|
| `0x203`      | Earliest documented   |
| `0x305`      | Mid-generation        |
| `0x307`      | Latest (default)      |

If no startup packet is available, the decoder falls back to the latest
schema. The differences between schemas are confined to body field names and
widths; the framing layer (CCSDS, sync, CRC) is identical across versions.

### 7.2 Required packet families

After decoding, the pipeline must have access to (per session):

#### Startup packet
One per session, carrying:

* `SW_version` — software version.
* `FW_Version`, `FW_ID`, `FW_Date`, `FW_Time` — firmware identity and
  build metadata.
* `unique_packet_id` — session-start identity.
* `time_32`, `time_16` — split mission time at session start.

#### Metadata packets
Each metadata packet defines an acquisition context. Multiple metadata
packets can appear within a session; each opens a new context and remains in
force until the next one. Each carries:

* Acquisition time (the fields used to compute `raw_seconds` per §6.1's
  formula).
* Averaging settings (`Navg1_shift`, `Navg2_shift`, `Navgf`, etc.).
* Output format (full-resolution / medium-resolution / low-resolution flag).
* The bitmask of enabled correlation products (16 bits).
* TR window definition (`tr_start`, `tr_stop`, `tr_avg_shift`).
* Per-channel gain and routing state (`gain[4]`, `actual_gain[4]`,
  `route.plus[4]`, `route.minus[4]`).
* Slicer state (`bitslice[16]`, `actual_bitslice[16]`, `gain_auto_min[4]`,
  `gain_auto_mult[4]`).
* ADC statistics (`adc_min[4]`, `adc_max[4]`, `adc_mean[4]`, `adc_rms[4]`,
  `adc_invalid_count_min[4]`, `adc_invalid_count_max[4]`).
* Embedded low-rate spectrometer-side telemetry (separate from DCB telemetry).
* Various error flags and status fields.

Each metadata packet seeds:
* one normal-spectrum sample,
* one time-resolved-spectrum sample.

#### Normal spectra
Each normal-spectrum sample carries up to 16 correlation products, indexed
0..15. Indices 0..3 are autocorrelations of channels 0..3; indices 4..15 are
the 12 cross-correlations. The four "priority classes" of spectra (high,
medium, low) all collapse onto the same 16 logical indices — the priority is
a transport concern and is not preserved in the science-array layout.

The transmitted form of a spectrum may be a compressed 16-bit per-channel
representation; the decoder is expected to expand it back to the canonical
floating-point form. The frequency axis depends on the metadata averaging
mode:

| Mode                | Channels per product |
|---------------------|----------------------|
| Full resolution     | 2048                 |
| Medium resolution   | 1024                 |
| Low resolution      | 512                  |

The HDF5 cube always reserves 2048 channels; lower-resolution spectra occupy
the leading channels and the remainder is filled with `NaN` (see §8.5).

#### Time-resolved (TR) spectra
Each TR sample carries up to 16 correlation products tied to the same
metadata context. Each product is a 2D array:

* Axis 0 — `2^Navg2_shift` accumulation slices.
* Axis 1 — `(tr_stop - tr_start) / 2^tr_avg_shift` time-resolved bins.

The exact axis lengths are per-sample. The HDF5 cube pads to the session-wide
maxima with `NaN`.

#### Calibrator zoom spectra
Each zoom packet is a focused 64-bin product around one selected polyphase-
filter-bank (PFB) bin. It contains four 64-bin float arrays:

| Index | Quantity                        |
|-------|---------------------------------|
| 0     | Channel-1 autocorrelation       |
| 1     | Channel-2 autocorrelation       |
| 2     | Cross-correlation, real part    |
| 3     | Cross-correlation, imaginary    |

Plus the selected PFB bin index. Zoom packets do not carry their own
high-resolution timestamp; the consumer assigns the `raw_seconds` of the
nearest preceding normal-spectrum metadata in session order.

#### Grimm spectra (optional)
Selected-frequency spectral products with the same `(16, 2048)` shape as
normal spectra. Only written when present and decodable. Not guaranteed in
any session.

#### Waveform packets
Raw ADC snapshots. There are up to four input channels (0..3); each waveform
packet carries 16384 samples for one channel as signed 16-bit integers. A
separate waveform-metadata packet (Class B) carries the timestamp shared by a
capture set. The HDF5 group is organized by channel.

#### Housekeeping packets
Low-rate engineering packets, with multiple subtypes (denoted `hk_type`
0..3). Each subtype has its own field set; see §8.10. Housekeeping is
preserved as one record per packet rather than collapsed into a dense table.

#### Calibrator data packets (optional)
Variable-size complex per-channel calibration arrays. Stored as one dataset
per packet per channel.

### 7.3 Packet families not promoted to first-class products

Some packet families (notably bootloader, calibrator-debug, calibrator
raw-PFB) appear in the session directory but are not currently exposed as
first-class HDF5/FITS products. A re-implementation may keep them around for
auditing but is not required to write them.

---

## 8. Stage 7 — HDF5 output

One HDF5 file is written per session. The file is self-contained; cross-file
references are not used.

### 8.1 File naming

The output filename is:

```
session_NNN_YYYYMMDD_HHMMSS.h5
```

Where:

* `NNN` is the zero-padded three-digit session ordinal within the run
  (`000`, `001`, …). In raw flash mode, ordinals are assigned in the order
  sessions are detected. In existing-session mode, ordinals follow the order
  the user supplies session directories.
* `YYYYMMDD_HHMMSS` is the timestamp formed from the session start time
  (§6.1) interpreted as Unix seconds. The required convention is **UTC**;
  see the note in §11.2 about the legacy local-time behavior.

If a session has no startup packet, the current raw-flash code substitutes
`raw_seconds = 0` and lets the filename go through unchanged, which yields
a Unix-epoch-derived timestamp (`19700101_000000` in UTC, or its
local-time equivalent). A re-implementation should treat "no startup
packet" as a distinct condition and produce `session_NNN.h5` with no
timestamp suffix instead, since a zero stamp is misleading.

### 8.2 Top-level attributes

| Attribute name   | Type      | Value                                                  |
|------------------|-----------|--------------------------------------------------------|
| `cdi_directory`  | UTF-8 str | Source uncrater session path that produced this file   |
| `layout_version` | int64     | Schema version (current value: `2`)                    |
| `n_items`        | int64     | Compatibility field, `0` in schema 2                   |

`layout_version == 2` means the dense top-level science arrays described
below are present and the legacy per-metadata-group `metadata_items/item_*`
layout (schema 1) is suppressed. Older readers that expect schema 1 must
either upgrade or be handled separately.

### 8.3 Top-level group inventory

Mandatory:

* `/session_invariants`
* `/constants`

Always present when the session has any spectra, which it normally does:

* `/spectra`

Optional, present only when the corresponding packet family appears:

* `/tr_spectra`
* `/calibrator/zoom_spectra`
* `/grimm_spectra`
* `/waveform`
* `/housekeeping`
* `/calibrator/data`

Optional, present only when telemetry was supplied — either via single-pass
raw-flash conversion, or via an existing-session input that carries a legacy
`DCB_telemetry.json` sidecar:

* `/DCB_telemetry`
* `/spectra_interpolated_telemetry` (when interpolation was requested)

Compatibility (currently empty in schema 2):

* `/metadata_items` (omitted entirely, or present and empty)

### 8.4 `/session_invariants`

Group with attributes only, no datasets. Sourced from the startup packet of
the session.

| Attribute name           | Type   | Source                          |
|--------------------------|--------|---------------------------------|
| `software_version`       | int64  | `Hello.SW_version`              |
| `firmware_version`       | int64  | `Hello.FW_Version`              |
| `firmware_id`            | int64  | `Hello.FW_ID`                   |
| `firmware_date`          | int64  | `Hello.FW_Date`                 |
| `firmware_time`          | int64  | `Hello.FW_Time`                 |
| `start_unique_packet_id` | int64  | `Hello.unique_packet_id`        |
| `start_time_32`          | int64  | `Hello.time_32`                 |
| `start_time_16`          | int64  | `Hello.time_16`                 |

If the session has no startup packet, this group is created but its
attributes may be absent or zero; consumers must tolerate either.

### 8.5 `/constants`

Group with attributes only.

| Attribute name              | Type    | Default | Meaning                                |
|-----------------------------|---------|---------|----------------------------------------|
| `lun_lat_deg`               | float64 | `-15.0` | Lunar landing latitude, degrees        |
| `lun_long_deg`              | float64 | `175.0` | Lunar landing longitude, degrees       |
| `lun_height_m`              | float64 | `0.0`   | Lunar landing height, meters           |
| `raw_time_subtract_seconds` | float64 | `0.0`   | Offset applied before MJD conversion   |
| `mjd_epoch_offset_days`     | float64 | `0.0`   | MJD epoch offset                       |

The MJD conversion used for the `mjd_times` datasets below is:

```
mjd_time = (raw_seconds - raw_time_subtract_seconds) / 86400 + mjd_epoch_offset_days
```

With both offsets at their defaults this is just `raw_seconds / 86400`,
which is not yet a calibrated MJD; calibrated values are filled in by
overriding the constants from a YAML config when one is supplied.

### 8.6 `/spectra`

The primary dense science product.

| Path                           | Shape              | Dtype    | Compression |
|--------------------------------|--------------------|----------|-------------|
| `/spectra/data`                | `(N, 16, 2048)`    | float32  | gzip        |
| `/spectra/unique_ids`          | `(N,)`             | int64    | gzip        |
| `/spectra/raw_times`           | `(N,)`             | float64  | gzip        |
| `/spectra/mjd_times`           | `(N,)`             | float64  | gzip        |
| `/spectra/original_indices`    | `(N,)`             | int64    | gzip        |
| `/spectra/metadata/...`        | various, leading axis `N` | various | gzip |

Group attributes:

| Attribute | Type  | Meaning                                  |
|-----------|-------|------------------------------------------|
| `count`   | int64 | `N` (number of retained spectra rows)    |

`N` is the count after sample-retention filtering (§8.13). The data axes are:

* Axis 0 — time (sample index).
* Axis 1 — correlation product (0..15).
* Axis 2 — frequency channel (0..2047, with trailing channels `NaN` for
  reduced-resolution spectra).

`/spectra/metadata/` is a subgroup containing one dataset per metadata field,
all sharing leading axis `N`. The full set of fields tracks the metadata
packet schema (§7.2); typical fields include:

| Field name                  | Shape    | Dtype   |
|-----------------------------|----------|---------|
| `Navg1_shift`               | `(N,)`   | int64   |
| `Navg2_shift`               | `(N,)`   | int64   |
| `Navgf`                     | `(N,)`   | int64   |
| `_time_32`                  | `(N,)`   | int64   |
| `_time_16`                  | `(N,)`   | int64   |
| `_uC_time`                  | `(N,)`   | int64   |
| `actual_bitslice`           | `(N, 16)`| int64   |
| `actual_gain`               | `(N, 4)` | int64   |
| `adc_mean`, `adc_rms`       | `(N, 4)` | float64 |
| `adc_min`, `adc_max`        | `(N, 4)` | int64   |
| `adc_invalid_count_min`     | `(N, 4)` | int64   |
| `adc_invalid_count_max`     | `(N, 4)` | int64   |
| `bitslice`                  | `(N, 16)`| int64   |
| `gain`                      | `(N, 4)` | int64   |
| `gain_auto_min`             | `(N, 4)` | int64   |
| `gain_auto_mult`            | `(N, 4)` | int64   |
| `route/plus`, `route/minus` | `(N, 4)` | int64   |

Per-channel telemetry fields embedded in the metadata packet are written as
separate datasets under `/spectra/metadata/` with names that begin
`telemetry_*`.

The metadata-to-dataset translation is recursive: nested dicts/structs become
nested HDF5 groups, lists/arrays become datasets, scalars become attributes
of the appropriate parent (or single-element datasets for uniformity).

### 8.7 `/tr_spectra`

Same shape as `/spectra` but with two extra axes for the TR window.

| Path                                | Shape                                 | Dtype    |
|-------------------------------------|---------------------------------------|----------|
| `/tr_spectra/data`                  | `(N, 16, Navg2_max, tr_length_max)`   | float32  |
| `/tr_spectra/unique_ids`            | `(N,)`                                | int64    |
| `/tr_spectra/raw_times`             | `(N,)`                                | float64  |
| `/tr_spectra/mjd_times`             | `(N,)`                                | float64  |
| `/tr_spectra/navg2_per_sample`      | `(N,)`                                | int64    |
| `/tr_spectra/tr_length_per_sample`  | `(N,)`                                | int64    |
| `/tr_spectra/original_indices`      | `(N,)`                                | int64    |
| `/tr_spectra/metadata/...`          | various                               | various  |

Group attributes:

| Attribute                | Type  | Meaning                                                         |
|--------------------------|-------|-----------------------------------------------------------------|
| `count`                  | int64 | `N`                                                             |
| `tr_spectra_Navg2`       | int64 | `Navg2_max` — maximum first TR axis across all samples          |
| `tr_spectra_tr_length`   | int64 | `tr_length_max` — maximum second TR axis across all samples     |

A given sample's valid extent is `(navg2_per_sample[i], tr_length_per_sample[i])`;
indices beyond that are `NaN`.

### 8.8 `/calibrator/zoom_spectra`

| Path                                                | Shape       | Dtype    |
|-----------------------------------------------------|-------------|----------|
| `/calibrator/zoom_spectra/data`                     | `(N, 4, 64)`| float32  |
| `/calibrator/zoom_spectra/unique_ids`               | `(N,)`      | int64    |
| `/calibrator/zoom_spectra/pfb_indices`              | `(N,)`      | int32    |
| `/calibrator/zoom_spectra/raw_times`                | `(N,)`      | float64  |
| `/calibrator/zoom_spectra/mjd_times`                | `(N,)`      | float64  |
| `/calibrator/zoom_spectra/original_indices`         | `(N,)`      | int64    |

Zoom calibrator data **must** be present in HDF5 whenever zoom packets
arrived; it is not an optional cosmetic feature. The single 3D `data`
dataset above is the canonical form. (The earlier draft also listed
unrolled per-component aliases like `ch1_autocorr`; those are removed
because backward compatibility with older readers is not a concern. The
four components are recovered by indexing `data[:, k, :]`, with `k`
defined below.)

Axis 1 of `data` indexes:

| Index | Component                              |
|-------|----------------------------------------|
| 0     | Channel-1 autocorrelation              |
| 1     | Channel-2 autocorrelation              |
| 2     | Cross-correlation, real part           |
| 3     | Cross-correlation, imaginary part      |

Group attribute `count` = `N`.

Zoom timestamps are inherited from the nearest preceding normal-spectrum
metadata in session order.

### 8.9 `/DCB_telemetry`

This group is present only when telemetry data is available — i.e. raw flash
mode (single-pass), or existing-session mode where a legacy
`DCB_telemetry.json` sidecar is found.

The group holds two parallel time series — FPGA/DCB telemetry and encoder
telemetry — distinguished by name prefix. Each series has its own time
columns, since the two streams have independent cadences:

* FPGA fields are prefixed `fpga_`. Time axis:
  * `/DCB_telemetry/fpga_mission_seconds`
  * `/DCB_telemetry/fpga_lusee_subsecs`
  * `/DCB_telemetry/fpga_<FIELD_NAME>` for each of the 57 telemetry channel
    names listed in §6.2.2 (e.g. `fpga_THERM_FPGA`, `fpga_VMON_6V`, …,
    `fpga_ADC_PWR`).
* Encoder fields are prefixed `encoder_`. Datasets:
  * `/DCB_telemetry/encoder_mission_seconds`
  * `/DCB_telemetry/encoder_lusee_subsecs`
  * `/DCB_telemetry/enc_pos` (encoder position counts)
  * `/DCB_telemetry/enc_status` (encoder status flags)

| Path                                                    | Shape       | Dtype   |
|---------------------------------------------------------|-------------|---------|
| `/DCB_telemetry/fpga_<field>`                           | `(M_fpga,)` | float64 |
| `/DCB_telemetry/encoder_mission_seconds`                | `(M_enc,)`  | float64 |
| `/DCB_telemetry/encoder_lusee_subsecs`                  | `(M_enc,)`  | float64 |
| `/DCB_telemetry/enc_pos`                                | `(M_enc,)`  | int     |
| `/DCB_telemetry/enc_status`                             | `(M_enc,)`  | int     |

`M_fpga` is the number of FPGA telemetry samples assigned to this session;
`M_enc` is the number of encoder samples assigned to this session. Within
each series all datasets share the same length. The encoder datasets are
omitted when no encoder data exists for the session (e.g. when a legacy
sidecar is the only telemetry source — see §6.2.3).

FPGA field values are in **engineering units** for fields with a
calibration formula (§10), and in **raw 12-bit counts** for fields without
one. A re-implementation may choose to add a `units` attribute per
dataset; the current code does not.

There is no interpolation in this group — it is the per-session telemetry
time series in its native cadence.

### 8.10 `/spectra_interpolated_telemetry`

Optional. Present only when the pipeline was asked to resample telemetry onto
the spectra time axis. Contains:

* `/spectra_interpolated_telemetry/time` — `(N,)`, float64, the spectrum raw
  time axis used for interpolation.
* `/spectra_interpolated_telemetry/mjd_time` — `(N,)`, float64, only when MJD
  conversion is possible.
* `/spectra_interpolated_telemetry/<field_name>` — `(N,)`, float64, the
  interpolated telemetry field, one per FPGA telemetry channel.

Interpolation contract:

* Telemetry timestamps are formed as `mission_seconds + lusee_subsecs / 65536`.
* Two interpolation modes are available:
  1. **Normalized relative position (default)** — both telemetry and
     spectra timestamps are mapped to a session-relative `[0, 1]` axis
     before linear interpolation.
  2. **Absolute time** — telemetry is interpolated directly on its physical
     time axis.
* Duplicate telemetry timestamps are averaged before interpolation.
* A single telemetry sample yields a constant series.
* Values outside the telemetry span are held at the nearest endpoint
  (no extrapolation).
* Missing fields produce all-`NaN` arrays.

### 8.11 `/waveform`

Group hierarchy organized by input channel:

```
/waveform/
  @total_count  : int64, total waveforms across all channels
  @channels     : int64[K], list of channel indices present (0..3, in order)
  channel_0/
    @count      : int64, waveforms in this channel
    @channel    : int64, channel index (0)
    waveforms   : (n_ch0, 16384), int16, gzip
    timestamps  : (n_ch0,)       , float64, gzip
  channel_1/
    ... (same structure, present only if channel 1 has data)
  ...
```

Timestamps are `raw_seconds` derived from the waveform-metadata packet that
opened each capture set.

### 8.12 `/housekeeping`

One subgroup per received housekeeping packet, named `packet_<i>` with `i`
the packet's ordinal among housekeeping packets in the session.

```
/housekeeping/
  @count : int64, number of housekeeping packets
  packet_0/
    @hk_type          : int64, subtype (0..3)
    @version          : int64
    @unique_packet_id : int64
    @errors           : int64
    (additional per-subtype attributes — see below)
  packet_1/
    ...
```

Per-subtype additional attributes:

| `hk_type` | Additional attributes                                                                                |
|-----------|------------------------------------------------------------------------------------------------------|
| 0         | `time` (float64)                                                                                     |
| 1         | `adc_min[4]`, `adc_max[4]` (int64); `adc_mean[4]`, `adc_rms[4]` (float64); `actual_gain[4]` (int64)  |
| 2         | `time` (float64); `ok` (bool); `telemetry_*` (one float32 attribute per included telemetry channel)  |
| 3         | `checksum` (int64); `weight_ndx` (int64)                                                             |

Housekeeping is intentionally stored as records, not as a dense array, because
the heterogeneous attribute set across subtypes does not fit a single rectangular table.

### 8.13 `/calibrator/data`

Optional. Group with one dataset per packet per channel:

```
/calibrator/data/
  @count : int64, number of calibrator data packets
  packet_0_ch_0 : variable-length float32 array
  packet_0_ch_1 : variable-length float32 array
  ...
```

Dataset names follow the `packet_<i>_ch_<j>` convention. Each dataset's shape
is whatever the calibrator decoder produces for that capture/channel.

### 8.14 `/grimm_spectra`

Optional. Same layout as `/spectra` (3D `(N, 16, 2048)` float32 cube plus
`unique_ids`, `raw_times`, `mjd_times`). Group attribute `count = N`.
Written only when Grimm packets are present and decodable into spectral
arrays.

### 8.15 Sample-retention rules

When building the dense science cubes:

1. The cube is preallocated as `float32` filled with `NaN`, with shape
   `(n_time, NPRODUCTS, NCHANNELS)` (and additional axes for TR / zoom).
2. For each time index, only products that actually arrived are written;
   missing products keep their `NaN`s.
3. Reduced-resolution spectra fill only the leading frequency channels;
   the rest stay `NaN`.
4. After the cube is filled, any time row that is **entirely** `NaN`
   (no products at all) is removed. The corresponding entries in
   `unique_ids`, `raw_times`, `mjd_times`, `original_indices`, and every
   dataset under `metadata/` are removed too.
5. The `original_indices` dataset records the row index in the
   pre-trimmed cube for each retained sample, so analysis code can map
   back to the unfiltered stream if needed.

`/spectra_interpolated_telemetry` is trimmed to the same retained spectrum
rows as `/spectra`.

### 8.16 Compression

All dense science datasets and telemetry datasets are compressed with the
HDF5 default gzip (level 4 unless overridden). The `metadata/` sub-datasets
may be gzipped or stored uncompressed depending on size; either is acceptable.

---

## 9. Stage 8 — FITS output

> **Status: subject to change.** All of §9, including every HDU name,
> column TFORM, header keyword, and shape claim below, is provisional.
> FITS is a low-priority output: there are no committed downstream
> consumers yet. The new pipeline is free to redefine the FITS schema as
> needed, drop FITS support entirely, or replace it with another format.
> The description below documents the **current** behavior for reference
> only; do not treat it as a contract.

A FITS file is written alongside the HDF5 file for every session. Its
filename mirrors the HDF5 filename with `.fits` substituted for `.h5`.

The FITS file is a parallel export of the same in-memory products; it is
self-contained and does not reference the HDF5 file. Some HDU shapes are
transposed relative to the HDF5 datasets to match astronomical FITS
conventions (frequency on `NAXIS1`, time on the slowest axis).

### 9.1 Primary HDU

* `BITPIX = 8`, `NAXIS = 0`, `SIMPLE = T`, `EXTEND = T` — empty image.
* Header keywords:

  | Keyword     | Type    | Meaning                                              |
  |-------------|---------|------------------------------------------------------|
  | `CDI_DIR`   | string  | Source uncrater session path                         |
  | `DATE`      | string  | File creation timestamp (ISO-8601, UTC)              |
  | `ORIGIN`    | string  | Producer name (e.g. `LuSEE-Receive`)                 |
  | `N_ITEMS`   | int     | Number of metadata-context groups (informational)    |
  | `N_GROUPS`  | int     | Synonym for `N_ITEMS`, kept for backward compatibility |

### 9.2 Required extension HDUs

These HDUs are always present (some may be empty):

#### `SESSION_INV` — binary table

Single-row table with a placeholder column to satisfy the FITS structural
requirement; the data is in the header keywords.

Columns:

| Name          | TFORM | Notes               |
|---------------|-------|---------------------|
| `PLACEHOLDER` | `J`   | int32 dummy column  |

Header keywords (mirroring §8.4):

| Keyword     | Type | Meaning                            |
|-------------|------|------------------------------------|
| `SW_VERS`   | int  | software version                   |
| `FW_VERS`   | int  | firmware version                   |
| `FW_ID`     | int  | firmware ID                        |
| `FW_DATE`   | int  | firmware build date                |
| `FW_TIME`   | int  | firmware build time                |
| `ST_UID`    | int  | start `unique_packet_id`           |
| `ST_T32`    | int  | start `time_32`                    |
| `ST_T16`    | int  | start `time_16`                    |
| `EXTDESC`   | str  | `"Session invariants"`             |

#### `CONSTANTS` — binary table

Same shape as `SESSION_INV`. Header keywords mirroring §8.5:

| Keyword   | Type   | Meaning                          |
|-----------|--------|----------------------------------|
| `LUN_LAT` | float  | lunar latitude (deg)             |
| `LUN_LON` | float  | lunar longitude (deg)            |
| `LUN_HGT` | float  | lunar height (m)                 |
| `RAWSHFT` | float  | `raw_time_subtract_seconds`      |
| `MJDOFF`  | float  | `mjd_epoch_offset_days`          |
| `EXTDESC` | str    | `"Landing coordinates"`          |

#### `META` — binary table

Concatenated metadata across all spectra. Has `N` rows (one per retained
spectrum). Each metadata field becomes one column. Column names are the
metadata field names, sanitized to fit FITS column-name limits (≤ 16 chars,
deduplicated by appending suffixes when collisions occur).

Column TFORMs are inferred from the field type:

| Python type  | TFORM   |
|--------------|---------|
| `float`/`np.float64` | `D` |
| `int`/`np.int64`     | `K` |
| `bool`               | `L` |
| string               | `<n>A` |
| array of floats      | `<n>D` |
| array of ints        | `<n>K` |

Header keywords:

| Keyword   | Type | Meaning                            |
|-----------|------|------------------------------------|
| `NROWS`   | int  | `N` (rows in the table)            |
| `EXTDESC` | str  | `"Concatenated metadata"`          |

#### `SPEC_P00` … `SPEC_P15` — image HDUs (16 of them)

One image per correlation product. Shape transposed for FITS:

* `NAXIS1 = 2048` (frequency axis, fastest-varying)
* `NAXIS2 = N` (time axis)

Dtype: `float32` (`BITPIX = -32`).

Header keywords:

| Keyword    | Type | Meaning                               |
|------------|------|---------------------------------------|
| `PROD_IDX` | int  | correlation product index (0..15)     |
| `NTIME`    | int  | `N`                                   |
| `NCHANS`   | int  | `2048`                                |
| `AXIS1`    | str  | `"CHANNEL"`                           |
| `AXIS2`    | str  | `"TIME"`                              |
| `EXTDESC`  | str  | `"Spectra product vs time/frequency"` |

#### `SPEC_ID` — binary table

Per-spectrum identity and time:

| Column      | TFORM | Meaning                         |
|-------------|-------|---------------------------------|
| `UNIQUE_ID` | `K`   | per-row `unique_packet_id`      |
| `TIMESTAMP` | `D`   | per-row `raw_seconds`           |
| `MJD_TIME`  | `D`   | per-row MJD                     |

Header keywords:

| Keyword   | Type | Meaning                                    |
|-----------|------|--------------------------------------------|
| `COUNT`   | int  | `N`                                        |
| `EXTDESC` | str  | `"Spectra unique IDs and timestamps"`      |

### 9.3 Optional extension HDUs

#### `TR_P00` … `TR_P15` — image HDUs (when TR spectra present)

* `NAXIS1 = tr_length_max`
* `NAXIS2 = Navg2_max`
* `NAXIS3 = N`
* Dtype `float32`.

Header keywords:

| Keyword    | Type | Meaning                                 |
|------------|------|-----------------------------------------|
| `PROD_IDX` | int  | product index (0..15)                   |
| `NTIME`    | int  | `N`                                     |
| `NAVG2`    | int  | `Navg2_max`                             |
| `TR_LEN`   | int  | `tr_length_max`                         |
| `AXIS1`    | str  | `"TR_BIN"`                              |
| `AXIS2`    | str  | `"AVG2"`                                |
| `AXIS3`    | str  | `"TIME"`                                |
| `EXTDESC`  | str  | `"TR spectra product vs time/bin"`      |

#### `TR_ID` — binary table (when TR spectra present)

Same column set as `SPEC_ID`. Header keyword `EXTDESC = "TR spectra unique
IDs and timestamps"`.

#### Per-metadata-group HDUs

For each metadata-context group `GGG` (zero-padded ordinal of the group),
the following HDUs may be written:

* `WF_GGG_CH<ch>` — binary table with one row per waveform:

  | Column      | TFORM       |
  |-------------|-------------|
  | `WAVEFORM`  | `16384I`    |
  | `TIMESTAMP` | `D`         |

  Header keywords: `ITEM_ID`, `GROUP_ID` (synonym), `CHANNEL`, `COUNT`,
  `EXTDESC = "Waveform data"`.

* `WF_GGG_SUM` — binary table summarizing waveforms in this group:

  | Column   | TFORM | Notes                  |
  |----------|-------|------------------------|
  | `CHANNEL`| `J`   | channel index          |
  | `COUNT`  | `J`   | waveforms on that ch.  |

  Header: `ITEM_ID`, `GROUP_ID`, `TOTCOUNT`, `EXTDESC = "Waveform summary"`.

* `HK_GGG_T<type>` — binary table for one housekeeping subtype. Columns vary
  by subtype (mirroring §8.10's per-`hk_type` attribute set). Common columns:

  | Column      | TFORM | Notes                |
  |-------------|-------|----------------------|
  | `VERSION`   | `J`   |                      |
  | `UNIQUE_ID` | `K`   |                      |
  | `ERRORS`    | `J`   |                      |

  Plus subtype-specific columns:

  * Type 0 — `TIME` (`D`).
  * Type 1 — `ADC_MIN`, `ADC_MAX`, `ADC_MEAN`, `ADC_RMS` (`<n>E`),
    `ACTUAL_GAIN` (`<m>A`).
  * Type 2 — `TIME` (`D`), `OK` (`L`), per-telemetry `TELEM_<NAME>` (`E`).
  * Type 3 — `CHECKSUM` (`K`), `WEIGHT_NDX` (`J`).

  Header keywords: `ITEM_ID`, `GROUP_ID`, `HK_TYPE`, `COUNT`,
  `EXTDESC = "Housekeeping type N"`.

* `HK_GGG_SUM` — summary table:

  | Column    | TFORM |
  |-----------|-------|
  | `HK_TYPE` | `J`   |
  | `COUNT`   | `J`   |

  Header: `ITEM_ID`, `GROUP_ID`, `TOTCOUNT`, `EXTDESC = "Housekeeping summary"`.

* `ZOOM_GGG` — binary table for zoom spectra:

  | Column            | TFORM | Notes |
  |-------------------|-------|-------|
  | `CH1_AUTOCORR`    | `64E` |       |
  | `CH2_AUTOCORR`    | `64E` |       |
  | `CORR_REAL`       | `64E` |       |
  | `CORR_IMAG`       | `64E` |       |
  | `UNIQUE_ID`       | `K`   |       |
  | `PFB_INDEX`       | `J`   |       |
  | `TIMESTAMP`       | `D`   |       |
  | `MJD_TIME`        | `D`   |       |

  Header: `ITEM_ID`, `GROUP_ID`, `COUNT`, `FFT_SIZE = 64`,
  `EXTDESC = "Calibrator zoom spectra"`.

* `CALDAT_GGG` — binary table for calibrator data:

  | Column        | TFORM        | Notes                         |
  |---------------|--------------|-------------------------------|
  | `PACKET_IDX`  | `J`          |                               |
  | `CHANNEL_IDX` | `J`          |                               |
  | `DATA_LEN`    | `J`          | actual length used            |
  | `DATA`        | `<max_len>E` | padded with zeros to max_len  |

  Header: `ITEM_ID`, `GROUP_ID`, `COUNT`, `NROWS`, `MAX_LEN`,
  `EXTDESC = "Calibrator data"`.

#### Telemetry HDUs

When telemetry is available (single-pass raw-flash conversion, or
existing-session mode with a legacy `DCB_telemetry.json` sidecar), one
image HDU per telemetry field:

* HDU name: `DCB_<field_name>` (e.g. `DCB_THERM_FPGA`, `DCB_VMON_6V`).
* Shape: `(M,)` (1D image).
* Dtype: `float64`.
* Header keywords: `FIELD = "<field_name>"`, `NROWS = M`,
  `EXTDESC = "DCB telemetry field"`.

The encoder fields produce `DCB_encoder_*` HDUs analogously when present.

### 9.4 FITS / HDF5 consistency

> **Status: subject to change** (see §9 banner).

In an ideal pipeline the FITS HDU and the HDF5 dataset for every dense
science product would be generated from the same in-memory array, with
the only differences being:

* axis ordering (FITS puts the fastest-varying axis first),
* per-extension headers vs. per-group attributes,
* compression (FITS extensions uncompressed; HDF5 datasets gzipped).

The current code does **not** maintain this strict lockstep:

* HDF5 drops all-NaN spectra rows (§8.15 retention rules); the FITS
  spectra writer does not apply the same retention filter, so its time
  axis can be longer than the HDF5 one.
* For TR spectra, when per-sample dimensions vary, the FITS path falls
  back to a per-item layout instead of the max-padded cube used by HDF5.

A re-implementation that keeps writing FITS should restore strict
parity, or — equally acceptable — drop FITS and use only HDF5.

---

## 10. DCB telemetry decoding and engineering-unit conversions

The detailed DCB / encoder telemetry layout, channel names, and
engineering-unit conversion formulas are restricted; they live with the
private [`lusee_telemetry`][lusee-tel] package. The public pipeline
calls into that package via :mod:`lusee.ingest.telemetry`. When the
package is not on `sys.path` (or `LUSEE_TELEMETRY_PATH` is not set),
the pipeline still produces HDF5 / FITS output -- only the
``/DCB_telemetry/`` group is omitted.

[lusee-tel]: https://github.com/lusee-night/lusee_telemetry

The contract the public pipeline enforces, regardless of decoder
specifics:

* **Error handling.** A math error in any per-field conversion sets
  that one sample to `0.0` and emits a deduplicated warning per field.
* **Bad-packet diagnostics.** The decoder may flag a record as bad and
  emit a warning; values still flow through to the datasets.
* **Telemetry-to-session assignment.** In raw flash mode each telemetry
  record is assigned to a session by mission time, with the same rule
  used for science packets: ``bisect_right(starts, mission_seconds) - 1``,
  records before the first session attach to the first session.
* **Per-session output.** One ``(fpga_arrays, encoder_arrays)`` pair
  per session, each a dict mapping channel name to a 1D NumPy array.

### 10.1 Encoder telemetry HDF5 layout

Encoder fields land under the ``encoder_`` prefix:

| HDF5 dataset path                        | Type     | Meaning                                |
|------------------------------------------|----------|----------------------------------------|
| `/DCB_telemetry/encoder_mission_seconds` | float64  | Same as DCB telemetry                  |
| `/DCB_telemetry/encoder_lusee_subsecs`   | float64  | Same as DCB telemetry                  |
| `/DCB_telemetry/enc_pos`                 | int      | Encoder counts                         |
| `/DCB_telemetry/enc_status`              | int/bits | Status flags                           |

---

## 11. Time and identity model

### 11.1 The three identifiers

* `unique_packet_id` — 32-bit logical identifier assigned by the producer.
  Drives sessionization and product correlation. Distinct from
  `sequence_cnt`.
* `sequence_cnt` — 14-bit CCSDS counter, used only at the transport layer
  for segment reassembly. Not preserved past Stage 2 in the persisted
  artefacts.
* Mission time — split into `time_32` (lower 32 bits, integer fixed-point)
  and `time_16` (upper 16 bits) on packets that carry it. Combined per the
  formula in §6.1 to produce `raw_seconds`.

### 11.2 Time conversion summary

| Quantity            | Formula                                                          |
|---------------------|------------------------------------------------------------------|
| `raw_seconds`       | `((((time_16 & 0xFFFF) << 32) + time_32) >> 4) / 4096`           |
| `mjd_time`          | `(raw_seconds - raw_time_subtract_seconds) / 86400 + mjd_epoch_offset_days` |
| Telemetry time      | `mission_seconds + lusee_subsecs / 65536`                        |
| Filename timestamp  | `YYYYMMDD_HHMMSS` of session-start `raw_seconds` reinterpreted as Unix seconds (see note) |

> **Note on the filename timestamp time zone.** The current code formats
> this timestamp using local time (`datetime.fromtimestamp` without an
> explicit tz). A re-implementation should use **UTC**
> (`datetime.fromtimestamp(..., tz=timezone.utc)` or
> `datetime.utcfromtimestamp`) so that filenames are stable across
> machines and time zones. Existing files generated under the old rule
> will have local-time stamps; the new rule applies only to newly
> produced files.

The MJD calculation is currently a stub: with the default constants it
produces `raw_seconds / 86400`, which is not a true MJD. Once the mission
provides calibrated values for `raw_time_subtract_seconds` and
`mjd_epoch_offset_days`, the same code path produces correct MJDs without
schema changes.

---

## 12. Bad-data and edge-case handling

| Condition                                 | Current behavior                                   |
|-------------------------------------------|----------------------------------------------------|
| CRC mismatch in CCSDS frame               | Fatal — abort the bank file                        |
| Unexpected APID in `b01`                  | Drop the packet                                    |
| Segmented telemetry packet                | Hard error                                         |
| APID changes mid logical packet           | Warn, keep accumulating with new APID              |
| Logical packet has no extractable ID      | Drop the packet                                    |
| Class-C packet with no preceding ID       | Drop the packet                                    |
| Session has no startup packet             | `session_NNN.h5` (no timestamp suffix), `session_invariants` may be sparse |
| Telemetry math error (e.g. `log(neg)`)    | Required: catch, store `0.0`, emit a warning (uniform across single-pass and legacy-sidecar paths — see §10.1) |
| Bad telemetry packet (`SPE_1VA8_V < 15`)  | Flag and dump blob to a sidecar file; values still stored |
| Spectrum row with no products             | Drop row from `/spectra/data` and aligned arrays   |
| Reduced-resolution spectrum               | Trailing channels stay `NaN`                       |
| Missing science bank file                 | Skip bank, continue with present banks             |
| Missing legacy `DCB_telemetry.json` sidecar | Omit `/DCB_telemetry` group (existing-session mode) |
| Existing-session input vs. saved sim run  | Same handling; telemetry just absent if no sidecar |

A re-implementation is free to soften the fatal CRC behavior to a warn-and-
resync, but must clearly document the chosen semantics in this same spec.

---

## 13. Verification

A re-implementation can be verified end-to-end as follows.

### 13.1 Outputs that must match

For a fixed input — either a `FLASH_TLMFS` directory or a saved
`session_*` directory — the new pipeline must produce HDF5 and FITS files
that are equivalent to the current ones up to:

* Floating-point rounding within 1 ULP for engineering-unit conversions.
* Order of attribute definition in HDF5 groups.
* Order of HDU keywords in FITS headers.
* Bit-exact agreement on:
  * All science array shapes and byte layouts.
  * `unique_ids`, `original_indices`, `sequence_cnt` (where preserved).
  * `cdi_output/` filenames and contents.

### 13.2 Recommended cross-checks

1. **CCSDS round-trip on a known good bank file** — for each emitted
   logical packet, recompute the per-frame CRCs against the stored payloads
   and confirm they would have validated under the original parser. (This
   tests Stages 1-2 without depending on Stage 3+.)
2. **Filename consistency** — list `cdi_output/` for the new and old
   pipelines on the same input; the lists must be identical.
3. **HDF5 binary diff** — a tool such as `h5diff` should produce no
   differences on `/spectra/data`, `/tr_spectra/data`,
   `/calibrator/zoom_spectra/data`, `/waveform/channel_*/waveforms`, and
   `/DCB_telemetry/*` (when applicable).
4. **FITS structural diff** — `astropy.io.fits.diff.FITSDiff` (with a
   floating-point tolerance) should produce no significant differences.
5. **Telemetry presence/absence** — verify that `/DCB_telemetry` is present
   exactly when the input is either a single-pass raw-flash conversion
   that yielded telemetry, or an existing-session input with a legacy
   `DCB_telemetry.json` sidecar, and is absent otherwise.
6. **Corrupted telemetry tolerance** — feed in the existing fixture
   sessions whose telemetry produces non-physical intermediate values; the
   pipeline must complete and produce zeroed (or otherwise sentinel)
   values for the affected fields without crashing.
7. **Multi-session bank** — point the pipeline at a flash directory that
   contains several startup packets and confirm that exactly one HDF5/FITS
   pair is produced per session, with the correct `session_NNN_*` names
   and disjoint packet sets.

### 13.3 Reference fixtures

The repository's existing `receive/` directory contains several reference
sessions and their HDF5/FITS outputs (e.g. `session_000.h5`,
`session_001_*.h5`, `session_003_*.h5`). These are suitable for §13.1's
"must match" comparison. Note that some of them carry corrupted telemetry
(see Context); the new pipeline should produce identical sentinel-zero
behavior on those fields.

---

## 14. Summary of constants and magic numbers

| Constant / magic                          | Value                       |
|-------------------------------------------|-----------------------------|
| Sync word                                 | `0xECA0`                    |
| Padding byte                              | `0xA5`                      |
| CCSDS primary header size                 | 6 bytes                     |
| CCSDS CRC size                            | 2 bytes                     |
| CRC-16-CCITT polynomial                   | `0x1021`                    |
| CRC-16-CCITT initial value                | `0xFFFF`                    |
| Telemetry APIDs                           | restricted (see private package) |
| Telemetry record / payload sizes          | restricted                  |
| Legacy sidecar filename (read-only)       | `DCB_telemetry.json` (binary) |
| Spectra cube shape                        | `(N, 16, 2048)` float32     |
| TR spectra cube shape                     | `(N, 16, Navg2_max, tr_length_max)` float32 |
| Zoom spectra cube shape                   | `(N, 4, 64)` float32        |
| Waveform sample count per packet          | 16384                       |
| HDF5 layout version                       | `2`                         |
| Filename packet-index width               | 5 (or 6 if ≥1,000,000 packets) |
| Filename APID width                       | 4 hex digits                |
| Mission-time fixed-point fractional bits  | 12                          |
| Telemetry sub-second LSB                  | `1/65536` second            |

This table is the complete set of magic numbers an implementor should keep
in mind; everything else flows from the schema definitions of the external
`uncrater` and `pycoreloop`-style packet libraries and from the conversion
formulas in §10.
