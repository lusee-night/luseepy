Downlink ingestion contracts
============================

The ``lusee.ingest`` package decodes raw CCSDS downlinks or extracted
``uncrater`` sessions. Decoder-derived products carry immutable decoder and
packet provenance records. Compatibility rows produced by the older adapters
are marked ``legacy_adapter_provenance_pending`` and are not counted as
validated product rows. The layout-v4 HDF5 and FITS writers accept only the
strict, family-specific ``WriteRequest`` boundary. ``process_session`` and
``process_flash`` use that boundary and write manifest-v3 run results.

Installation and command line
-----------------------------

Install the ingestion dependencies from the checkout:

.. code-block:: console

   pip install ".[ingest]"

The ingest extra carries the reviewed ``uncrater`` revision pinned in
``pyproject.toml``. That exact pin is authoritative until a compatible release
is reviewed. The production command has three operations:

.. code-block:: console

   lusee-ingest process-session extracted/session_000 --landing-time-file landing.json --h5-dir products/h5 --fits-dir products/fits --manifest-dir products/manifests
   lusee-ingest process-flash FLASH_TLMFS --landing-time-file landing.json --sessions-root extracted --h5-dir products/h5 --fits-dir products/fits
   lusee-ingest validate products/h5/session_000.h5

``process-session`` requires an explicit external manifest directory.
Successful raw FLASH processing writes the canonical
``sessions_root/flash.json``. A failure before output preflight writes no
manifest because no output destination has been accepted yet.
``--overwrite`` is required to replace an existing destination. Plot output
requires both ``--plots-dir`` and ``--h5-dir``; plots use the persisted v4
masks and preserve separate frequency grids.

HDF5, FITS, and extracted-session outputs are written directly to their
requested destinations. A failed write can leave a partial destination;
manual reruns may remove it or use ``--overwrite``.

``--issue-policy`` controls whether the shared issue collector accumulates
findings or stops at the first one. ``--decoder-strict`` separately controls
the repaired decoder's execution mode. ``--schema-variant`` is an explicit
decoder override, including the reviewed early/final 306 distinction. One raw
FLASH input must select one binding across all derived sessions; a mismatch is
reported before any science product is written.

Pipeline, manifests, and manual operation
-----------------------------------------

Raw FLASH processing uses the existing legacy frame recovery and logical
packet reassembly, then the established UID/order heuristic, session split,
repaired decoder, and layout-v4 writers. The CLI does not activate an
alternate CCSDS profile or comparison path. Extracted-session processing
starts at the repaired-decoder stage.

Session and FLASH manifests record the source identity, decoder binding,
issues, stage and family counts, timing reference, output locations, and final
``clean|partial|failed`` status. A decoded family in a manifest-only run uses
``decoded_not_persisted`` and never claims persisted rows.

Operation assumes one ingestion process and an ordinary local filesystem.
There is no daemon, worker pool, lock, resume protocol, or automatic retry.
After a failure, inspect the emitted manifest when present, correct the input
or command, and rerun manually; use ``--overwrite`` when replacing the failed
run's destinations.

Fixed optional DCB telemetry
----------------------------

Telemetry has two frozen input forms: complete logical ``0x314`` packets with
the existing 57-field record, and the historic binary
``DCB_telemetry.json`` sidecar. Despite its suffix, the sidecar is not JSON
text. ``process-flash`` decodes ``0x314`` once and assigns disjoint row slices
to science sessions. ``process-session`` uses only the sidecar in that
session, when present; it does not reopen FLASH or try another telemetry
source. ``0x325`` encoder packets are not part of this path.

Both forms use the optional private ``lusee_telemetry`` package. If that
package is absent, fails to import, raises, or returns malformed data,
ingestion warns once, omits telemetry, and continues otherwise valid science
output successfully. A telemetry problem does not change science family,
root, run, or command-exit quality. No public decoder negotiation,
interpolation, source fallback, or encoder interpretation is performed.

When decoding succeeds, HDF5 and FITS store the same fixed table: source kind,
57 field names and units, source indices, integer ``mission_seconds`` and
``lusee_subsecs``, optional calibrated MJD, raw ``uint16`` counts,
engineering values, and their validity mask. Invalid engineering values are
NaN with a false validity bit; raw counts remain available. A decoded
zero-byte sidecar is a valid zero-row table.

Caller-owned gain telemetry
---------------------------

Public users can run gain conversion without the private decoder by manually
supplying all six gain-model inputs for each call. The following values are an
illustrative operating assumption used by an existing public golden test;
they were not measured in, or derived from, the good telemetry trees and must
not be described as representative flight telemetry. The temperatures are
mutually consistent, the voltages are close to nominal 1.8 V and 1.2 V rails,
and 45 mA is physically plausible but not independently validated.

.. code-block:: python

   assumed_telemetry = {
       "THERM_FPGA": 30.4,    # degC
       "SPE_ADC0_T": 29.8,    # degC
       "SPE_ADC1_T": 28.5,    # degC
       "SPE_1VAD8_V": 1.799,  # V
       "VMON_1V2D": 1.201,    # V
       "SPE_1VAD8_C": 0.045,  # A
   }

   physical = data.to_physical(telemetry=assumed_telemetry)
   physical_psd = data.to_physical_psd(telemetry=assumed_telemetry)

This dictionary is strictly opt-in and caller-owned. Construct it explicitly
and pass it through ``telemetry=`` on every conversion call. Each value may be
a scalar, broadcast only for that call, or an array of exact shape
``(Nspectra,)``. The package never defines, selects, caches, or persists these
assumptions and never substitutes them when decoded row-aligned telemetry is
absent. With neither row-aligned telemetry nor an explicit ``telemetry=``
mapping, both conversion methods raise the existing missing-input error.

Status, issues, and exits
-------------------------

Packet- or product-local damage removes only the smallest affected unit.
Independent products continue, and fixed-shape spectral storage receives NaN
where a product plane is absent or rejected. Issue policy is independent of
the aggregate quality status.

Completed processing prints the final status, artifact and manifest paths, and
sorted issue counts. An unexpected processing, preflight, or command-line
exception prints a concise error; any direct partial output is left for manual
inspection and rerun. Processing and validation return ``0`` for clean, ``2``
for usable partial data, and ``1`` for failed, invalid, or command-line input.

Validation and compatibility
----------------------------

``lusee-ingest validate`` applies the full reader contract to one layout-v4
HDF5 or FITS product. For manifests it accepts the canonical in-session
``session.json`` or canonical ``flash.json`` and checks current schema and
run/session linkage. Manifest validation does not replay raw banks or
revalidate a referenced HDF5/FITS file; validate each product separately.

Layout v2 and v3 remain read-only compatibility inputs. Validation does not
upgrade them. Re-ingest the original raw FLASH or extracted packet tree to
produce a current v4 product and manifest-v3 provenance.

Quality and execution mode
--------------------------

Decoder execution mode is either ``collect`` or ``strict`` and is independent
of aggregate data quality. Issue-collecting operation retains usable products;
any decoder issue or rejected packet makes a usable aggregate ``partial``.
An aggregate with no usable payload is ``failed``. A fully usable aggregate
with no reported decoder damage is ``clean``. Caller-built ``Products`` remain
unassessed (``quality_status is None``) until a decoder boundary classifies
them.

Repaired-decoder spectrum boundary
----------------------------------

``read_uncrater_session`` imports each public uncrater decoder issue exactly
once into the caller's ``IssueCollector``. Fatal decoder findings are
``error/dropped``; nonfatal diagnostics are ``warning/kept``. Product
provenance links those issue IDs to concrete packet indices and original and
normalized AppIDs. The adapter does not invent packet filenames.

Science metadata is normalized into a frozen ``SpectrumMetadata`` record for
the reviewed ``203``, ``305``, ``306-early``, ``306-final``, and ``307``
bindings. Required vectors have exact lengths. Split mission time is recomputed
from ``time_32`` and ``time_16``; missing or inconsistent fields reject that
metadata row instead of receiving a zero default.

Normal spectra remain native-width ``float32`` records of shape
``(16, Nfreq)``. The adapter validates ``Nfreq`` against ``Navgf``, applies the
established bit-slice restoration exactly once, and keeps missing or rejected
products as absent planes. TR spectra remain unscaled native ``int32`` records
of shape ``(16, Navg2, Ntr)`` with geometry derived from the metadata settings.
A metadata-only group does not create a fabricated science row. Duplicate or
malformed products are dropped individually, so valid siblings survive.
The top-level ``lusee.ingest`` spectrum exports are these validated records.
The mutable rows still defined inside ``lusee.ingest.decode`` exist only for
layout-v2/v3 caller-built compatibility and are excluded from validated
product counts pending the layout-v4 cutover.

Repaired-decoder auxiliary boundary
-----------------------------------

Auxiliary products are adopted only from repaired uncrater's typed packets
and complete public groups. Exact native shapes and dtypes are validated
before an immutable luseepy row is created. A malformed or orphaned packet is
reported and omitted; it never becomes a plausible all-zero row.

Zoom rows are native ``float32 (4, 64)`` arrays in ``AA, BB, ABR, ABI`` order
and retain ``pfb_bin``. Grimm rows remain native ``int32
(Navg2, 16, 4)``; they are not normal spectra and receive no normal-spectrum
scaling. Either family receives mission time only from one exact-UID science
metadata match. Missing or ambiguous association never borrows a preceding
row's time.

Every valid waveform is retained as a signed ``int16 (16384,)`` row, even
when metadata is missing or ambiguous. The decoder considers the complete
interval between same-stream Hello/EOS boundaries, independently of metadata arrival
between waveform packets. Firmware sends one selected channel or an ordered
four-channel capture; only associations shared by every possible grouping
are accepted. Unexplained counts, duplicate metadata UIDs, or backward times leave metadata
unresolved, without discarding the samples or interpolating any field.

Associated rows retain the metadata UID, split mission time, exact ``uint64``
ADC timestamp, and both packet references. Unresolved rows use ``uint32`` UID
``0``, an empty UID source, NaN mission/absolute times, and an invalid ADC
timestamp. A genuine metadata UID zero remains distinguishable through its
nonempty source and metadata reference. Unmatched metadata values and
association diagnostics remain in the stored canonical decoder report.

FLASH extraction validates associations across the full input in original
order within each bank. Matched samples are routed to their metadata packet's
session; unresolved samples retain their heuristic session placement. UIDs
are repaired after ordering and before writing the packet map. Format-2 maps
preserve source order, CCSDS spans, and an explicit metadata reference or null
for every waveform. Subsequent decoding replays those decisions without
rematching session subsets. Bank concatenation cannot establish cross-bank
Hello/EOS placement, and the CCSDS counters are not assumed to be per APID.
Observed framing damage or relevant reassembly loss leaves waveform metadata
unresolved for the affected bank; dropped packets cannot silently change the
inferred capture mode.
Re-ingest raw input to replace older packet maps. Run again with all accumulated packets
and ``overwrite=True`` to replace incomplete outputs after late arrivals.
An incomplete interval may leave earlier pairs ambiguous too; ingestion never
assumes that all missing packets belong at the tail. Without reliable loss
evidence, losses compatible with another capture grouping are undetectable:
four singleton channels with three lost metadata packets resemble one intact
four-channel capture.

Housekeeping types ``0``, ``1``, ``2``, ``3``, ``100``, and ``101`` use one
normalized field union with explicit per-field presence. ADC statistics retain
all valid, invalid, and total counts plus a per-channel statistics-valid mask;
telemetry is flattened through its reviewed four-field mapping. Firmware
``errors`` remain separate from decoder issues. Only types 0 and 2 carry a
real mission time; uncrater's zero initialization for the other types is not a
timestamp.

Direct calibrator metadata and complete data, raw-PFB, and debug groups retain
their UID, native arrays, all contributing page packet references, and raw
mission time. Multipart rows are accepted only when every required page is
valid, ordered, binding-consistent, and has the same UID.
Compatibility aggregates such as ``calib_data``, ``calib_pfb``, ``cd_*``, and
stacked Grimm arrays are not product inputs because they discard packet/page
identity. Every multipart page clock is retained; page zero supplies the row's
reference ``raw_seconds`` without requiring later page clocks to be equal.
FW-direct spectra remain unsupported until their payload contract is
established.

Dense spectrum storage
----------------------

Layout 4 defines the same rectangular HDF5 and FITS storage convention. Normal
spectra are ``(time, 16, 2048)`` in restored-SDU ``float32``. A row's native
values occupy the decoded prefix; its ``Navgf`` and native frequency count
identify that prefix. The unproduced tail and every absent or rejected product
plane are NaN.

TR storage is ``(time, 16, Navg2, Ntr)`` using the session's validated TR
geometry. It is ``float64`` on disk so every native ``int32`` value is exact
while absent or rejected products remain NaN. Mixed TR geometries in one
session are rejected. The strict in-memory record keeps its integer payload
and internal product-presence mask. No separate persisted bad-frame,
element-valid, or product-presence flag array is added at this stage.

The full on-disk contract is documented in :doc:`ingest_layout_v4`.

Waveform clocks in layout v4
----------------------------

Waveform mission time and the ADC hardware counter are separate clock domains.
The layout-v4 fields are:

* HDF5: ``raw_seconds`` plus ``raw_time_valid`` (mission seconds),
  ``adc_timestamps`` (``uint64``), and ``adc_timestamp_valid``;
* FITS: the same canonical lowercase column names. ``adc_timestamps`` is a
  ``K`` column with unsigned scaling ``TZERO = 2**63`` and round-trips through
  Astropy and fitsio, including ``UINT64_MAX``.

An ADC timestamp is valid only when repaired uncrater attached the public
waveform metadata record. Its orphan-packet all-ones sentinel is never treated
as a measurement.

Spectrometer frequency windows
------------------------------

``spectrometer_frequency_window(Navgf)`` records the reviewed coreloop 3r09
native-bin membership:

.. list-table::
   :header-rows: 1

   * - ``Navgf``
     - Output bins
     - Native bins for output ``i``
     - Divisor
   * - 1
     - 2048
     - ``i``
     - 1
   * - 2
     - 1024
     - ``2i, 2i+1``
     - 2
   * - 3
     - 512
     - ``4i, 4i+1, 4i+2``
     - 4
   * - 4
     - 512
     - ``4i, 4i+1, 4i+2, 4i+3``
     - 4

The exposed floating-point weights are nominal linear-response coefficients,
not a claim about bit-exact firmware rounding. The native-bin origin and the
physical coordinate assigned to an averaged bin are unresolved. The contract
therefore records ``frequency_coordinate_status="unresolved"`` and refuses to
produce ``frequency_mhz`` rather than inventing a coordinate.

External clock reference
------------------------

Absolute time conversion requires one UTF-8 JSON file loaded with
``load_clock_reference_set``. Format 1 has exactly these fields:

.. code-block:: json

   {
     "format_version": 1,
     "reference_event": "landing",
     "clock_reference_isot": "2024-03-04T12:34:56",
     "time_scale": "utc",
     "clocks": {
       "spectrometer": {"clock_reference_raw_seconds": 123.0},
       "dcb": {"clock_reference_raw_seconds": 456.0}
     },
     "source": "mission timing record",
     "assumed": false
   }

Unknown fields, duplicate keys, nonfinite numbers, unsupported clock names,
and invalid time scales are rejected. The normalized record retains the exact
source-file SHA-256. For each covered clock, conversion is

.. code-block:: text

   absolute_time = clock_reference_isot
                 + (raw_seconds - clock_reference_raw_seconds) seconds

Spectrometer and DCB clocks have separate anchors. ADC hardware timestamps
remain independent and unmapped in format 1. Missing coverage leaves a product
on raw time; it never borrows another clock's anchor.

Raw FLASH CLI processing always requires ``--landing-time-file``. An extracted
session may instead reuse a verified embedded clock-reference set. The CLI has
no loose epoch, scale, or offset flags: ``assume_scale`` alone never authorizes
absolute time.
