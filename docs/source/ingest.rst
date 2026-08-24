Downlink ingestion contracts
============================

The ``lusee.ingest`` package decodes raw CCSDS downlinks or extracted
``uncrater`` sessions. Decoder-derived products carry immutable decoder and
packet provenance records. Compatibility rows produced by the older adapters
are marked ``legacy_adapter_provenance_pending`` and are not counted as
validated product rows. Strict, family-specific product adoption and layout-v4
output are being introduced incrementally; the existing writers remain layout
v3 until that cutover.

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

Waveforms are consumed from metadata-associated groups as signed ``int16
(16384,)`` rows. Channel, UID, split mission time, and the independent
``uint64`` ADC timestamp are validated without truthiness defaults. Provenance
links the waveform and metadata packet. Orphans, duplicate channels, and
groups rejected upstream create no waveform row.

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

Waveform clocks in layout v3
----------------------------

Waveform mission time and the ADC hardware counter are separate clock domains.
The additive layout-v3 fields are:

* HDF5: ``timestamps`` (mission seconds, NaN when absent),
  ``adc_timestamps`` (``uint64``), and ``adc_timestamp_valid``;
* FITS: ``TIMESTAMP``, ``ADC_TIME``, and ``ADC_VALID``. ``ADC_TIME`` is a
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
