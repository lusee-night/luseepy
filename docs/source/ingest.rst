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
