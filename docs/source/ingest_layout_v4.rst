Ingest layout version 4
=======================

Layout 4 is the first ingest format written from a validated, format-neutral
``WriteRequest``. HDF5 and FITS implement the same science and provenance
tree. Layout versions 2 and 3 remain read-only compatibility formats.

Write contract
--------------

The request is validated in full before an output file is opened. It contains
strict decoded ``Products``, the complete issue set, one status for every known
product family, lunar location, clock-reference availability, portable input
identity or an explicit reason it is unavailable, optional fixed
``TelemetryData``, and writer options. Legacy mutable decoder rows, anonymous
calibrator arrays, and untyped telemetry mappings are not accepted.

Existing destinations are refused by default. An explicit ``overwrite=True``
is recorded in run provenance. Writers write directly to the requested path
and verify the closed output. If writing or verification fails, a partial
output may remain; inspect it, then remove it or rerun with overwrite.

Common provenance
-----------------

The layout-4 serializations preserve:

* root clean/partial quality and issue counts;
* session invariants and lunar location;
* the complete decoder/schema report;
* normalized issues and product-to-issue references;
* product-row and source-packet provenance;
* per-family support, coverage, quality, row counts, and issue references;
* the complete external clock-reference record, or an explicit unavailable
  reason; and
* portable run/input identity, optional absolute source path, compression
  policy, and overwrite provenance.

Required values are never replaced with plausible zeroes. Optional scalar
metadata uses explicit presence information. Raw mission time and its clock
source remain authoritative; calibrated MJD is present only where the supplied
clock-reference set covers that source.

Spectrum arrays
---------------

Normal spectra use one dense ``float32`` array with shape
``(time, 16, 2048)`` in restored-SDU representation. Native decoded values
occupy the prefix selected by the row's frequency-window contract. Missing or
rejected products and the unused tail are NaN. ``Navgf``, native frequency
count, and the versioned frequency-window contract are stored per row or by
indexed contract. No element-valid, product-present, or bad-frame array is
persisted in layout 4.

Time-resolved spectra use one dense ``float64`` array with shape
``(time, 16, Navg2, Ntr)``. ``Navg2`` and ``Ntr`` come from the validated
homogeneous session settings. Native ``int32`` values are represented exactly;
missing or rejected products are NaN. Mixed time-resolved geometries in one
request are rejected, and no separate product-present mask is persisted.

Other product families
----------------------

Zoom spectra are ``(time, 4, 64)`` with ``AA``, ``BB``, ``ABR``, and ``ABI``
component labels and an explicit PFB bin. Waveforms are row-aligned signed
``int16`` arrays of 16,384 samples with channel, mission time, and exact
``uint64`` ADC timestamp kept distinct. Grimm products retain the native
``(time, Navg2_max, 16, 4)`` integer geometry and average-valid padding.

Waveform rows without associated metadata retain their samples and channel.
``unique_ids`` stays ``uint32`` and uses ``0`` with an empty provenance
``uid_source`` for missing UID. Mission and MJD times are NaN with their
existing validity flags false. ``adc_timestamps`` stays ``uint64`` and uses
``0`` with ``adc_timestamp_valid=false``. Valid metadata containing zero UID
or zero ADC timestamp remains valid and retains its metadata source reference.
No additional datasets or layout-version change are required. The canonical
decoder report also retains unmatched metadata values and association details.

Housekeeping, calibrator metadata, and calibrator debug pages use a normalized
field union with per-field presence. Calibrator complex arrays are serialized
as paired real and imaginary arrays. Every calibrator page retains its raw
clock and optional calibrated time. A known family that is not implemented is
listed explicitly as unsupported rather than silently omitted.

Optional DCB telemetry
----------------------

``/telemetry`` is present only when the optional private decoder returned one
valid fixed ``TelemetryData`` table. A decoded zero-byte legacy sidecar still
creates the group with zero rows. Telemetry is not a science product family,
does not contribute issues to root quality, and has no interpolation policy.

The group attribute ``source_kind`` is ``b01_0x314`` or
``legacy_binary_sidecar``. It contains exactly ``field_names``, ``units``,
``source_indices``, ``mission_seconds``, ``lusee_subsecs``, ``mjd_times``,
``raw_counts``, ``values``, and ``valid``. There are exactly 57 columns in
decoder order. Source time and counts retain their integer dtypes; engineering
values are ``float64`` and invalid cells are NaN exactly where ``valid`` is
false. ``raw_seconds`` is derived by the reader and is not persisted. Decoder
identity, dynamic metadata, private issues, encoder data, unassigned rows, and
interpolated telemetry are not part of layout 4.

Finite telemetry MJD values require a DCB entry in the stored clock-reference
set and must equal that reference applied to
``mission_seconds + lusee_subsecs / 65536``. Without a DCB entry every
telemetry MJD value is NaN. The strict HDF5 and FITS readers repeat this check,
and format parity covers the complete fixed table.

HDF5 organization
-----------------

The HDF5 root attribute ``layout_version`` is ``4``. Common records live under
``/session_invariants``, ``/constants``, ``/clock_reference``,
``/run_provenance``, ``/issues``, ``/provenance``, and ``/status``. Science
families use ``/spectra``, ``/tr_spectra``, ``/waveform``,
``/grimm_spectra``, ``/housekeeping``, and ``/calibrator``. Each emitted family
has a row count plus row-aligned identity, raw-time validity, optional MJD
validity, original-index, and product-provenance references.
The optional fixed DCB table uses ``/telemetry`` as described above.

FITS organization
-----------------

The FITS primary HDU records ``LAYOUTV = 4`` and ``FITSFMT = 1``. Every
layout group is identified by its canonical absolute path in ``LUSEEPTH``.
Row-aligned datasets use one binary table with the same lowercase public names
as HDF5 when they fit in a FITS header card. Longer normalized field names use
short physical ``TTYPE`` aliases recorded in ``COLJSON``. A group whose direct
datasets have different row counts, or exceed FITS's 999-column limit, uses a
header-only group HDU followed by deterministic table partitions. ``ATTRJSON``
and ``COLJSON`` are versioned, ASCII JSON transport records for exact logical
attribute values, column dtypes, shapes, and encodings; non-ASCII text itself
is stored as UTF-8 bytes in ``B`` columns rather than being forced into FITS
ASCII strings. Fixed-width byte arrays likewise use exact raw-byte columns.

FITS signed integers, floats, complex values, logical values, and
multidimensional cell shapes retain their canonical widths. Unsigned 16-,
32-, and 64-bit columns use FITS unsigned scaling. In particular, waveform
``adc_timestamps`` is a ``K`` column with ``TZERO = 2**63`` and an independent
validity column, so ``UINT64_MAX`` round-trips exactly through Astropy and
fitsio. The transport encoding does not change the logical public dtype.

The writer validates the complete request before opening the output, writes
checksums for every HDU, reopens the closed file with unsigned reading enabled,
verifies structure and every checksum explicitly, reconstructs the logical
layout tree, and requires exact attribute/dataset parity. HDF5 and FITS
therefore preserve the same strict product rows; neither serializer is a
layout-v2/v3 writing path.
