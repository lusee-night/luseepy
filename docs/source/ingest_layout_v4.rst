Ingest layout version 4
=======================

Layout 4 is the first ingest format written from a validated, format-neutral
``WriteRequest``. HDF5 implements the contract described here. The legacy FITS
writer remains explicitly labelled layout 3 until its separate layout-4
serializer implements the same science and provenance contract. Layout
versions 2 and 3 remain read-only compatibility formats once that migration is
complete.

Write contract
--------------

The request is validated in full before an output file is opened. It contains
strict decoded ``Products``, the complete issue set, one status for every known
product family, lunar location, clock-reference availability, portable input
identity or an explicit reason it is unavailable, interpolation policy, and
writer options. Legacy mutable decoder rows, anonymous calibrator arrays, and
untyped telemetry mappings are not accepted.

Existing destinations are refused by default. An explicit ``overwrite=True``
is recorded in run provenance. Writers create and validate a sibling temporary
file and then install it atomically, so a failed write does not damage an
existing product.

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
* portable run/input identity, optional absolute source path, interpolation
  policy, compression policy, and overwrite provenance.

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

Housekeeping, calibrator metadata, and calibrator debug pages use a normalized
field union with per-field presence. Calibrator complex arrays are serialized
as paired real and imaginary arrays. Every calibrator page retains its raw
clock and optional calibrated time. A known family that is not implemented is
listed explicitly as unsupported rather than silently omitted.

HDF5 organization
-----------------

The HDF5 root attribute ``layout_version`` is ``4``. Common records live under
``/session_invariants``, ``/constants``, ``/clock_reference``,
``/run_provenance``, ``/issues``, ``/provenance``, and ``/status``. Science
families use ``/spectra``, ``/tr_spectra``, ``/waveform``,
``/grimm_spectra``, ``/housekeeping``, and ``/calibrator``. Each emitted family
has a row count plus row-aligned identity, raw-time validity, optional MJD
validity, original-index, and product-provenance references.

The layout-4 FITS serialization uses ``LAYOUTV = 4`` and equivalent named
images/tables. The canonical reader normalizes both serializations to one
bundle contract; FITS-specific integer encoding is an on-disk transport detail
and does not change the public dtype. The existing legacy FITS writer still
emits ``LAYOUTV = 3`` and is not a layout-4 compatibility path.
