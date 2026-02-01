# HDF5 file layout (LuSEE receive)

Each HDF5 file corresponds to a single observation session. The file is produced by `hdf5_writer.py` from an `uncrater.Collection` and is organized as a set of **items** (formerly “groups”) that share a common metadata configuration.

## Top-level

- Attributes
  - `cdi_directory` : path to the CDI directory used to build the file
  - `n_items` : number of metadata items in this file
  - `n_groups` : legacy alias of `n_items`

- Groups
  - `item_000`, `item_001`, … : one item per distinct metadata configuration encountered in order of the spectra stream

## Item group: `/item_###`

Each item contains:

- `meta/` : metadata for this item (attributes and datasets)
- `spectra/` : normal spectra for this item (power spectra)
- `tr_spectra/` : time‑resolved spectra for this item
- `waveform/` : raw ADC waveform packets grouped by channel (if present)
- `housekeeping/` : housekeeping packets (if present)
- `calibrator/` : calibrator data and zoom spectra (if present)

### Metadata group: `/item_###/meta`

All fields returned by `metadata_utils.metadata_to_dict()` are written under this group. Scalars are stored as **attributes**, and arrays/lists are stored as **datasets** or subgroups.

#### Scalar attributes (examples)

These are stored as attributes on `/item_###/meta`:

- `_uC_time`, `_time_32`, `_time_16`
- `loop_count_min`, `loop_count_max`
- `Navgf`, `Navg1_shift`, `Navg2_shift`
- `notch`, `format`, `corr_products_mask`
- `tr_start`, `tr_stop`, `tr_avg_shift`
- `grimm_enable`, `averaging_mode`
- `reject_ratio`, `reject_maxbad`
- `bitslice_keep_bits`
- `weight`, `weight_current`
- `hi_frac`, `med_frac`
- `rand_state`
- `num_bad_min_current`, `num_bad_max_current`
- `num_bad_min`, `num_bad_max`
- `spec_overflow`, `notch_overflow`
- `errors`
- `calibrator_enable`, `spectrometer_enable`
- plus any extra scalar fields attached to the metadata packet (e.g. `time`)

#### Array datasets (examples)

Stored as datasets under `/item_###/meta`:

- `gain` (shape 4)
- `actual_gain` (shape 4)
- `gain_auto_min` (shape 4)
- `gain_auto_mult` (shape 4)
- `bitslice` (shape 16)
- `actual_bitslice` (shape 16)

#### Nested groups

Some metadata fields are structured groups:

- `/item_###/meta/raw_ADC_stat/` : ADC statistics per antenna
  - `item_000/`, `item_001/`, `item_002/`, `item_003/` each contain attributes:
    - `min`, `max`, `valid_count`, `invalid_count_min`, `invalid_count_max`, `sumv`, `sumv2`
- `/item_###/meta/route/` : routing per antenna
  - `item_000/`, `item_001/`, `item_002/`, `item_003/` each contain attributes:
    - `plus`, `minus`

#### Processed ADC / telemetry fields

If `Packet_Metadata` attaches processed fields, they are included as attributes on `/item_###/meta`:

- `adc_min`, `adc_max`, `adc_mean`, `adc_rms`, `adc_valid_count`, `adc_invalid_count_min`, `adc_invalid_count_max`, `adc_total_count`
- telemetry fields such as `telemetry_V1_0`, `telemetry_V1_8`, `telemetry_V2_5`, `telemetry_T_FPGA`

### Spectra: `/item_###/spectra`

- `data` : float32 array, shape `(n_time, 16, 2048)`
- `unique_ids` : uint32/uint64 array, length `n_time`
- `timestamps` : float64 array, length `n_time` (from `meta.time`)

Attributes on `/item_###`:
- `spectra_count` : number of spectra time samples

### Time‑resolved spectra: `/item_###/tr_spectra`

- `data` : float32 array, shape `(n_time, 16, Navg2, tr_length)`
- `unique_ids` : uint32/uint64 array, length `n_time`
- `timestamps` : float64 array, length `n_time` (from `meta.time`)

Attributes on `/item_###`:
- `tr_spectra_count`
- `tr_spectra_Navg2`
- `tr_spectra_tr_length`

### Waveforms: `/item_###/waveform`

- `channel_0`, `channel_1`, `channel_2`, `channel_3` (only if present)
  - `waveforms` : int16 array, shape `(n_waveforms, 16384)`
  - `timestamps` : float64 array, length `n_waveforms`
  - attributes: `count`, `channel`
- attributes on `/item_###/waveform`: `total_count`, `channels`

### Housekeeping: `/item_###/housekeeping`

- `packet_0`, `packet_1`, …
  - attributes: `hk_type`, `version`, `unique_packet_id`, `errors`, plus type‑specific fields
- attributes on `/item_###/housekeeping`: `count`

### Calibrator: `/item_###/calibrator`

- `zoom_spectra/`
  - `ch1_autocorr`, `ch2_autocorr`, `ch1_2_corr_real`, `ch1_2_corr_imag` : float32 arrays, shape `(n_packets, 64)`
  - `unique_ids` : array, length `n_packets`
  - `pfb_indices` : array, length `n_packets`
  - `timestamps` : float64 array, length `n_packets`
  - attributes: `count`

- `data/`
  - `packet_{i}_ch_{j}` datasets for calibrator packets (variable sizes)
  - attribute: `count`
