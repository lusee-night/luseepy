#!python3

from astropy.io import fits
import warnings
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

import uncrater as unc
from uncrater.utils import NPRODUCTS, NCHANNELS
from metadata_utils import metadata_to_dict
from const_storage import Constants


class FITSWriter:
    """Class to handle FITS writing with separate methods for each data type.

    Mapping from HDF5 to FITS:
    - HDF5 file → FITS HDUList
    - HDF5 items (item_000, item_001, ...) → Separate extensions with ITEM_ID in header
    - HDF5 subgroups (waveform, spectra, etc.) → Binary table extensions with EXTNAME
    - HDF5 attributes → Header keywords (HIERARCH for hierarchy)
    - HDF5 datasets → Binary table columns or Image extensions

    Each metadata item becomes a set of related extensions sharing the same ITEM_ID.
    """

    def __init__(
        self,
        output_file: str,
        cdi_dir: str,
        consts: Optional[Constants] = None,
        dcb_telemetry: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.output_file = output_file
        self.cdi_dir = cdi_dir
        self.coll = None
        self.hdu_list = None
        self.consts = consts
        self.dcb_telemetry = dcb_telemetry

    def write(self):
        """Main method to write FITS file."""
        # Load uncrater Collection
        self.coll = unc.Collection(self.cdi_dir)

        self._assign_zoom_timestamps()

        # Create primary HDU with file-level metadata
        primary = fits.PrimaryHDU()
        primary.header['CDI_DIR'] = self.cdi_dir
        primary.header['DATE'] = (datetime.utcnow().isoformat(), 'File creation date (UTC)')
        primary.header['ORIGIN'] = ('FITSWriter', 'File origin')

        self.hdu_list = [primary]

        self._write_session_invariants()
        self._write_constants()
        self._write_dcb_telemetry()

        # Track metadata changes
        metadata_groups = self._group_by_metadata()

        # Write concatenated metadata for the whole file
        self._write_metadata_global()

        # Write each metadata group to FITS
        for group_idx, group in enumerate(metadata_groups):
            self._write_waveforms(group_idx, group['waveform'])
            self._write_housekeeping(group_idx, group['housekeeping'])
            self._write_zoom_spectra(group_idx, group['zoom_spectra'])
            self._write_calibrator_data(group_idx, group['calibrator_data'])

        # Write concatenated spectra/TR spectra across the whole file
        self._write_spectra_global()
        self._write_tr_spectra_global(metadata_groups)

        # Add summary info to primary header
        primary.header['N_ITEMS'] = (len(metadata_groups), 'Number of metadata items')
        primary.header['N_GROUPS'] = (len(metadata_groups), 'Number of metadata groups (legacy)')

        # Write the FITS file
        hdul = fits.HDUList(self.hdu_list)
        hdul.writeto(self.output_file, overwrite=True)

    def _group_by_metadata(self) -> List[Dict]:
        """Group data by metadata configuration."""
        metadata_groups = []
        current_metadata_dict = None
        groups_by_meta_uid = {}

        # Group data by metadata configuration
        for sp_dict in self.coll.spectra:
            meta_pkt = sp_dict['meta']
            meta_dict = metadata_to_dict(meta_pkt)

            if current_metadata_dict is None or not self._metadata_equal(current_metadata_dict, meta_dict):
                # New metadata configuration
                current_metadata_dict = meta_dict
                group_entry = {
                    'metadata': meta_pkt,
                    'meta_unique_id': getattr(meta_pkt, 'unique_packet_id', None),
                    'metadata_dict': meta_dict,
                    'spectra': [],
                    'tr_spectra': [],
                    'housekeeping': [],
                    'waveform': [],
                    'zoom_spectra': [],
                    'calibrator_data': [],
                    'calibrator_debug': []
                }
                metadata_groups.append(group_entry)
                if group_entry['meta_unique_id'] is not None:
                    groups_by_meta_uid[group_entry['meta_unique_id']] = group_entry

            # Add this spectrum set to current group
            metadata_groups[-1]['spectra'].append(sp_dict)

        # Match TR spectra to metadata groups
        for trs_dict in self.coll.tr_spectra:
            meta_pkt = trs_dict['meta']
            meta_dict = metadata_to_dict(meta_pkt)

            # Find matching metadata group
            meta_uid = getattr(meta_pkt, 'unique_packet_id', None)
            group = groups_by_meta_uid.get(meta_uid) if meta_uid is not None else None

            if group is None:
                for candidate in metadata_groups:
                    if self._metadata_equal(candidate['metadata_dict'], meta_dict):
                        group = candidate
                        break

            if group is not None:
                group['tr_spectra'].append(trs_dict)
            else:
                print(f"Warning: TR spectrum with unmatched metadata")

        # Assign other packets to groups
        if metadata_groups:
            last_group = metadata_groups[-1]
            last_group['housekeeping'].extend(self.coll.housekeeping_packets)
            last_group['waveform'].extend(self.coll.waveform_packets)
            last_group['zoom_spectra'].extend(self.coll.zoom_spectra_packets)
            if hasattr(self.coll, 'calib_data') and len(self.coll.calib_data) > 0:
                last_group['calibrator_data'].extend(self.coll.calib_data)
            if hasattr(self.coll, 'calib_debug') and len(self.coll.calib_debug) > 0:
                last_group['calibrator_debug'].extend(self.coll.calib_debug)

        return metadata_groups

    def _metadata_equal(self, a: Dict, b: Dict) -> bool:
        if a.keys() != b.keys():
            return False
        for key in a.keys():
            if not self._values_equal(a[key], b[key]):
                return False
        return True

    def _values_equal(self, av, bv) -> bool:
        if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
            try:
                return np.array_equal(np.asarray(av), np.asarray(bv))
            except Exception:
                return False

        if isinstance(av, (list, tuple)) or isinstance(bv, (list, tuple)):
            if not isinstance(av, (list, tuple)) or not isinstance(bv, (list, tuple)):
                return False
            if len(av) != len(bv):
                return False
            for a_item, b_item in zip(av, bv):
                if not self._values_equal(a_item, b_item):
                    return False
            return True

        try:
            result = av == bv
        except Exception:
            try:
                return np.array_equal(np.asarray(av), np.asarray(bv))
            except Exception:
                return False

        if isinstance(result, np.ndarray):
            return np.array_equal(np.asarray(av), np.asarray(bv))

        return bool(result)

    def _write_session_invariants(self):
        hello = self._extract_hello_packet()
        if hello is None:
            print("Warning: no hello packet found; SESSION_INV extension omitted")
            return

        cols = [fits.Column(name='PLACEHOLDER', format='J', array=[0])]
        hdu = fits.BinTableHDU.from_columns(cols, name='SESSION_INV')
        hdu.header['SW_VERS'] = int(hello.SW_version)
        hdu.header['FW_VERS'] = int(hello.FW_Version)
        hdu.header['FW_ID'] = int(hello.FW_ID)
        hdu.header['FW_DATE'] = int(hello.FW_Date)
        hdu.header['FW_TIME'] = int(hello.FW_Time)
        hdu.header['ST_UID'] = int(hello.unique_packet_id)
        hdu.header['ST_T32'] = int(hello.time_32)
        hdu.header['ST_T16'] = int(hello.time_16)
        hdu.header['EXTDESC'] = ('Session invariants', 'Extension description')
        self.hdu_list.append(hdu)

    def _extract_hello_packet(self):
        for pkt in self.coll.cont:
            if isinstance(pkt, unc.Packet_Hello):
                pkt._read()
                return pkt
        return None

    def _write_constants(self):
        if not self.consts:
            return
        cols = [fits.Column(name='PLACEHOLDER', format='J', array=[0])]
        hdu = fits.BinTableHDU.from_columns(cols, name='CONSTANTS')
        hdu.header['LUN_LAT'] = float(self.consts.lun_lat_deg)
        hdu.header['LUN_LON'] = float(self.consts.lun_long_deg)
        hdu.header['LUN_HGT'] = float(self.consts.lun_height_m)
        hdu.header['EXTDESC'] = ('Landing coordinates', 'Extension description')
        self.hdu_list.append(hdu)

    def _write_dcb_telemetry(self):
        if not self.dcb_telemetry:
            return
        for name, values in self.dcb_telemetry.items():
            data = np.asarray(values)
            hdu = fits.ImageHDU(data=data, name=f'DCB_{self._sanitize_key(name)[:16]}')
            hdu.header['EXTDESC'] = ('DCB telemetry field', 'Extension description')
            hdu.header['FIELD'] = (name, 'Field name')
            hdu.header['NROWS'] = (len(data), 'Number of samples')
            self.hdu_list.append(hdu)

    def _write_metadata_global(self):
        """Write concatenated metadata as a single binary table."""
        if not self.coll.spectra:
            return

        meta_rows = []
        for sp_dict in self.coll.spectra:
            meta_rows.append(metadata_to_dict(sp_dict['meta']))

        flat_rows = [self._flatten_metadata_dict(row) for row in meta_rows]
        if not flat_rows:
            return

        keys = sorted(flat_rows[0].keys())
        cols = []
        used_names = set()
        for key in keys:
            data = np.array([row.get(key, np.nan) for row in flat_rows])
            base_name = self._sanitize_key(key)[:16]
            col_name = base_name
            if col_name in used_names:
                col_name = self._dedupe_name(base_name, used_names)
            used_names.add(col_name)
            fmt = self._fits_format_for_array(data)
            cols.append(fits.Column(name=col_name, format=fmt, array=data))

        hdu = fits.BinTableHDU.from_columns(cols, name='META')
        hdu.header['EXTDESC'] = ('Concatenated metadata', 'Extension description')
        hdu.header['NROWS'] = (len(flat_rows), 'Number of metadata packets')
        self.hdu_list.append(hdu)

    def _flatten_metadata_dict(self, meta: Dict) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for key, value in meta.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat.update(self._flatten_metadata_dict({f"{key}.{sub_key}": sub_val}))
            elif isinstance(value, (list, tuple, np.ndarray)):
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    flat[key] = value.item()
                else:
                    for idx, item in enumerate(value):
                        flat.update(self._flatten_metadata_dict({f"{key}.{idx:03d}": item}))
            else:
                flat[key] = value
        return flat

    def _fits_format_for_array(self, arr: np.ndarray) -> str:
        if np.issubdtype(arr.dtype, np.floating):
            return 'D'
        if np.issubdtype(arr.dtype, np.integer):
            return 'K'
        if np.issubdtype(arr.dtype, np.bool_):
            return 'L'
        return 'D'

    def _dedupe_name(self, base_name: str, used_names: set) -> str:
        idx = 1
        while True:
            suffix = str(idx)
            candidate = base_name[: max(0, 16 - len(suffix) - 1)] + "_" + suffix
            if candidate not in used_names:
                return candidate
            idx += 1

    def _write_metadata_value(self, header, path_parts, value):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._write_metadata_value(header, path_parts + [sub_key], sub_value)
            return

        if isinstance(value, np.ndarray) and value.ndim == 0:
            self._write_metadata_value(header, path_parts, value.item())
            return

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                warnings.warn(f"Empty metadata list at {'.'.join(map(str, path_parts))}")
                return
            for idx, item in enumerate(value):
                idx_key = f"{idx:03d}"
                self._write_metadata_value(header, path_parts + [idx_key], item)
            return

        fits_key = self._sanitize_path(path_parts)
        try:
            if isinstance(value, (bool, int, float, str, np.number)):
                header[f'HIERARCH META.{fits_key}'] = value
            else:
                header[f'HIERARCH META.{fits_key}'] = str(value)
        except (ValueError, TypeError) as e:
            warnings.warn(f"Could not write metadata key {'.'.join(map(str, path_parts))}: {e}")

    def _sanitize_key(self, key: str) -> str:
        """Sanitize a key for use in FITS headers."""
        # Replace problematic characters
        return key.upper().replace(' ', '_').replace('-', '_')[:68]

    def _sanitize_path(self, path_parts) -> str:
        cleaned = [self._sanitize_key(str(part)) for part in path_parts]
        return '.'.join(cleaned)[:68]

    def _write_waveforms(self, group_idx: int, waveform_packets):
        """Write waveform packets to FITS as binary table extensions."""
        if not waveform_packets:
            return

        # Group waveforms by channel
        waveforms_by_channel = {}
        for wf_pkt in waveform_packets:
            wf_pkt._read()
            if wf_pkt.ch not in waveforms_by_channel:
                waveforms_by_channel[wf_pkt.ch] = []
            waveforms_by_channel[wf_pkt.ch].append(wf_pkt)

        # Write each channel as a separate extension
        for ch, wf_list in waveforms_by_channel.items():
            n_waveforms = len(wf_list)
            waveform_array = np.zeros((n_waveforms, 16384), dtype=np.int16)
            timestamps = np.zeros(n_waveforms, dtype=np.float64)

            for i, wf_pkt in enumerate(wf_list):
                waveform_array[i] = wf_pkt.waveform
                timestamps[i] = wf_pkt.timestamp

            # Create binary table with waveform data
            # Using TDIM to specify 2D shape for waveforms column
            cols = [
                fits.Column(name='WAVEFORM', format='16384I', array=waveform_array),
                fits.Column(name='TIMESTAMP', format='D', array=timestamps),
            ]
            hdu = fits.BinTableHDU.from_columns(cols, name=f'WF_{group_idx:03d}_CH{ch}')

            hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
            hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
            hdu.header['CHANNEL'] = (ch, 'Waveform channel')
            hdu.header['COUNT'] = (n_waveforms, 'Number of waveforms')
            hdu.header['EXTDESC'] = ('Waveform data', 'Extension description')

            self.hdu_list.append(hdu)

        # Write summary extension for waveforms
        summary_cols = [
            fits.Column(name='CHANNEL', format='J', array=list(waveforms_by_channel.keys())),
            fits.Column(name='COUNT', format='J', array=[len(wf_list) for wf_list in waveforms_by_channel.values()]),
        ]
        summary_hdu = fits.BinTableHDU.from_columns(summary_cols, name=f'WF_{group_idx:03d}_SUM')
        summary_hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
        summary_hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
        summary_hdu.header['TOTCOUNT'] = (len(waveform_packets), 'Total waveform count')
        summary_hdu.header['EXTDESC'] = ('Waveform summary', 'Extension description')

        self.hdu_list.append(summary_hdu)

    def _write_spectra_global(self):
        """Write spectra concatenated across all items, one image per product."""
        spectra_dicts = self.coll.spectra
        if not spectra_dicts:
            return

        n_time = len(spectra_dicts)
        spectra_uids = np.zeros(n_time, dtype=np.int64)
        spectra_times = np.zeros(n_time, dtype=np.float64)

        per_product = [
            np.full((n_time, NCHANNELS), np.nan, dtype=np.float32)
            for _ in range(NPRODUCTS)
        ]

        for t_idx, sp_dict in enumerate(spectra_dicts):
            meta = sp_dict['meta']
            spectra_uids[t_idx] = meta.unique_packet_id
            spectra_times[t_idx] = meta.time if hasattr(meta, 'time') else np.nan

            for prod_idx in range(NPRODUCTS):
                if prod_idx in sp_dict:
                    spectrum = sp_dict[prod_idx].data
                    per_product[prod_idx][t_idx, :len(spectrum)] = spectrum

        for prod_idx, prod_data in enumerate(per_product):
            img_hdu = fits.ImageHDU(data=prod_data, name=f'SPEC_P{prod_idx:02d}')
            img_hdu.header['PROD_IDX'] = (prod_idx, 'Correlation product index')
            img_hdu.header['EXTDESC'] = ('Spectra product vs time/frequency', 'Extension description')
            img_hdu.header['NTIME'] = (n_time, 'Number of time samples')
            img_hdu.header['NCHANS'] = (NCHANNELS, 'Number of frequency channels')
            img_hdu.header['AXIS1'] = ('CHANNEL', 'Frequency channel')
            img_hdu.header['AXIS2'] = ('TIME', 'Time sample')
            self.hdu_list.append(img_hdu)

        id_cols = [
            fits.Column(name='UNIQUE_ID', format='K', array=spectra_uids),
            fits.Column(name='TIMESTAMP', format='D', array=spectra_times),
        ]
        id_hdu = fits.BinTableHDU.from_columns(id_cols, name='SPEC_ID')
        id_hdu.header['EXTDESC'] = ('Spectra unique IDs and timestamps', 'Extension description')
        id_hdu.header['COUNT'] = (n_time, 'Number of spectra')
        self.hdu_list.append(id_hdu)

    def _write_tr_spectra_global(self, metadata_groups: List[Dict]):
        """Write time-resolved spectra concatenated across all items, one image per product."""
        tr_spectra_dicts = self.coll.tr_spectra
        if not tr_spectra_dicts:
            return

        first_trs = tr_spectra_dicts[0]
        meta0 = first_trs['meta']
        tr_length0 = meta0.base.tr_stop - meta0.base.tr_start
        if meta0.base.tr_avg_shift > 0:
            tr_length0 = tr_length0 // (1 << meta0.base.tr_avg_shift)
        Navg20 = 1 << meta0.base.Navg2_shift

        for trs_dict in tr_spectra_dicts[1:]:
            meta = trs_dict['meta']
            tr_length = meta.base.tr_stop - meta.base.tr_start
            if meta.base.tr_avg_shift > 0:
                tr_length = tr_length // (1 << meta.base.tr_avg_shift)
            Navg2 = 1 << meta.base.Navg2_shift
            if tr_length != tr_length0 or Navg2 != Navg20:
                warnings.warn("TR spectra dimensions vary across items; falling back to per-item TR output")
                for group_idx, group in enumerate(metadata_groups):
                    self._write_tr_spectra_per_item(group_idx, group['tr_spectra'])
                return

        n_time = len(tr_spectra_dicts)
        tr_uids = np.zeros(n_time, dtype=np.int64)
        tr_times = np.zeros(n_time, dtype=np.float64)

        per_product = [
            np.full((n_time, Navg20, tr_length0), np.nan, dtype=np.float32)
            for _ in range(NPRODUCTS)
        ]

        for t_idx, trs_dict in enumerate(tr_spectra_dicts):
            meta = trs_dict['meta']
            tr_uids[t_idx] = meta.unique_packet_id
            tr_times[t_idx] = meta.time if hasattr(meta, 'time') else np.nan

            for prod_idx in range(NPRODUCTS):
                if prod_idx in trs_dict:
                    tr_spectrum = trs_dict[prod_idx].data
                    if tr_spectrum.ndim == 2:
                        per_product[prod_idx][t_idx, :tr_spectrum.shape[0], :tr_spectrum.shape[1]] = tr_spectrum
                    else:
                        flat_len = min(len(tr_spectrum), Navg20 * tr_length0)
                        per_product[prod_idx][t_idx].flat[:flat_len] = tr_spectrum.flat[:flat_len]

        for prod_idx, prod_data in enumerate(per_product):
            img_hdu = fits.ImageHDU(data=prod_data, name=f'TR_P{prod_idx:02d}')
            img_hdu.header['PROD_IDX'] = (prod_idx, 'Correlation product index')
            img_hdu.header['EXTDESC'] = ('TR spectra product vs time/bin', 'Extension description')
            img_hdu.header['NTIME'] = (n_time, 'Number of time samples')
            img_hdu.header['NAVG2'] = (Navg20, 'Navg2 dimension')
            img_hdu.header['TR_LEN'] = (tr_length0, 'TR length dimension')
            img_hdu.header['AXIS1'] = ('TR_BIN', 'Time-resolved bin')
            img_hdu.header['AXIS2'] = ('AVG2', 'Navg2 index')
            img_hdu.header['AXIS3'] = ('TIME', 'Time sample')
            self.hdu_list.append(img_hdu)

        id_cols = [
            fits.Column(name='UNIQUE_ID', format='K', array=tr_uids),
            fits.Column(name='TIMESTAMP', format='D', array=tr_times),
        ]
        id_hdu = fits.BinTableHDU.from_columns(id_cols, name='TR_ID')
        id_hdu.header['EXTDESC'] = ('TR spectra unique IDs and timestamps', 'Extension description')
        id_hdu.header['COUNT'] = (n_time, 'Number of TR spectra')
        self.hdu_list.append(id_hdu)

    def _write_tr_spectra_per_item(self, group_idx: int, tr_spectra_dicts):
        """Write time-resolved spectra to FITS per item (fallback)."""
        if not tr_spectra_dicts:
            return

        n_time = len(tr_spectra_dicts)
        first_trs = tr_spectra_dicts[0]
        meta = first_trs['meta']

        tr_length = meta.base.tr_stop - meta.base.tr_start
        if meta.base.tr_avg_shift > 0:
            tr_length = tr_length // (1 << meta.base.tr_avg_shift)

        Navg2 = 1 << meta.base.Navg2_shift

        tr_array = np.full((n_time, NPRODUCTS, Navg2, tr_length), np.nan, dtype=np.float32)
        tr_uids = np.zeros(n_time, dtype=np.int64)
        tr_times = np.zeros(n_time, dtype=np.float64)

        for t_idx, trs_dict in enumerate(tr_spectra_dicts):
            meta = trs_dict['meta']
            tr_uids[t_idx] = meta.unique_packet_id
            tr_times[t_idx] = meta.time if hasattr(meta, 'time') else np.nan

            for prod_idx in range(NPRODUCTS):
                if prod_idx in trs_dict:
                    tr_spectrum = trs_dict[prod_idx].data
                    if tr_spectrum.ndim == 2:
                        tr_array[t_idx, prod_idx, :tr_spectrum.shape[0], :tr_spectrum.shape[1]] = tr_spectrum
                    else:
                        flat_len = min(len(tr_spectrum), Navg2 * tr_length)
                        tr_array[t_idx, prod_idx].flat[:flat_len] = tr_spectrum.flat[:flat_len]

        img_hdu = fits.ImageHDU(data=tr_array, name=f'TRSPEC_{group_idx:03d}')
        img_hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
        img_hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
        img_hdu.header['EXTDESC'] = ('Time-resolved spectra', 'Extension description')
        img_hdu.header['NTIME'] = (n_time, 'Number of time samples')
        img_hdu.header['NPRODS'] = (NPRODUCTS, 'Number of correlation products')
        img_hdu.header['NAVG2'] = (Navg2, 'Navg2 dimension')
        img_hdu.header['TR_LEN'] = (tr_length, 'TR length dimension')
        img_hdu.header['AXIS1'] = ('TR_BIN', 'Time-resolved bin')
        img_hdu.header['AXIS2'] = ('AVG2', 'Navg2 index')
        img_hdu.header['AXIS3'] = ('PRODUCT', 'Correlation product')
        img_hdu.header['AXIS4'] = ('TIME', 'Time sample')

        self.hdu_list.append(img_hdu)

        id_cols = [
            fits.Column(name='UNIQUE_ID', format='K', array=tr_uids),
            fits.Column(name='TIMESTAMP', format='D', array=tr_times),
        ]
        id_hdu = fits.BinTableHDU.from_columns(id_cols, name=f'TRSPEC_{group_idx:03d}_ID')
        id_hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
        id_hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
        id_hdu.header['EXTDESC'] = ('TR spectra unique IDs and timestamps', 'Extension description')
        id_hdu.header['COUNT'] = (n_time, 'Number of TR spectra')
        self.hdu_list.append(id_hdu)

    def _write_housekeeping(self, group_idx: int, housekeeping_packets):
        """Write housekeeping packets to FITS as binary tables grouped by type."""
        if not housekeeping_packets:
            return

        # Group packets by type for more efficient storage
        packets_by_type = {}
        for hk_pkt in housekeeping_packets:
            hk_pkt._read()
            hk_type = hk_pkt.hk_type
            if hk_type not in packets_by_type:
                packets_by_type[hk_type] = []
            packets_by_type[hk_type].append(hk_pkt)

        # Write each type as a separate extension
        for hk_type, packets in packets_by_type.items():
            cols = self._build_housekeeping_columns(hk_type, packets)
            if cols:
                hdu = fits.BinTableHDU.from_columns(cols, name=f'HK_{group_idx:03d}_T{hk_type}')
                hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
                hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
                hdu.header['HK_TYPE'] = (hk_type, 'Housekeeping packet type')
                hdu.header['COUNT'] = (len(packets), 'Number of packets')
                hdu.header['EXTDESC'] = (f'Housekeeping type {hk_type}', 'Extension description')

                self.hdu_list.append(hdu)

        # Write summary extension
        summary_cols = [
            fits.Column(name='HK_TYPE', format='J', array=list(packets_by_type.keys())),
            fits.Column(name='COUNT', format='J', array=[len(p) for p in packets_by_type.values()]),
        ]
        summary_hdu = fits.BinTableHDU.from_columns(summary_cols, name=f'HK_{group_idx:03d}_SUM')
        summary_hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
        summary_hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
        summary_hdu.header['TOTCOUNT'] = (len(housekeeping_packets), 'Total HK packet count')
        summary_hdu.header['EXTDESC'] = ('Housekeeping summary', 'Extension description')

        self.hdu_list.append(summary_hdu)

    def _build_housekeeping_columns(self, hk_type: int, packets) -> List[fits.Column]:
        """Build column definitions for housekeeping packets by type."""
        n_packets = len(packets)

        # Common columns for all types
        versions = np.array([p.version for p in packets], dtype=np.int32)
        uids = np.array([p.unique_packet_id for p in packets], dtype=np.int64)
        errors = np.array([p.errors for p in packets], dtype=np.int32)

        cols = [
            fits.Column(name='VERSION', format='J', array=versions),
            fits.Column(name='UNIQUE_ID', format='K', array=uids),
            fits.Column(name='ERRORS', format='J', array=errors),
        ]

        if hk_type == 0:
            times = np.array([getattr(p, 'time', 0) for p in packets], dtype=np.float64)
            cols.append(fits.Column(name='TIME', format='D', array=times))

        elif hk_type == 1:
            # ADC statistics - these are likely arrays
            adc_mins = np.array([getattr(p, 'min', []) for p in packets])
            adc_maxs = np.array([getattr(p, 'max', []) for p in packets])
            adc_means = np.array([getattr(p, 'mean', []) for p in packets])
            adc_rms = np.array([getattr(p, 'rms', []) for p in packets])

            if len(adc_mins) > 0 and hasattr(adc_mins[0], '__len__'):
                n_adc = len(adc_mins[0]) if len(adc_mins) > 0 else 4
                cols.extend([
                    fits.Column(name='ADC_MIN', format=f'{n_adc}E', array=adc_mins),
                    fits.Column(name='ADC_MAX', format=f'{n_adc}E', array=adc_maxs),
                    fits.Column(name='ADC_MEAN', format=f'{n_adc}E', array=adc_means),
                    fits.Column(name='ADC_RMS', format=f'{n_adc}E', array=adc_rms),
                ])

            # Actual gain as string
            gains = np.array([''.join(getattr(p, 'actual_gain', '')) for p in packets])
            max_gain_len = max(len(g) for g in gains) if len(gains) > 0 else 10
            cols.append(fits.Column(name='ACTUAL_GAIN', format=f'{max_gain_len}A', array=gains))

        elif hk_type == 2:
            times = np.array([getattr(p, 'time', 0) for p in packets], dtype=np.float64)
            oks = np.array([getattr(p, 'ok', False) for p in packets], dtype=bool)
            cols.extend([
                fits.Column(name='TIME', format='D', array=times),
                fits.Column(name='OK', format='L', array=oks),
            ])

            # Handle telemetry dict - flatten to individual columns
            # First, collect all telemetry keys
            all_telem_keys = set()
            for p in packets:
                if hasattr(p, 'telemetry') and p.telemetry:
                    all_telem_keys.update(p.telemetry.keys())

            for telem_key in sorted(all_telem_keys):
                telem_values = []
                for p in packets:
                    if hasattr(p, 'telemetry') and p.telemetry and telem_key in p.telemetry:
                        telem_values.append(p.telemetry[telem_key])
                    else:
                        telem_values.append(np.nan)
                col_name = f'TELEM_{self._sanitize_key(telem_key)[:8]}'
                cols.append(fits.Column(name=col_name, format='E', array=np.array(telem_values, dtype=np.float32)))

        elif hk_type == 3:
            checksums = np.array([getattr(p, 'checksum', 0) for p in packets], dtype=np.int64)
            weight_ndxs = np.array([getattr(p, 'weight_ndx', 0) for p in packets], dtype=np.int32)
            cols.extend([
                fits.Column(name='CHECKSUM', format='K', array=checksums),
                fits.Column(name='WEIGHT_NDX', format='J', array=weight_ndxs),
            ])

        return cols

    def _write_zoom_spectra(self, group_idx: int, zoom_packets):
        """Write zoom spectra as a binary table extension."""
        if not zoom_packets:
            return

        n_packets = len(zoom_packets)
        fft_size = 64

        # Initialize arrays
        ch1_autocorr_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        ch2_autocorr_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        ch1_2_corr_real_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        ch1_2_corr_imag_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        unique_ids = np.zeros(n_packets, dtype=np.int64)
        pfb_indices = np.zeros(n_packets, dtype=np.int32)
        timestamps = np.zeros(n_packets, dtype=np.float64)

        for i, zoom_pkt in enumerate(zoom_packets):
            zoom_pkt._read()

            ch1_autocorr_all[i] = zoom_pkt.ch1_autocorr
            ch2_autocorr_all[i] = zoom_pkt.ch2_autocorr
            ch1_2_corr_real_all[i] = zoom_pkt.ch1_2_corr_real
            ch1_2_corr_imag_all[i] = zoom_pkt.ch1_2_corr_imag

            uid = zoom_pkt.unique_packet_id[0] if isinstance(zoom_pkt.unique_packet_id, tuple) else zoom_pkt.unique_packet_id
            pfb_idx = zoom_pkt.pfb_index[0] if isinstance(zoom_pkt.pfb_index, tuple) else zoom_pkt.pfb_index

            unique_ids[i] = uid
            pfb_indices[i] = pfb_idx
            timestamps[i] = zoom_pkt.zoom_timestamp if hasattr(zoom_pkt, 'zoom_timestamp') else 0.0

        # Create binary table
        cols = [
            fits.Column(name='CH1_AUTOCORR', format=f'{fft_size}E', array=ch1_autocorr_all),
            fits.Column(name='CH2_AUTOCORR', format=f'{fft_size}E', array=ch2_autocorr_all),
            fits.Column(name='CORR_REAL', format=f'{fft_size}E', array=ch1_2_corr_real_all),
            fits.Column(name='CORR_IMAG', format=f'{fft_size}E', array=ch1_2_corr_imag_all),
            fits.Column(name='UNIQUE_ID', format='K', array=unique_ids),
            fits.Column(name='PFB_INDEX', format='J', array=pfb_indices),
            fits.Column(name='TIMESTAMP', format='D', array=timestamps),
        ]

        hdu = fits.BinTableHDU.from_columns(cols, name=f'ZOOM_{group_idx:03d}')
        hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
        hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
        hdu.header['EXTDESC'] = ('Calibrator zoom spectra', 'Extension description')
        hdu.header['COUNT'] = (n_packets, 'Number of zoom spectra')
        hdu.header['FFT_SIZE'] = (fft_size, 'FFT size')

        self.hdu_list.append(hdu)

    def _write_calibrator_data(self, group_idx: int, calibrator_data):
        """Write calibrator data to FITS."""
        if not calibrator_data:
            return

        # Flatten calibrator data into a table structure
        rows = []
        for pkt_idx, cal_data in enumerate(calibrator_data):
            for ch_idx, ch_data in enumerate(cal_data):
                if ch_data is not None:
                    rows.append({
                        'packet_idx': pkt_idx,
                        'channel_idx': ch_idx,
                        'data': ch_data
                    })

        if not rows:
            return

        # Determine max data length
        max_len = max(len(r['data']) for r in rows)

        # Pad all data arrays to same length
        packet_indices = np.array([r['packet_idx'] for r in rows], dtype=np.int32)
        channel_indices = np.array([r['channel_idx'] for r in rows], dtype=np.int32)

        # Create 2D array for data
        data_array = np.zeros((len(rows), max_len), dtype=np.float32)
        data_lengths = np.zeros(len(rows), dtype=np.int32)

        for i, r in enumerate(rows):
            data_len = len(r['data'])
            data_array[i, :data_len] = r['data']
            data_lengths[i] = data_len

        cols = [
            fits.Column(name='PACKET_IDX', format='J', array=packet_indices),
            fits.Column(name='CHANNEL_IDX', format='J', array=channel_indices),
            fits.Column(name='DATA_LEN', format='J', array=data_lengths),
            fits.Column(name='DATA', format=f'{max_len}E', array=data_array),
        ]

        hdu = fits.BinTableHDU.from_columns(cols, name=f'CALDAT_{group_idx:03d}')
        hdu.header['ITEM_ID'] = (group_idx, 'Metadata item index')
        hdu.header['GROUP_ID'] = (group_idx, 'Metadata group index (legacy)')
        hdu.header['EXTDESC'] = ('Calibrator data', 'Extension description')
        hdu.header['COUNT'] = (len(calibrator_data), 'Number of calibrator packets')
        hdu.header['NROWS'] = (len(rows), 'Number of data rows')
        hdu.header['MAX_LEN'] = (max_len, 'Maximum data array length')

        self.hdu_list.append(hdu)

    def _assign_zoom_timestamps(self):
        """Assign timestamps to zoom spectrum packets efficiently."""
        if not self.coll.zoom_spectra_packets:
            return

        # Start from the beginning of spectra list
        spectra_idx = 0

        for zoom_pkt in self.coll.zoom_spectra_packets:
            zoom_pkt._read()

            # Get the packet index from the zoom packet
            zoom_packet_index = zoom_pkt.packet_index

            # Find the metadata packet that comes before this zoom packet
            meta_time = None

            # Move forward through spectra until we pass the zoom packet index
            while spectra_idx < len(self.coll.spectra):
                spectra_dict = self.coll.spectra[spectra_idx]
                meta_pkt = spectra_dict['meta']

                # Check if metadata packet index is past zoom packet
                if meta_pkt.packet_index > zoom_packet_index:
                    # Use the previous metadata if available
                    if spectra_idx > 0:
                        prev_meta = self.coll.spectra[spectra_idx - 1]['meta']
                        meta_time = prev_meta.time
                    break

                # This metadata is still before or at zoom packet, keep it as candidate
                meta_time = meta_pkt.time
                spectra_idx += 1

            # Assign the timestamp to zoom packet
            zoom_pkt.zoom_timestamp = meta_time if meta_time is not None else 0.0


def save_to_fits(
    cdi_dir: str,
    output_file: str,
    consts: Optional[Constants] = None,
    dcb_telemetry: Optional[Dict[str, np.ndarray]] = None,
):
    writer = FITSWriter(output_file=output_file, cdi_dir=cdi_dir, consts=consts, dcb_telemetry=dcb_telemetry)
    writer.write()
