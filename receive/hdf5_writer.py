#!python3

import warnings
import numpy as np
from typing import List, Dict, Optional
import h5py

import uncrater as unc
from uncrater.utils import NPRODUCTS, NCHANNELS
from metadata_utils import metadata_to_dict
from const_storage import Constants

from icecream import ic


class HDF5Writer:
    """Class to handle HDF5 writing with separate methods for each data type."""

    def __init__(self, output_file: str, cdi_dir: str, consts: Optional[Constants]=None):
        self.output_file = output_file
        self.cdi_dir = cdi_dir
        self.coll = None
        self.consts = consts

    def write(self):
        """Main method to write HDF5 file."""
        # Load uncrater Collection
        self.coll = unc.Collection(self.cdi_dir)

        self._assign_zoom_timestamps()

        with h5py.File(self.output_file, 'w') as f:
            # Store session-level info
            f.attrs['cdi_directory'] = self.cdi_dir
            self._write_session_invariants(f)
            self._write_constants(f)

            # Track metadata changes
            metadata_groups = self._group_by_metadata()

            # Write each metadata group to HDF5
            for group_idx, group in enumerate(metadata_groups):
                group_name = f'item_{group_idx:03d}'
                h5_group = f.create_group(group_name)

                # Store metadata under a dedicated subgroup
                meta_group = h5_group.create_group('meta')
                self._write_metadata(meta_group, group['metadata_dict'])


                # Write different data types
                self._write_waveforms(h5_group, group['waveform'])
                self._write_spectra(h5_group, group['spectra'])
                self._write_tr_spectra(h5_group, group['tr_spectra'])
                self._write_housekeeping(h5_group, group['housekeeping'])
                self._write_zoom_spectra(h5_group, group['zoom_spectra'])
                self._write_calibrator_data(h5_group, group['calibrator_data'])

            # Add summary information
            f.attrs['n_items'] = len(metadata_groups)

    def _write_constants(self, h5_file):
        if not self.consts:
            return
        const_group = h5_file.create_group("constants")
        const_group.attrs["lun_lat_deg"] = self.consts.lun_lat_deg
        const_group.attrs["lun_long_deg"] = self.consts.lun_long_deg
        const_group.attrs["lun_height_m"] = self.consts.lun_height_m

    def _write_session_invariants(self, h5_file):
        invariants = h5_file.create_group('session_invariants')
        hello = self._extract_hello_packet()
        if hello is None:
            print("Warning: no hello packet found; session_invariants is empty")
            return

        invariants.attrs['software_version'] = int(hello.SW_version)
        ic(invariants.attrs['software_version'], int(hello.SW_version))
        invariants.attrs['firmware_version'] = int(hello.FW_Version)
        invariants.attrs['firmware_id'] = int(hello.FW_ID)
        invariants.attrs['firmware_date'] = int(hello.FW_Date)
        invariants.attrs['firmware_time'] = int(hello.FW_Time)
        invariants.attrs['start_unique_packet_id'] = int(hello.unique_packet_id)
        invariants.attrs['start_time_32'] = int(hello.time_32)
        invariants.attrs['start_time_16'] = int(hello.time_16)

    def _extract_hello_packet(self):
        for pkt in self.coll.cont:
            if isinstance(pkt, unc.Packet_Hello):
                pkt._read()
                return pkt
        return None

    def _is_scalar(self, value) -> bool:
        return isinstance(value, (int, float, bool, np.number, str, bytes))

    def _write_dataset(self, h5_group, name: str, value):
        if isinstance(value, np.ndarray):
            h5_group.create_dataset(name, data=value, compression='gzip')
            return

        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            str_dtype = h5py.string_dtype(encoding='utf-8')
            h5_group.create_dataset(name, data=np.array(value, dtype=str_dtype), compression='gzip')
            return

        h5_group.create_dataset(name, data=np.asarray(value), compression='gzip')

    def _write_metadata_value(self, h5_group, key, value, path: str):
        name = str(key)

        if isinstance(value, dict):
            sub_group = h5_group.create_group(name)
            for sub_key, sub_value in value.items():
                self._write_metadata_value(sub_group, sub_key, sub_value, f"{path}.{sub_key}")
            return

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                warnings.warn(f"Empty metadata list at {path}")
            if all(isinstance(item, dict) for item in value):
                sub_group = h5_group.create_group(name)
                for idx, item in enumerate(value):
                    item_group = sub_group.create_group(f'item_{idx:03d}')
                    for sub_key, sub_value in item.items():
                        self._write_metadata_value(item_group, sub_key, sub_value, f"{path}[{idx}].{sub_key}")
                sub_group.attrs['count'] = len(value)
                return

            if all(self._is_scalar(item) for item in value):
                self._write_dataset(h5_group, name, value)
                return

            sub_group = h5_group.create_group(name)
            for idx, item in enumerate(value):
                item_group = sub_group.create_group(f'item_{idx:03d}')
                self._write_metadata_value(item_group, 'value', item, f"{path}[{idx}]")
            sub_group.attrs['count'] = len(value)
            return

        if self._is_scalar(value):
            h5_group.attrs[name] = value
            return

        if isinstance(value, np.ndarray):
            self._write_dataset(h5_group, name, value)
            return

        warnings.warn(f"Unsupported metadata type at {path}: {type(value).__name__}, storing as string")
        h5_group.attrs[name] = str(value)

    def _write_metadata(self, h5_group, metadata_dict: Dict):
        for key, value in metadata_dict.items():
            self._write_metadata_value(h5_group, key, value, str(key))

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
            av = a[key]
            bv = b[key]
            if not self._values_equal(av, bv):
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

    def _write_waveforms(self, h5_group, waveform_packets):
        """Write waveform packets to HDF5."""
        if not waveform_packets:
            return

        wf_group = h5_group.create_group('waveform')

        # Group waveforms by channel
        waveforms_by_channel = {}
        for wf_pkt in waveform_packets:
            wf_pkt._read()
            if wf_pkt.ch not in waveforms_by_channel:
                waveforms_by_channel[wf_pkt.ch] = []
            waveforms_by_channel[wf_pkt.ch].append(wf_pkt)

        # Write each channel's waveforms
        for ch, wf_list in waveforms_by_channel.items():
            ch_group = wf_group.create_group(f'channel_{ch}')

            n_waveforms = len(wf_list)
            waveform_array = np.zeros((n_waveforms, 16384), dtype=np.int16)
            timestamps = []

            for i, wf_pkt in enumerate(wf_list):
                waveform_array[i] = wf_pkt.waveform
                timestamps.append(wf_pkt.timestamp)

            ch_group.create_dataset('waveforms', data=waveform_array, compression='gzip')
            ch_group.create_dataset('timestamps', data=timestamps)
            ch_group.attrs['count'] = n_waveforms
            ch_group.attrs['channel'] = ch

        wf_group.attrs['total_count'] = len(waveform_packets)
        wf_group.attrs['channels'] = list(waveforms_by_channel.keys())

    def _write_spectra(self, h5_group, spectra_dicts):
        """Write spectra to HDF5."""
        if not spectra_dicts:
            return

        n_time = len(spectra_dicts)

        spectra_array = np.full((n_time, NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
        spectra_uids = []
        spectra_times = []

        for t_idx, sp_dict in enumerate(spectra_dicts):
            meta = sp_dict['meta']
            spectra_uids.append(meta.unique_packet_id)
            spectra_times.append(meta.time if hasattr(meta, 'time') else np.nan)

            for prod_idx in range(NPRODUCTS):
                if prod_idx in sp_dict:
                    spectrum = sp_dict[prod_idx].data
                    spectra_array[t_idx, prod_idx, :len(spectrum)] = spectrum

        h5_group.create_dataset('spectra/data', data=spectra_array, compression='gzip')
        h5_group.create_dataset('spectra/unique_ids', data=spectra_uids)
        h5_group.create_dataset('spectra/timestamps', data=np.array(spectra_times, dtype=np.float64))
        h5_group.attrs['spectra_count'] = n_time

    def _write_tr_spectra(self, h5_group, tr_spectra_dicts):
        """Write time-resolved spectra to HDF5."""
        if not tr_spectra_dicts:
            return

        n_time = len(tr_spectra_dicts)

        # Get dimensions from first packet
        first_trs = tr_spectra_dicts[0]
        meta = first_trs['meta']

        tr_length = meta.base.tr_stop - meta.base.tr_start
        if meta.base.tr_avg_shift > 0:
            tr_length = tr_length // (1 << meta.base.tr_avg_shift)

        Navg2 = 1 << meta.base.Navg2_shift

        tr_array = np.full((n_time, NPRODUCTS, Navg2, tr_length), np.nan, dtype=np.float32)
        tr_uids = []
        tr_times = []

        for t_idx, trs_dict in enumerate(tr_spectra_dicts):
            meta = trs_dict['meta']
            tr_uids.append(meta.unique_packet_id)
            tr_times.append(meta.time if hasattr(meta, 'time') else np.nan)

            for prod_idx in range(NPRODUCTS):
                if prod_idx in trs_dict:
                    tr_spectrum = trs_dict[prod_idx].data
                    if tr_spectrum.ndim == 2:
                        tr_array[t_idx, prod_idx, :tr_spectrum.shape[0], :tr_spectrum.shape[1]] = tr_spectrum
                    else:
                        print(f"Warning: TR spectrum has unexpected shape {tr_spectrum.shape}")
                        flat_len = min(len(tr_spectrum), Navg2 * tr_length)
                        tr_array[t_idx, prod_idx].flat[:flat_len] = tr_spectrum.flat[:flat_len]

        h5_group.create_dataset('tr_spectra/data', data=tr_array, compression='gzip')
        h5_group.create_dataset('tr_spectra/unique_ids', data=tr_uids)
        h5_group.create_dataset('tr_spectra/timestamps', data=np.array(tr_times, dtype=np.float64))
        h5_group.attrs['tr_spectra_count'] = n_time
        h5_group.attrs['tr_spectra_Navg2'] = Navg2
        h5_group.attrs['tr_spectra_tr_length'] = tr_length

    def _write_housekeeping(self, h5_group, housekeeping_packets):
        """Write housekeeping packets to HDF5."""
        if not housekeeping_packets:
            return

        hk_group = h5_group.create_group('housekeeping')

        for i, hk_pkt in enumerate(housekeeping_packets):
            hk_pkt._read()

            pkt_group = hk_group.create_group(f'packet_{i}')
            pkt_group.attrs['hk_type'] = hk_pkt.hk_type
            pkt_group.attrs['version'] = hk_pkt.version
            pkt_group.attrs['unique_packet_id'] = hk_pkt.unique_packet_id
            pkt_group.attrs['errors'] = hk_pkt.errors

            if hk_pkt.hk_type == 0:
                if hasattr(hk_pkt, 'time'):
                    pkt_group.attrs['time'] = hk_pkt.time
            elif hk_pkt.hk_type == 1:
                if hasattr(hk_pkt, 'min'):
                    pkt_group.attrs['adc_min'] = hk_pkt.min
                if hasattr(hk_pkt, 'max'):
                    pkt_group.attrs['adc_max'] = hk_pkt.max
                if hasattr(hk_pkt, 'mean'):
                    pkt_group.attrs['adc_mean'] = hk_pkt.mean
                if hasattr(hk_pkt, 'rms'):
                    pkt_group.attrs['adc_rms'] = hk_pkt.rms
                if hasattr(hk_pkt, 'actual_gain'):
                    pkt_group.attrs['actual_gain'] = ''.join(hk_pkt.actual_gain)
            elif hk_pkt.hk_type == 2:
                if hasattr(hk_pkt, 'time'):
                    pkt_group.attrs['time'] = hk_pkt.time
                if hasattr(hk_pkt, 'ok'):
                    pkt_group.attrs['ok'] = hk_pkt.ok
                if hasattr(hk_pkt, 'telemetry'):
                    for k, v in hk_pkt.telemetry.items():
                        pkt_group.attrs[f'telemetry_{k}'] = v
            elif hk_pkt.hk_type == 3:
                if hasattr(hk_pkt, 'checksum'):
                    pkt_group.attrs['checksum'] = hk_pkt.checksum
                if hasattr(hk_pkt, 'weight_ndx'):
                    pkt_group.attrs['weight_ndx'] = hk_pkt.weight_ndx

        hk_group.attrs['count'] = len(housekeeping_packets)

    def _write_zoom_spectra(self, h5_group, zoom_packets):
        """Write zoom spectra as concatenated arrays."""
        if not zoom_packets:
            return

        zoom_group = h5_group.create_group('calibrator/zoom_spectra')

        n_packets = len(zoom_packets)
        fft_size = 64

        # Initialize arrays
        ch1_autocorr_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        ch2_autocorr_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        ch1_2_corr_real_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        ch1_2_corr_imag_all = np.zeros((n_packets, fft_size), dtype=np.float32)
        unique_ids = []
        pfb_indices = []
        timestamps = []

        for i, zoom_pkt in enumerate(zoom_packets):
            zoom_pkt._read()

            ch1_autocorr_all[i] = zoom_pkt.ch1_autocorr
            ch2_autocorr_all[i] = zoom_pkt.ch2_autocorr
            ch1_2_corr_real_all[i] = zoom_pkt.ch1_2_corr_real
            ch1_2_corr_imag_all[i] = zoom_pkt.ch1_2_corr_imag

            uid = zoom_pkt.unique_packet_id[0] if isinstance(zoom_pkt.unique_packet_id, tuple) else zoom_pkt.unique_packet_id
            pfb_idx = zoom_pkt.pfb_index[0] if isinstance(zoom_pkt.pfb_index, tuple) else zoom_pkt.pfb_index

            unique_ids.append(uid)
            pfb_indices.append(pfb_idx)

            # Use the assigned timestamp
            timestamps.append(zoom_pkt.zoom_timestamp if hasattr(zoom_pkt, 'zoom_timestamp') else 0.0)

        # Store as datasets
        zoom_group.create_dataset('ch1_autocorr', data=ch1_autocorr_all, compression='gzip')
        zoom_group.create_dataset('ch2_autocorr', data=ch2_autocorr_all, compression='gzip')
        zoom_group.create_dataset('ch1_2_corr_real', data=ch1_2_corr_real_all, compression='gzip')
        zoom_group.create_dataset('ch1_2_corr_imag', data=ch1_2_corr_imag_all, compression='gzip')
        zoom_group.create_dataset('unique_ids', data=unique_ids)
        zoom_group.create_dataset('pfb_indices', data=pfb_indices)
        zoom_group.create_dataset('timestamps', data=timestamps)
        zoom_group.attrs['count'] = n_packets

    def _write_calibrator_data(self, h5_group, calibrator_data):
        """Write calibrator data to HDF5."""
        if not calibrator_data:
            return

        cal_data_group = h5_group.create_group('calibrator/data')

        for i, cal_data in enumerate(calibrator_data):
            for ch_idx, ch_data in enumerate(cal_data):
                if ch_data is not None:
                    cal_data_group.create_dataset(f'packet_{i}_ch_{ch_idx}',
                                                 data=ch_data, compression='gzip')
        cal_data_group.attrs['count'] = len(calibrator_data)


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
            # by looking through spectra dictionaries
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


def save_to_hdf5(cdi_dir: str, output_file: str, consts: Optional[Constants]=None):
    """
    Save a session of packets to HDF5 file.

    Args:
        cdi_dir: Directory containing CDI output files
        output_file: Path to output HDF5 file
    """
    writer = HDF5Writer(output_file, cdi_dir, consts)
    writer.write()
