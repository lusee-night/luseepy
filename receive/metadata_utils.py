#!python3

import numpy as np
from uncrater.utils import NPRODUCTS
from icecream import ic


def ADC_stat_to_arrays(adc_stats):
    return {
        'min': np.array([int(x.min) for x in adc_stats], dtype=np.int16),
        'max': np.array([int(x.max) for x in adc_stats], dtype=np.int16),
        'valid_count': np.array([int(x.valid_count) for x in adc_stats], dtype=np.uint32),
        'invalid_count_min': np.array([int(x.invalid_count_min) for x in adc_stats], dtype=np.uint32),
        'invalid_count_max': np.array([int(x.invalid_count_max) for x in adc_stats], dtype=np.uint32),
        'sumv': np.array([int(x.sumv) for x in adc_stats], dtype=np.uint64),
        'sumv2': np.array([int(x.sumv2) for x in adc_stats], dtype=np.uint64),
    }


def route_to_arrays(routes):
    return {
        'plus': np.array([int(x.plus) for x in routes], dtype=np.uint8),
        'minus': np.array([int(x.minus) for x in routes], dtype=np.uint8),
    }

def metadata_to_dict(meta_pkt, print_debug=False) -> dict:
    """Convert metadata packet to dictionary"""
    if hasattr(meta_pkt, '_read'):
        meta_pkt._read()
    base = getattr(meta_pkt, 'base', meta_pkt)

    # if print_debug:
    #     ic(int(base.uC_time), int(base.time_32), int(base.time_16), int(base.Navgf), int(base.Navg1_shift), int(base.Navg2_shift))

    result = {
        '_uC_time': int(base.uC_time),
        '_time_32': int(base.time_32),
        '_time_16': int(base.time_16),
        'loop_count_min': int(base.loop_count_min),
        'loop_count_max': int(base.loop_count_max),
        'Navgf': int(base.Navgf),
        'Navg1_shift': int(base.Navg1_shift),
        'Navg2_shift': int(base.Navg2_shift),
        'notch': int(base.notch),
        'format': int(base.format),
        'corr_products_mask': int(base.corr_products_mask),
        'tr_start': int(base.tr_start),
        'tr_stop': int(base.tr_stop),
        'tr_avg_shift': int(base.tr_avg_shift),
        'grimm_enable': int(base.grimm_enable),
        'averaging_mode': int(base.averaging_mode),
        'reject_ratio': int(base.reject_ratio),
        'reject_maxbad': int(base.reject_maxbad),
        'bitslice_keep_bits': int(base.bitslice_keep_bits),
        'weight': int(base.weight),
        'weight_current': int(base.weight_current),
        'gain': [int(base.gain[i]) for i in range(4)],
        'actual_gain': [int(base.actual_gain[i]) for i in range(4)],
        'gain_auto_min': [int(base.gain_auto_min[i]) for i in range(4)],
        'gain_auto_mult': [int(base.gain_auto_mult[i]) for i in range(4)],
        'bitslice': [int(base.bitslice[i]) for i in range(NPRODUCTS)],
        'actual_bitslice': [int(base.actual_bitslice[i]) for i in range(NPRODUCTS)],
        'hi_frac': int(base.hi_frac),
        'med_frac': int(base.med_frac),
        'rand_state': int(base.rand_state),
        'num_bad_min_current': int(base.num_bad_min_current),
        'num_bad_max_current': int(base.num_bad_max_current),
        'num_bad_min': int(base.num_bad_min),
        'num_bad_max': int(base.num_bad_max),
        'spec_overflow': int(base.spec_overflow),
        'notch_overflow': int(base.notch_overflow),
        'errors': int(base.errors),
        'calibrator_enable': int(base.calibrator_enable),
        'spectrometer_enable': int(base.spectrometer_enable),
        'raw_ADC_stat': ADC_stat_to_arrays(base.ADC_stat),
        'route': route_to_arrays(base.route),
    }

    for attr_name, attr_value in vars(meta_pkt).items():
        if attr_name.startswith("adc_") or attr_name.startswith("telemetry_") or attr_name == "time":
            result[attr_name] = attr_value
        # if print_debug and attr_name == "time":
        #     ic(attr_value)

    return result
