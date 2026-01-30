#!python3

from uncrater.utils import NPRODUCTS


def ADC_stat_to_dict(adc_stat):
    return { 'min' : int(adc_stat.min),
             'max': int(adc_stat.max),
             'valid_count': int(adc_stat.valid_count),
             'invalid_count_min': int(adc_stat.invalid_count_min),
             'invalid_count_max': int(adc_stat.invalid_count_max),
             'sumv': int(adc_stat.sumv),
             'sumv2': int(adc_stat.sumv2),
             }

def route_to_dict(route):
    return { 'plus' : int(route.plus),
             'minux': int(route.minus) }

def metadata_to_dict(meta_pkt) -> dict:
    """Convert metadata packet to dictionary"""
    if hasattr(meta_pkt, '_read'):
        meta_pkt._read()
    base = getattr(meta_pkt, 'base', meta_pkt)

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
        'raw_ADC_stat': [ ADC_stat_to_dict(base.ADC_stat[i]) for i in range(4) ],
        'route': [ route_to_dict(base.route[i]) for i in range(4)],
    }

    for attr_name, attr_value in vars(meta_pkt).items():
        if attr_name.startswith("adc_") or attr_name.startswith("telemetry_") or attr_name == "time":
            result[attr_name] = attr_value

    return result