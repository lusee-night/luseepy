#!python3

from uncrater.utils import NPRODUCTS

def metadata_to_dict(meta_pkt) -> dict:
    """Convert metadata packet to dictionary"""
    if hasattr(meta_pkt, '_read'):
        meta_pkt._read()
    base = getattr(meta_pkt, 'base', meta_pkt)

    return {
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
        'bitslice': [int(base.bitslice[i]) for i in range(NPRODUCTS)],
        'actual_bitslice': [int(base.actual_bitslice[i]) for i in range(NPRODUCTS)],
    }