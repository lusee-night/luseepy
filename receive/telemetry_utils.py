#!/usr/bin/env python3
import math
import os.path
from typing import Dict, Tuple

import numpy as np
from bitstring import BitStream

from low_level import *


# ---------------------------
# Packet / field definitions
# ---------------------------

TELEMETRY_FIELDS = [
    ("mission_seconds", "uint:32"),
    ("lusee_subsecs", "uint:16"),
    ("THERM_FPGA", "uint:12"), ("THERM_DCB", "uint:12"), ("VMON_6V", "uint:12"), ("VMON_3V7", "uint:12"), ("VMON_1V8", "uint:12"), ("VMON_3V3D", "uint:12"),
    ("VMON_2V5D", "uint:12"), ("VMON_1V2D", "uint:12"), ("VMON_VCC_FLASH0", "uint:12"), ("VMON_VCC_FLASH1", "uint:12"),
    ("VMON_VCC_FLASH2", "uint:12"), ("VMON_VCC_FLASH3", "uint:12"), ("VMON_3V3_HK", "uint:12"), ("VMON_GND", "uint:12"),
    ("GND_RES", "uint:12"), ("GND_RES2", "uint:12"), ("SPE_P5_V", "uint:12"), ("SPE_P5_C", "uint:12"), ("SPE_N5_V", "uint:12"), ("SPE_N5_C", "uint:12"),
    ("SPE_1VA8_V", "uint:12"), ("SPE_1VA8_C", "uint:12"), ("SPE_1VAD8_V", "uint:12"), ("SPE_1VAD8_C", "uint:12"),
    ("SPE_3VD3_V", "uint:12"), ("SPE_3VD3_C", "uint:12"), ("SPE_2VD5_V", "uint:12"), ("SPE_2VD5_C", "uint:12"),
    ("SPE_1VD8_V", "uint:12"), ("SPE_1VD8_C", "uint:12"), ("SPE_1VD5_V", "uint:12"), ("SPE_1VD5_C", "uint:12"),
    ("SPE_1VD0_V", "uint:12"), ("SPE_1VD0_C", "uint:12"), ("SPE_FPGA_T", "uint:12"), ("SPE_ADC0_T", "uint:12"), ("SPE_ADC1_T", "uint:12"),
    ("PFPS_DCB_1V8", "uint:12"), ("PFPS_DCB_3V7", "uint:12"), ("PFPS_DCB_5V", "uint:12"), ("PFPS_SPE_1V5", "uint:12"),
    ("PFPS_SPE_2V3", "uint:12"), ("PFPS_SPE_3V6", "uint:12"), ("PFPS_SPE_P5V5", "uint:12"), ("PFPS_SPE_N5V5", "uint:12"),
    ("PFPS_MOT_A", "uint:12"), ("PFPS_PA3_T", "uint:12"), ("PFPS_PA2_T", "uint:12"), ("PFPS_PA1_T", "uint:12"), ("PFPS_PA0_T", "uint:12"),
    ("PFPS_CAR_T", "uint:12"), ("PFPS_PFPS_T", "uint:12"), ("PFPS_BAT_T", "uint:12"), ("VMON_PDU_COMMS", "uint:12"),
    ("VMON_PDU_PFPS", "uint:12"), ("VMON_PDU_CAROUSEL", "uint:12"), ("ADC_PWR", "uint:12")
]

ENCODER_FIELDS = [
    ("mission_seconds", "uint:32"),
    ("lusee_subsecs", "uint:16"),
    ("mbstats_cmd_cnt", "uint:32"),
    ("mbstats_cmd_proc_errors", "uint:32"),
    ("mbstats_hsk_req_cnt", "uint:32"),
    ("mbstats_hsk_req_proc_errors", "uint:32"),
    ("mbstats_misc_cnt", "uint:32"),
    ("mbstats_misc_proc_errors", "uint:32"),
    ("mbstats_dropped_msgs", "uint:32"),
    ("mbstats_received_msgs", "uint:32"),
    ("mbstats_peak_msgs", "uint:32"),
    ("iostats_secs_since_last_rx", "uint:32"),
    ("iostats_secs_since_last_tx", "uint:32"),
    ("iostats_total_read_bytes", "uint:32"),
    ("iostats_total_dropped_rx_bytes", "uint:32"),
    ("iostats_total_read_errors", "uint:32"),
    ("iostats_written_bytes", "uint:32"),
    ("iostats_write_errors", "uint:32"),
    ("pcdu_tlm_pkt_proc_errors", "uint:32"),
    ("pcdu_crc_errors", "uint:32"),
    ("pcdu_oos_cnt", "uint:32"),
    ("pcdu_cmt_cnt", "uint:32"),
    ("pcdu_bcr_hk_cnt", "uint:32"),
    ("pcdu_apr_hk_cnt", "uint:32"),
    ("enc_pos", "uint:32"),
    ("enc_status", "uint:32"),
]

ENCODER_KEEP_FIELDS = [ "mission_seconds", "lusee_subsecs", "enc_pos", "enc_status" ]

TELEMETRY_FIELD_NAMES = [
    "THERM_FPGA", "THERM_DCB", "VMON_6V", "VMON_3V7", "VMON_1V8", "VMON_3V3D",
    "VMON_2V5D", "VMON_1V2D", "VMON_VCC_FLASH0", "VMON_VCC_FLASH1",
    "VMON_VCC_FLASH2", "VMON_VCC_FLASH3", "VMON_3V3_HK", "VMON_GND",
    "GND_RES", "GND_RES2", "SPE_P5_V", "SPE_P5_C", "SPE_N5_V", "SPE_N5_C",
    "SPE_1VA8_V", "SPE_1VA8_C", "SPE_1VAD8_V", "SPE_1VAD8_C",
    "SPE_3VD3_V", "SPE_3VD3_C", "SPE_2VD5_V", "SPE_2VD5_C",
    "SPE_1VD8_V", "SPE_1VD8_C", "SPE_1VD5_V", "SPE_1VD5_C",
    "SPE_1VD0_V", "SPE_1VD0_C", "SPE_FPGA_T", "SPE_ADC0_T", "SPE_ADC1_T",
    "PFPS_DCB_1V8", "PFPS_DCB_3V7", "PFPS_DCB_5V", "PFPS_SPE_1V5",
    "PFPS_SPE_2V3", "PFPS_SPE_3V6", "PFPS_SPE_P5V5", "PFPS_SPE_N5V5",
    "PFPS_MOT_A", "PFPS_PA3_T", "PFPS_PA2_T", "PFPS_PA1_T", "PFPS_PA0_T",
    "PFPS_CAR_T", "PFPS_PFPS_T", "PFPS_BAT_T", "VMON_PDU_COMMS",
    "VMON_PDU_PFPS", "VMON_PDU_CAROUSEL", "ADC_PWR"
]

TELEMETRY_PACKET_SIZE = 98      # bytes
HEADER_SIZE = 12      # bytes


# ---------------------------
# Conversion helpers
# ---------------------------

def dcb_thermistor(x):
    """DCB thermistor conversion from ADC counts to °C."""
    try:
        A = 2.5
        B = 10000
        U = (4.068 / 4096) * x
        RK = (U * B / (A - U)) / 1000
        RLOG = math.log(RK)
        return round(1.303 * (RLOG ** 2) - 31.38 * RLOG + 91.44, 3)
    except Exception:
        return 0.0


def SPE_FPGA_T(x, V=2.07):
    """FPGA temperature conversion using measured 1VA8 rail."""
    try:
        R = (25.37 * x - 474.5) / (V + 0.04745 - 0.002537 * x)
        return round(3969 / (13.31 - math.log(10000 / R)) - 273.15, 3)
    except Exception:
        return 0.0


def pfps_1k_thermistor(counts):
    """PFPS 1k thermistor conversion."""
    try:
        A = 5.010
        B = 1000
        U = (4.068 / 4096) * counts * 2
        R = (U * B / (A - U))
        return round(((R - B) / 3.85), 3)
    except Exception:
        return 0.0


def spec_adc(x, V=2.07):
    """Shared ADC temperature conversion."""
    try:
        R = (25.37 * x - 474.5) / (V + 0.04745 - 0.002537 * x)
        return round(3694 / (12.39 - math.log(10000 / R)) - 273.15, 3)
    except Exception:
        # ic(x, V, R)
        # raise
        return 0.0


def SPE_ADC0_T(x, V=2.07): return spec_adc(x, V)
def SPE_ADC1_T(x, V=2.07): return spec_adc(x, V)


# ---------------------------
# Packet decoding
# ---------------------------

def decode_telemetry_directory(path) -> Dict[str, np.ndarray]:
    f = os.path.join(path, "b01", "FFFFFFFE")
    if not os.path.exists(f):
        print(f"WARNING: file {f} not found, skipping")
        return {}

    print(f"Decoding telemetry file {f}")
    data = open(f, "rb").read()
    pkts = L0_to_ccsds(data)
    pkts = extract_telemetry_packets(pkts)

    telemetry_values: Dict[str, list] = {name: [] for name in TELEMETRY_FIELD_NAMES}

    enc_values = { name : [] for name in ENCODER_KEEP_FIELDS }

    for pkt in pkts:
        if pkt.app_id == 0x314:
            row = decode_telemetry_packet(pkt)
            if row is None:
                continue
            for key in TELEMETRY_FIELD_NAMES:
                telemetry_values[key].append(row[key])
        elif pkt.app_id == 0x325:
            row = extract_encoder_info(pkt)
            for key in ENCODER_KEEP_FIELDS:
                enc_values[key].append(row[key])

    return {key: np.asarray(vals) for key, vals in enc_values.items()}


def decode_telemetry_packet(packet):
    """
      1) Extract payload bits (after header).
      2) Parse 12-bit unsigned fields into a dict.
      3) Apply per-field calibration formulas.
    """
    if packet.app_id != 0x314:
        return None
    payload = packet.blob
    stream = BitStream(payload)

    row = {}
    for field, field_type in TELEMETRY_FIELDS:
        row[field] = stream.read(field_type)

    old = row["SPE_1VA8_V"]
    # Apply engineering-unit conversions
    SPE_1VA8_V = 0.0025373 * row["SPE_1VA8_V"] - 0.0474504

    ic(hex(packet.app_id), old, row["mission_seconds"], SPE_1VA8_V)

    row["THERM_FPGA"] = dcb_thermistor(row["THERM_FPGA"])
    row["THERM_DCB"] = dcb_thermistor(row["THERM_DCB"])
    row["SPE_FPGA_T"] = SPE_FPGA_T(row["SPE_FPGA_T"], SPE_1VA8_V)
    row["SPE_ADC0_T"] = SPE_ADC0_T(row["SPE_ADC0_T"], SPE_1VA8_V)
    row["SPE_ADC1_T"] = SPE_ADC1_T(row["SPE_ADC1_T"], SPE_1VA8_V)

    row["SPE_P5_V"] = round(0.0134 * row["SPE_P5_V"] - 0.2505, 3)
    row["SPE_P5_C"] = round(0.0001269 * row["SPE_P5_C"] - 0.002373, 3)
    row["SPE_N5_V"] = round(-0.01021 * row["SPE_N5_V"] + 0.1908, 3)
    row["SPE_N5_C"] = round(0.0001269 * row["SPE_N5_C"] - 0.002373, 3)

    for key in ["SPE_1VA8", "SPE_1VAD8", "SPE_3VD3", "SPE_2VD5",
                "SPE_1VD8", "SPE_1VD5", "SPE_1VD0"]:
        row[f"{key}_V"] = round(0.0025373 * row[f"{key}_V"] - 0.0474504, 3)
        if key in ["SPE_1VA8", "SPE_1VAD8"]:
            row[f"{key}_C"] = round(0.0003252 * row[f"{key}_C"] - 0.006083, 3)
        else:
            row[f"{key}_C"] = round(0.00001269 * row[f"{key}_C"] - 0.0002373, 3)

    for key in ["VMON_6V", "VMON_3V7", "VMON_VCC_FLASH0", "VMON_VCC_FLASH1",
                "VMON_VCC_FLASH2", "VMON_VCC_FLASH3", "VMON_3V3_HK"]:
        row[key] = round((4.0 / 4096.0) * row[key] * 2.0, 3)

    for key in ["VMON_1V8", "VMON_2V5D", "VMON_1V2D",
                "VMON_GND", "GND_RES", "GND_RES2"]:
        row[key] = round((4.0 / 4096.0) * row[key], 3)

    for key in ["PFPS_PA3_T", "PFPS_PA2_T", "PFPS_PA1_T", "PFPS_PA0_T"]:
        row[key] = pfps_1k_thermistor(row[key])

    row["VMON_3V3D"] = round((4.0 / 4096.0) * row["VMON_3V3D"] * 2.0, 3)

    return row


def extract_encoder_info(packet) -> dict:
    if packet.app_id != 0x325:
        return None

    result = {}

    payload = packet.blob
    stream = BitStream(payload)

    for field_name, field_type in ENCODER_FIELDS:
        field_value = stream.read(field_type)
        if field_name in ENCODER_KEEP_FIELDS:
            result[field_name] = field_value

    return result
