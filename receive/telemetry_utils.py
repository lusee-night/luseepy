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

FIELD_NAMES = [
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

    values: Dict[str, list] = {name: [] for name in FIELD_NAMES}
    values["encoder_position"] = []
    values["encoder_status"] = []

    for pkt in pkts:
        if pkt.app_id == 0x314:
            row = decode_telemetry_packet(pkt)
            if row is None:
                continue
            for key in FIELD_NAMES:
                values[key].append(row[key])
        elif pkt.app_id == 0x325:
            pos, status = extract_encoder_info(pkt)
            values["encoder_position"].append(pos)
            values["encoder_status"].append(status)

    return {key: np.asarray(vals) for key, vals in values.items()}



def decode_telemetry_packet(packet: int):
    """
      1) Extract payload bits (after header).
      2) Parse 12-bit unsigned fields into a dict.
      3) Apply per-field calibration formulas.
    """
    if packet.app_id != 0x314:
        return None
    payload = packet.blob
    stream = BitStream(payload)

    _ = stream.read("uint:32")
    _ = stream.read("uint:16")

    row = {field: stream.read("uint:12") for field in FIELD_NAMES}

    # Apply engineering-unit conversions
    SPE_1VA8_V = 0.0025373 * row["SPE_1VA8_V"] - 0.0474504

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


def extract_encoder_info(packet) -> Tuple[int, int]:
    _ = packet
    return 0, 0
