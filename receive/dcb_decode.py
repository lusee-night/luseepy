#!/usr/bin/env python3

"""
Decode all session DCB telemetry files and write telemetry-only HDF5 outputs.

The input file extension is .json, but files are binary packet streams.
For each `session_*/DCB_telemetry.json`, this script writes
`session_*/DCB_telemetry.h5` with a single group:
  /DCB_telemetry
"""

import math
import struct
from pathlib import Path

import h5py
import numpy as np
from bitstring import BitStream
from icecream import ic


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

PACKET_SIZE = 98      # bytes
HEADER_SIZE = 12      # bytes
NUM_FIELDS = 57


# ---------------------------
# Conversion helpers
# ---------------------------

def dcb_thermistor(x):
    """DCB thermistor conversion from ADC counts to °C."""
    A = 2.5
    B = 10000
    U = (4.068 / 4096) * x
    RK = (U * B / (A - U)) / 1000
    RLOG = math.log(RK)
    return round(1.303 * (RLOG ** 2) - 31.38 * RLOG + 91.44, 3)


def SPE_FPGA_T(x, V=2.07):
    """FPGA temperature conversion using measured 1VA8 rail."""
    R = (25.37 * x - 474.5) / (V + 0.04745 - 0.002537 * x)
    return round(3969 / (13.31 - math.log(10000 / R)) - 273.15, 3)


def pfps_1k_thermistor(counts):
    """PFPS 1k thermistor conversion."""
    A = 5.010
    B = 1000
    U = (4.068 / 4096) * counts * 2
    R = (U * B / (A - U))
    return round(((R - B) / 3.85), 3)


def spec_adc(x, V=2.07):
    """Shared ADC temperature conversion."""
    R = (25.37 * x - 474.5) / (V + 0.04745 - 0.002537 * x)
    return round(3694 / (12.39 - math.log(10000 / R)) - 273.15, 3)


def SPE_ADC0_T(x, V=2.07): return spec_adc(x, V)
def SPE_ADC1_T(x, V=2.07): return spec_adc(x, V)


# ---------------------------
# Packet decoding
# ---------------------------

def decode_packets(filepath):
    """
    Decode fixed-size DCB telemetry packets into physical values.

    Steps:
      1) Read all complete PACKET_SIZE packets from file.
      2) Parse mission time from packet header.
      3) Extract payload bits (after header).
      3) Parse 12-bit unsigned fields into a dict.
      4) Apply per-field calibration formulas.
    """
    records = []
    mission_seconds = []
    lusee_subsecs = []

    with open(filepath, "rb") as f:
        data = f.read()

    packet_count = len(data) // PACKET_SIZE
    remainder = len(data) % PACKET_SIZE
    if remainder:
        print(
            f"Warning: {filepath} has {remainder} trailing byte(s) "
            f"after {packet_count} complete packets; ignoring tail."
        )

    for i in range(packet_count):
        packet = data[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
        if len(packet) < PACKET_SIZE:
            continue

        # Header format follows telemetry_utils.py expectations:
        # first 4 bytes mission seconds, then 2 bytes LuSEE subseconds.
        ms = struct.unpack(">I", packet[:4])[0]
        subsec = struct.unpack(">H", packet[4:6])[0]
        mission_seconds.append(ms)
        lusee_subsecs.append(subsec)

        payload = packet[HEADER_SIZE:]
        stream = BitStream(payload)
        record = {field: stream.read("uint:12") for field in FIELD_NAMES}
        records.append(record)

    # Apply engineering-unit conversions
    for row in records:
        SPE_1VA8_V = round(0.0025373 * row["SPE_1VA8_V"] - 0.0474504, 3)

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

    out = {
        "fpga_mission_seconds": np.asarray(mission_seconds, dtype=np.uint32),
        "fpga_lusee_subsecs": np.asarray(lusee_subsecs, dtype=np.uint16),
    }
    for name in FIELD_NAMES:
        out[name] = np.asarray([row[name] for row in records])

    return out


def write_telemetry_hdf5(output_path, telemetry):
    with h5py.File(output_path, "w") as h5f:
        group = h5f.create_group("DCB_telemetry")
        for name, values in telemetry.items():
            group.create_dataset(name, data=np.asarray(values), compression="gzip")


def iter_session_files(base_dir):
    for session_dir in sorted(base_dir.glob("session_*")):
        if not session_dir.is_dir():
            continue
        in_file = session_dir / "DCB_telemetry.json"
        if in_file.exists():
            yield session_dir, in_file


if __name__ == "__main__":
    base_dir = Path(".")
    session_files = list(iter_session_files(base_dir))

    if not session_files:
        print("No session_*/DCB_telemetry.json files found.")
        raise SystemExit(0)

    for i, (session_dir, in_file) in enumerate(session_files, 1):
        out_file = session_dir / "DCB_telemetry.h5"
        print(f"[{i}] Decoding {in_file} -> {out_file}")
        telemetry = decode_packets(in_file)
        write_telemetry_hdf5(out_file, telemetry)
        n_rows = len(telemetry["fpga_mission_seconds"])
        print(f"[{i}] Saved {n_rows} packets.")

    print("Done.")
