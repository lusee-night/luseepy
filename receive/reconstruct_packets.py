import uncrater as unc
from uncrater.utils import *
import ctypes
import os
import hashlib
import sys
import time
from datetime import datetime
from icecream import ic

if os.environ.get("CORELOOP_DIR") is not None:
    sys.path.append(os.environ.get("CORELOOP_DIR"))

try:
    import pycoreloop
except ImportError:
    print("Can't import pycoreloop\n")
    print("Please install the package or setup CORELOOP_DIR to point at CORELOOP repo.")
    sys.exit(1)

from pycoreloop import appId
from pycoreloop import core_loop as cl

from low_level import *
from fits_writer import save_to_fits
from hdf5_writer import save_to_hdf5
from telemetry_utils import decode_telemetry_directory
from const_storage import Constants


def assign_uids(pkts: List[CollatedPacket]):
    # try to get uid from the packet itself
    for pkt in pkts:
        extract_unique_id(pkt)
    # for waveform packets, assign the uid of the nearest preceding packet that has it
    for i, pkt in enumerate(pkts):
        if appid_is_raw_adc(pkt.app_id) or pkt.app_id in [appId.AppID_uC_Bootloader]:
            pred = i
            while pkts[pred].unique_packet_id is None and pred > 0:
                pred -= 1
            pkt.unique_packet_id = pkts[pred].unique_packet_id


def decode_directory(path):
    begin = time.time()
    files = [path + f'/b0{i}/FFFFFFFE' for i in [5, 6, 7, 8, 9]]
    allpkts = []
    for f in files:
        if not os.path.exists(f):
            print(f"WARNING: file {f} not found, skipping")
            continue
        print(f"Decoding file {f}")
        data = open(f, 'rb').read()
        pkts = L0_to_ccsds(data)
        collated = collate_packets(pkts)
        allpkts.extend(collated)

    assign_uids(allpkts)
    no_uid_appids = set([hex(p.app_id) for p in allpkts if p.unique_packet_id is None])
    # ic(no_uid_appids)

    all_with_uid = [p for p in allpkts if p.unique_packet_id is not None]
    # ic(len(allpkts), len(all_with_uid))

    all_with_uid.sort(key=lambda x: (x.unique_packet_id, x.seq))
    elapsed = time.time() - begin
    # ic(elapsed)

    return all_with_uid


def extract_unique_id(pkt: CollatedPacket):
    # if we have the corresponding header struct which was memcpy-d to TLM_BUF,
    # instantiate it and read the corresponding unique_packet_id field
    # sometimes we just directly write unique_packet_id, e.g., in spectra_out
    # in this case, uid_start is the offset from TLM_BUF at which uid is written (normally 0)
    # if neither header_class nor uid are set, the packet does not contain unique_packet_id (e.g., waveform packet)
    header_class = None
    uid_start = None
    if pkt.app_id == appId.AppID_uC_Housekeeping:
        header_class = cl.housekeeping_data_base
    elif pkt.app_id == appId.AppID_uC_Start:
        header_class = cl.startup_hello
        # ic(pkt.app_id)
    elif pkt.app_id == appId.AppID_uC_Heartbeat:
        # check: heartbeat does not have unique_packet_id
        header_class = cl.heartbeat
        header = header_class.from_buffer(pkt.blob[:ctypes.sizeof(header_class)])
        pkt.unique_packet_id = header.time32
        raise RuntimeError("heartbeats should be filtered out")
    elif appid_is_spectrum(pkt.app_id):
        uid_start = 0
    elif appid_is_tr_spectrum(pkt.app_id):
        uid_start = 0
    elif appid_is_zoom_spectrum(pkt.app_id):
        uid_start = 0
    elif pkt.app_id == appId.AppID_Calibrator_MetaData:
        header_class = cl.calibrator_metadata
    elif pkt.app_id in [appId.AppID_Calibrator_Data, appId.AppID_Calibrator_RawPFB]:
        uid_start = 0
    elif appid_is_cal_debug(pkt.app_id):
        uid_start = 0
    elif pkt.app_id == appId.AppID_SpectraGrimm:
        uid_start = 0
    elif appid_is_metadata(pkt.app_id):
        header_class = cl.meta_data
    elif pkt.app_id == appId.AppID_RawADC_Meta:
        header_class = cl.waveform_metadata
    elif pkt.app_id == appId.AppID_uC_Bootloader:
        # sent by the firmware? no code in coreloop
        pass

    if header_class is not None:
        header = header_class.from_buffer(pkt.blob[:ctypes.sizeof(header_class)])
        pkt.unique_packet_id = header.unique_packet_id

    if uid_start is not None:
        pkt.unique_packet_id = struct.unpack_from("<I", pkt.blob[uid_start:uid_start + 4])[0]


def md5_file(filepath):
    """Compute MD5 hash of the file at filepath."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def write_sessions(base_path: str, sessions: List[List[CollatedPacket]], check_existing: bool = False) -> List[str]:
    """Write each session to its own directory."""
    os.makedirs(base_path, exist_ok=True)
    session_dirs = []

    for session_idx, session_packets in enumerate(sessions):
        session_id = get_session_start_seconds(session_packets)

        # Create directory name with session ID (as integer seconds)
        dir_name = f"cdi_output_{session_id}"
        session_dir = os.path.join(base_path, dir_name)
        os.makedirs(session_dir, exist_ok=True)
        session_dirs.append(session_dir)

        # Determine format string based on number of packets
        n_packets = len(session_packets)
        if n_packets >= 1000000:
            fmt_str = "{:06d}_{:04x}.bin"
        else:
            fmt_str = "{:05d}_{:04x}.bin"

        # Write packets
        for i, p in enumerate(session_packets):
            fname = os.path.join(session_dir, fmt_str.format(i, p.app_id))
            # if p.app_id in [appId.AppID_SpectraHigh, appId.AppID_ZoomSpectra]:
            #     ic(p.unique_packet_id)
            if os.path.exists(fname) and check_existing:
                existing_hash = md5_file(fname)
                new_hash = hashlib.md5(p.blob).hexdigest()
                if existing_hash != new_hash:
                    raise ValueError(f"File {fname} exists but content is different.")
                else:
                    continue
            with open(fname, 'wb') as f:
                f.write(p.blob)

    return session_dirs


def get_session_start_seconds(session_packets: List[CollatedPacket]) -> int:
    # Get session ID from first uC_Start packet
    session_id = None
    for pkt in session_packets:
        if pkt.app_id == appId.AppID_uC_Start:
            header = cl.startup_hello.from_buffer(pkt.blob[:ctypes.sizeof(cl.startup_hello)])
            # Use Time2Time to convert to proper timestamp
            session_id = unc.utils.Time2Time(header.time_32, header.time_16)
            break

    if session_id is None:
        session_id = 0

    return int(session_id)


def split_into_sessions(pkts: List[CollatedPacket]) -> List[List[CollatedPacket]]:
    """
    Split packets into sessions. A sequence of AppID_uC_Start packets
    (1, 2 or more in a row) signals the beginning of a new session.
    Sessions are cut when new AppID_uC_Start packets are encountered.
    """
    sessions = []
    current_session = []

    for i, pkt in enumerate(pkts):
        if pkt.app_id == appId.AppID_uC_Start:
            # Check if this is part of a sequence of uC_Start packets
            # or if it's a new start after other packets
            if current_session and current_session[-1].app_id != appId.AppID_uC_Start:
                # We have a non-empty session and the last packet wasn't uC_Start
                # This means we're starting a new session
                sessions.append(current_session)
                current_session = [pkt]
            else:
                # Either starting fresh or continuing a sequence of uC_Start packets
                current_session.append(pkt)
        else:
            # Regular packet, just add to current session
            current_session.append(pkt)

    # Don't forget the last session
    if current_session:
        sessions.append(current_session)

    return sessions


def app_id_category(curr_app_id):
    if appid_is_spectrum(curr_app_id):
        curr_app_id = appId.AppID_SpectraHigh
    if appid_is_tr_spectrum(curr_app_id):
        curr_app_id = appId.AppID_SpectraTRHigh
    if appid_is_raw_adc(curr_app_id):
        curr_app_id = appId.AppID_RawADC
    if appid_is_zoom_spectrum(curr_app_id):
        curr_app_id = appId.AppID_ZoomSpectra
    if appid_is_grimm_spectrum(curr_app_id):
        curr_app_id = appId.AppID_SpectraGrimm
    if curr_app_id == 0x314:
        ic(11111111)
    for app_str in dir(appId):
        if "AppID" not in app_str:
            continue
        if getattr(appId, app_str) == curr_app_id:
            return app_str
    raise RuntimeError(f"AppID {curr_app_id} not found in appId module")


def print_packet_categories(pkts):
    prev_app_id = None
    cat_count = 0
    for p in pkts:
        curr_app_id = app_id_category(p.app_id)
        if prev_app_id is None or curr_app_id != prev_app_id:
            if prev_app_id is not None:
                print(f"{prev_app_id.ljust(25, " ")}: {cat_count} packets")
            cat_count = 1  # Start with 1 for the current packet
            prev_app_id = curr_app_id
        else:
            cat_count += 1
    # the last group
    if prev_app_id is not None:
        print(f"{prev_app_id.ljust(25, " ")}:\t {cat_count} packets")


if __name__ == "__main__":
    # tel_packets = decode_telemetry_directory('new_data/20251105_112220/fs/FLASH_TLMFS')
    # for i, pkt in enumerate(tel_packets):
    #     ic(i, decode_telemetry_packet(pkt))
    # sys.exit(0)

    flash_path = 'new_data/20251105_112220/fs/FLASH_TLMFS'

    pkts = decode_directory(flash_path)
    print(len(pkts), 'collated packets found')

    if True:
        print_packet_categories(pkts)

    # Split into sessions
    sessions = split_into_sessions(pkts)
    print(f"Found {len(sessions)} sessions")

    session_start_seconds = [get_session_start_seconds(s) for s in sessions]
    dcb_tel_by_session = decode_telemetry_directory(
        flash_path,
        session_start_seconds,
        skip_out_of_session=False,
    )

    # Write sessions to directories
    base_output = "new_data/20251105_112220/sessions"
    session_dirs = write_sessions(base_output, sessions)

    # Save each session to HDF5
    if len(session_dirs) != len(dcb_tel_by_session):
        print(
            f"Warning: {len(session_dirs)} sessions but {len(dcb_tel_by_session)} telemetry blocks; truncating to shortest."
        )
    for i, (session_dir, dcb_tel_pair) in enumerate(zip(session_dirs, dcb_tel_by_session)):
        fpga_tel, encoder_tel = dcb_tel_pair
        # Get session ID from directory name
        session_id = session_dir.split('_')[-1]

        # Create output filename with session number and ID
        # Convert Unix timestamp to readable date if possible
        try:
            timestamp = int(session_id)
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
            output_file = f"session_{i:03d}_{date_str}"
        except:
            output_file = f"session_{i:03d}_{session_id}"

        h5_output_file = output_file + ".h5"
        fits_output_file = output_file + ".fits"

        print(f"Processing session {i}: {session_dir} -> {h5_output_file}, {fits_output_file}")
        save_to_hdf5(session_dir, h5_output_file, Constants(), fpga_tel, encoder_tel)
        print(f"Saved to {h5_output_file}")
        save_to_fits(session_dir, fits_output_file, Constants(), fpga_tel, encoder_tel)
        print(f"Saved to {fits_output_file}")
