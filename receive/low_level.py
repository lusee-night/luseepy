import struct
import enum
from typing import Optional, List
from icecream import ic

def decode_ccsds_header(pkt) -> dict:
    """Decode CCSDS packet header."""
    formatted_data = struct.unpack_from(f">3H", pkt[:6])
    header = {}
    header['version'] = (formatted_data[0] >> 13)
    header['packet_type'] = ((formatted_data[0] >> 12) & 0x1)
    header['secheaderflag'] = ((formatted_data[0] >> 11) & 0x1)
    header['appid'] = (formatted_data[0] & 0x7FF)
    header['groupflags'] = (formatted_data[1] >> 14)
    header['sequence_cnt'] = (formatted_data[1] & 0x3FFF)
    header['packetlen'] = (formatted_data[2])
    # print(header)
    return header


def crc16_ccitt(data: bytearray) -> int:
    """Compute CRC-16-CCITT checksum.
        We use 16-bit big-endian CRC-CCITT (polynomial=0x1021, initial value=0xFFFF) """
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if (crc & 0x8000) != 0:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF  # Keep CRC as 16-bit value
    return crc


class PState(enum.Enum):
    FINDING_SYNC = enum.auto()
    READING_LEN = enum.auto()
    READING_DATA = enum.auto()


def L0_to_ccsds(data) -> list:
    """Convert L0 data to list of CCSDS packets (bytearray)"""
    pkts = []
    state = PState.FINDING_SYNC
    sync = 0
    garb = 0
    for v in data:
        match state:
            case PState.FINDING_SYNC:
                sync = ((sync << 8) | v) & 0xFFFF
                if v == 0xA5:
                    # 0xA5 is used for padding, ignore it
                    continue
                if sync == 0xECA0:
                    state = PState.READING_LEN
                    pkt_head = bytearray()
                    pkt_body = bytearray()
                    sync = 0
                    garb -= 1  # Compensate for earlier garb count when we see 0xEC
                    continue
                else:
                    garb += 1
                    continue

            case PState.READING_LEN:
                pkt_head.append(v)
                if len(pkt_head) == 6:
                    state = PState.READING_DATA
                    header = decode_ccsds_header(pkt_head)
                    pllen = header['packetlen']
                    sequence_cnt = header['sequence_cnt']
                    # print ('apid=', hex(header['appid']), 'seq=', sequence_cnt, 'len=', pllen)
                    continue

            case PState.READING_DATA:
                pkt_body.append(v)
                if len(pkt_body) >= pllen + 3:
                    # Check CRC
                    pktcrc = struct.unpack('>H', pkt_body[-2:])[0]
                    compcrc = crc16_ccitt(pkt_head + pkt_body[:-2])

                    if pktcrc != compcrc:
                        print(f"CRC mismatch in packet (pktcrc=0x{pktcrc:04X} compcrc=0x{compcrc:04X})")

                    pkts.append((sequence_cnt, pkt_head, pkt_body[:-2]))
                    state = PState.FINDING_SYNC

    print(f"Found {len(pkts)} packets, garb={garb}")
    return pkts


def reorder(data):
    cdata = bytearray(len(data))
    cdata[::2] = data[1::2]
    cdata[1::2] = data[::2]
    return cdata


class CollatedPacket:
    def __init__(self, start_seq: int, seq: int,
                 app_id: int, blob, single_packet: bool, unique_packet_id: Optional[int] = None):
        self.start_seq = start_seq
        self.seq = seq
        self.app_id = app_id
        self.blob = blob
        self.single_packet = single_packet
        self.unique_packet_id = unique_packet_id


def collate_packets(pkts) -> List[CollatedPacket]:
    """ Collate logical packets that have been split into multiple CCSDS packets."""
    collated = []
    pkt = bytearray()
    last_apid = None
    start_seq = None
    single_packet = True
    app_ids = dict()
    for p in pkts:
        seq, head, body = p
        pkt += reorder(body)
        header = decode_ccsds_header(head)
        cur_apid = header['appid']
        # print (f"Seq={seq} APID={hex(cur_apid)} GF={header['groupflags']} Len={len(body)} TotalLen={len(pkt)}")
        if last_apid is not None and last_apid != cur_apid:
            print(f"Warning: APID changed from {hex(last_apid)} to {hex(cur_apid)} in sequence {seq}")
        if header['groupflags'] == 3:
            # end of multi-packet
            if single_packet:
                start_seq = seq
            collated.append(CollatedPacket(start_seq=start_seq,
                                           seq=seq, app_id=cur_apid,
                                           blob=pkt, single_packet=single_packet))
            if cur_apid not in app_ids:
                app_ids[cur_apid] = 0
            else:
                app_ids[cur_apid] += 1
            pkt = bytearray()
            last_apid = None
            single_packet = True
        else:
            last_apid = cur_apid
            start_seq = seq
            single_packet = False
    return collated


def extract_telemetry_packets(pkts) -> List[CollatedPacket]:
    """ Collate logical packets that have been split into multiple CCSDS packets."""
    telemetry_appids = [0x314, 0x325]
    collated = []
    pkt = bytearray()
    last_apid = None
    start_seq = None
    single_packet = True
    app_ids = dict()
    for p in pkts:
        seq, head, body = p
        pkt += body
        header = decode_ccsds_header(head)
        cur_apid = header['appid']
        if cur_apid in telemetry_appids:
            print (f"Seq={seq} APID={hex(cur_apid)} GF={header['groupflags']} Len={len(body)} TotalLen={len(pkt)}")
            print(header)
            print(body.hex().upper())
            # print(reorder(body).hex().upper())
        if last_apid is not None and last_apid != cur_apid:
            print(f"Warning: APID changed from {hex(last_apid)} to {hex(cur_apid)} in sequence {seq}")
        if header['groupflags'] == 3:
            # end of multi-packet
            if single_packet:
                start_seq = seq
            else:
                assert cur_apid not in telemetry_appids
            if cur_apid in telemetry_appids:
                assert single_packet
                collated.append(CollatedPacket(start_seq=start_seq,
                                               seq=seq, app_id=cur_apid,
                                               blob=pkt, single_packet=single_packet))
            if cur_apid not in app_ids:
                app_ids[cur_apid] = 0
            else:
                app_ids[cur_apid] += 1
            pkt = bytearray()
            last_apid = None
            single_packet = True
        else:
            assert cur_apid not in telemetry_appids
            last_apid = cur_apid
            start_seq = seq
            single_packet = False
    return collated

