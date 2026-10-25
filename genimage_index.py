"""Read the file index of a split GenImage archive (.z01, .z02, ..., .zip) using only the
last part (.zip), and report which parts hold the val/ images.

    python genimage_index.py data/genimage/raw/imagenet_glide.zip

Split zips keep the central directory (the index) in the last part. Each entry records the
part ("disk") its data starts on, so we can download only the parts we need.
"""
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_index(last_part):
    data = Path(last_part).read_bytes()
    eocd = data.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("end-of-central-directory record not found")
    (_, this_disk, cd_disk, _, n_total, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", data[eocd : eocd + 22])

    # Zip64: the real values live in the zip64 end-of-central-directory record
    loc = data.rfind(b"PK\x06\x07", 0, eocd)
    if loc >= 0:
        (_, z64_disk, z64_off, _) = struct.unpack("<IIQI", data[loc : loc + 20])
        z64 = data.rfind(b"PK\x06\x06", 0, loc)
        (_, _, _, _, this_disk, cd_disk, _, n_total, cd_size, cd_offset) = struct.unpack(
            "<IQHHIIQQQQ", data[z64 : z64 + 56])

    if cd_disk != this_disk:
        raise ValueError(f"index starts on part {cd_disk + 1}, not in this file; download that part too")
    entries, pos = [], cd_offset
    for _ in range(n_total):
        if data[pos : pos + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central directory entry at {pos}")
        (method, csize, usize, nlen, xlen, clen, disk, offset) = (
            *struct.unpack("<H", data[pos + 10 : pos + 12]),
            *struct.unpack("<II", data[pos + 20 : pos + 28]),
            *struct.unpack("<HHHH", data[pos + 28 : pos + 36]),
            *struct.unpack("<I", data[pos + 42 : pos + 46]))
        name = data[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
        extra = data[pos + 46 + nlen : pos + 46 + nlen + xlen]
        # Zip64 extra field (id 0x0001): 64-bit values replace any 0xFFFF... placeholders, in order
        i = 0
        while i + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[i : i + 4])
            if hid == 1:
                vals, j = extra[i + 4 : i + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize, j = struct.unpack("<Q", vals[j : j + 8])[0], j + 8
                if csize == 0xFFFFFFFF:
                    csize, j = struct.unpack("<Q", vals[j : j + 8])[0], j + 8
                if offset == 0xFFFFFFFF:
                    offset, j = struct.unpack("<Q", vals[j : j + 8])[0], j + 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", vals[j : j + 4])[0]
            i += 4 + hlen
        entries.append({"name": name, "method": method, "csize": csize, "usize": usize,
                        "disk": disk, "offset": offset})
        pos += 46 + nlen + xlen + clen
    return entries, this_disk + 1  # number of parts


def part_name(last_part, disk, n_parts):
    p = Path(last_part)
    return p.name if disk == n_parts - 1 else f"{p.stem}.z{disk + 1:02d}"


def main(last_part):
    entries, n_parts = read_index(last_part)
    files = [e for e in entries if not e["name"].endswith("/")]
    print(f"{Path(last_part).name}: {len(files)} files in {n_parts} parts")
    groups = defaultdict(Counter)
    for e in files:
        parts = e["name"].split("/")
        key = "/".join(p for p in parts[1:-1]) or "(root)"  # e.g. val/ai, val/nature, train/ai
        groups[key][part_name(last_part, e["disk"], n_parts)] += 1
    for key in sorted(groups):
        locs = ", ".join(f"{p}: {n}" for p, n in sorted(groups[key].items()))
        print(f"  {key:14s} {sum(groups[key].values()):>7} files   -> {locs}")
    print("compression methods:", Counter(e["method"] for e in files), "(0 = stored, 8 = deflate)")


if __name__ == "__main__":
    main(sys.argv[1])
