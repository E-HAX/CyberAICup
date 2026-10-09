"""Ingestion of the auxiliary MIRAGE-AppAct-2024 corpus.

The archive is 6.97 GB. The server supports byte ranges, so members can be
listed and read selectively without downloading the whole file: `HttpZip` gives
`zipfile` a seekable file-like object backed by HTTP range requests. That makes
inspection cheap and lets the reduction step fan out - each container pulls only
the members of its own shard.

Whatever is read is immediately reduced to the only fields the model needs
(per-packet payload size, inter-arrival gap and direction, truncated to the
first N packets) and stored as compact float32 arrays. The raw archive is never
required on the local machine and, once reduced, need not be kept at all.
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

MIRAGE_URL = "https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-AppAct-2024.zip"

RAW_DIR = Path(os.environ.get("RTC_MIRAGE", "/mirage")) / "raw"
REDUCED_DIR = Path(os.environ.get("RTC_MIRAGE", "/mirage")) / "reduced"

# Field-name patterns, kept loose because the corpus has varied across releases.
LEN_KEYS = ("l4_payload_bytes", "packet_length", "l4_payload_len", "payload_len")
IAT_KEYS = ("iat", "inter_arrival", "timestamp", "time")
DIR_KEYS = ("packet_dir", "direction", "dir")


class HttpZip:
    """Minimal seekable HTTP reader so zipfile can work over the network."""

    def __init__(self, url: str, timeout: int = 60):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not advertise byte ranges")

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        self.pos += len(data)
        return data


def open_remote(url: str = MIRAGE_URL) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpZip(url))


def inspect(url: str = MIRAGE_URL, n_names: int = 25) -> dict:
    """List members and describe the JSON schema of one sample, over the network."""
    zf = open_remote(url)
    names = zf.namelist()
    json_names = [n for n in names if n.lower().endswith(".json")]
    sample_report = {}
    if json_names:
        smallest = min(
            (zf.getinfo(n) for n in json_names[:400]), key=lambda i: i.file_size
        )
        raw = zf.read(smallest.filename)
        obj = json.loads(raw)
        sample_report = {
            "member": smallest.filename,
            "bytes": smallest.file_size,
            "top_level_type": type(obj).__name__,
            "n_top_level": len(obj),
            "first_key": next(iter(obj)) if isinstance(obj, dict) else None,
            "structure": _describe(obj, depth=0),
        }
    return {
        "n_members": len(names),
        "n_json": len(json_names),
        "total_uncompressed_gb": round(
            sum(zf.getinfo(n).file_size for n in names) / 1e9, 2
        ),
        "sample_names": names[:n_names],
        "sample": sample_report,
    }


def schema(url: str = MIRAGE_URL) -> dict:
    """Compact schema reconnaissance: what a flow record actually contains."""
    zf = open_remote(url)
    names = zf.namelist()
    json_names = [n for n in names if n.lower().endswith(".json")]
    info = {
        "n_members": len(names),
        "n_json": len(json_names),
        "sample_names": json_names[:8],
        "dir_prefixes": sorted({str(Path(n).parent) for n in json_names})[:20],
    }
    if not json_names:
        return info
    smallest = min((zf.getinfo(n) for n in json_names[:300]), key=lambda i: i.file_size)
    obj = json.loads(zf.read(smallest.filename))
    info["sample_member"] = smallest.filename
    info["n_flows_in_member"] = len(obj)
    first = next(iter(obj.values())) if isinstance(obj, dict) else obj[0]
    info["flow_sections"] = list(first)
    for section in first:
        block = first[section]
        if not isinstance(block, dict):
            continue
        if section.lower().startswith("packet"):
            info["packet_keys"] = list(block)
            info["packet_head"] = {
                k: (v[:8] if isinstance(v, list) else str(v)[:80]) for k, v in block.items()
            }
        if "metadata" in section.lower():
            info["metadata"] = {k: str(v)[:60] for k, v in block.items()}
    return info


def _describe(obj, depth: int, max_depth: int = 4):
    if depth > max_depth:
        return "..."
    if isinstance(obj, dict):
        keys = list(obj)[:12]
        return {k: _describe(obj[k], depth + 1, max_depth) for k in keys}
    if isinstance(obj, list):
        return [f"list[{len(obj)}]", _describe(obj[0], depth + 1, max_depth) if obj else None]
    return type(obj).__name__


# ----------------------------------------------------------- downloading ----


def fetch(url: str = MIRAGE_URL, dest: Path | None = None, chunk_mb: int = 64) -> dict:
    """Resumable download into the Volume, checkpointing after every chunk."""
    dest = dest or (RAW_DIR / Path(url).name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = HttpZip(url)
    have = dest.stat().st_size if dest.exists() else 0
    chunk = chunk_mb * 1024 * 1024
    with open(dest, "ab") as fh:
        while have < src.size:
            src.seek(have)
            data = src.read(min(chunk, src.size - have))
            if not data:
                raise RuntimeError(f"empty read at offset {have}")
            fh.write(data)
            fh.flush()
            have += len(data)
    return {"path": str(dest), "bytes": have, "complete": have == src.size}


# ------------------------------------------------------------- reduction ----


def _find_key(d: dict, candidates: tuple[str, ...]) -> str | None:
    lower = {k.lower(): k for k in d}
    for cand in candidates:
        for lk, orig in lower.items():
            if cand in lk:
                return orig
    return None


def _label_from_name(name: str) -> str:
    """Application/activity label carried by the archive's directory layout."""
    parts = [p for p in Path(name).parts if p not in (".", "/")]
    if len(parts) >= 2:
        return parts[-2]
    return re.sub(r"\.json.*$", "", Path(name).name)


def reduce_members(
    members: list[str],
    url: str = MIRAGE_URL,
    zip_path: Path | None = None,
    max_packets: int = 20,
    min_packets: int = 5,
    max_flows_per_member: int = 4000,
    udp_only: bool = True,
) -> dict[str, np.ndarray]:
    """Parse a shard of members into compact arrays.

    The archive stores one JSON per capture, keyed by biflow identifier, and
    each record carries a `packet_data` block with per-packet `L4_payload_bytes`
    and `iat` series (alongside the raw payload bytes, which are what make the
    archive large and which are discarded here). The application name is the
    directory the member sits in - the corpus covers all five competition
    applications plus a dozen other real-time-communication apps.

    Returns lengths (n, max_packets), gaps (n, max_packets), a validity mask,
    per-flow transport flag and integer codes for the application label.
    """
    zf = zipfile.ZipFile(zip_path) if zip_path else open_remote(url)
    lengths, gaps, masks, labels, is_udp = [], [], [], [], []

    for name in members:
        if not name.lower().endswith(".json"):
            continue
        try:
            obj = json.loads(zf.read(name))
        except Exception:
            continue
        label = _label_from_name(name)
        n_taken = 0
        flows = obj.values() if isinstance(obj, dict) else obj
        for flow in flows:
            if n_taken >= max_flows_per_member:
                break
            arrs = _extract_flow(flow, max_packets, min_packets)
            if arrs is None:
                continue
            L, G, M, udp = arrs
            if udp_only and not udp:
                continue
            lengths.append(L)
            gaps.append(G)
            masks.append(M)
            labels.append(label)
            is_udp.append(udp)
            n_taken += 1

    if not lengths:
        return {}
    uniq = sorted(set(labels))
    code = {v: i for i, v in enumerate(uniq)}
    return {
        "lengths": np.asarray(lengths, dtype=np.float32),
        "gaps": np.asarray(gaps, dtype=np.float32),
        "mask": np.asarray(masks, dtype=bool),
        "is_udp": np.asarray(is_udp, dtype=bool),
        "label_code": np.asarray([code[v] for v in labels], dtype=np.int32),
        "label_names": np.asarray(uniq, dtype=object),
    }


def _extract_flow(flow, max_packets: int, min_packets: int):
    if not isinstance(flow, dict):
        return None
    container = flow.get("packet_data") if isinstance(flow.get("packet_data"), dict) else None
    if container is None:
        for key in ("packet_series", "packets"):
            if isinstance(flow.get(key), dict):
                container = flow[key]
                break
    if container is None:
        return None
    len_key = _find_key(container, LEN_KEYS)
    iat_key = _find_key(container, IAT_KEYS)
    if len_key is None or iat_key is None:
        return None
    L = np.asarray(container[len_key], dtype=np.float64).ravel()
    t = np.asarray(container[iat_key], dtype=np.float64).ravel()
    n = min(len(L), len(t))
    if n < min_packets:
        return None
    L, t = np.abs(L[:n]), t[:n]

    # An 8-byte transport header means UDP; the competition data is UDP media,
    # so keeping the transports aligned matters more than corpus size.
    udp = True
    hdr = container.get("L4_header_bytes")
    if isinstance(hdr, list) and hdr:
        h = np.asarray(hdr[:n], dtype=np.float64)
        udp = bool(np.median(h) == 8)

    # `iat` is already a per-packet gap series in this corpus (first entry 0),
    # but a monotone increasing series would be absolute timestamps instead.
    if np.all(np.diff(t) >= 0) and t[0] > 1e6:
        gaps = np.diff(t, prepend=t[0])
    else:
        gaps = t.copy()
    gaps = np.abs(gaps)
    gaps[0] = 0.0

    # Media flows only: drop records whose payloads are all empty (pure
    # handshake or signalling packets carry no L4 payload).
    if np.count_nonzero(L) < min_packets:
        return None

    take = min(n, max_packets)
    out_L = np.zeros(max_packets, dtype=np.float32)
    out_G = np.zeros(max_packets, dtype=np.float32)
    mask = np.zeros(max_packets, dtype=bool)
    out_L[:take] = L[:take]
    out_G[:take] = gaps[:take]
    mask[:take] = True
    return out_L, out_G, mask, udp


def save_shard(arrays: dict[str, np.ndarray], shard_id: int) -> str:
    REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    path = REDUCED_DIR / f"shard_{shard_id:04d}.npz"
    np.savez_compressed(path, **{k: v for k, v in arrays.items() if k != "label_names"},
                        label_names=np.asarray(arrays.get("label_names", []), dtype="U64"))
    return str(path)


def load_shards(limit: int | None = None) -> dict[str, np.ndarray]:
    files = sorted(REDUCED_DIR.glob("shard_*.npz"))[:limit]
    if not files:
        raise FileNotFoundError(f"no reduced shards in {REDUCED_DIR}")
    L, G, M, names, udp = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        L.append(z["lengths"])
        G.append(z["gaps"])
        M.append(z["mask"])
        udp.append(
            z["is_udp"] if "is_udp" in z else np.ones(len(z["lengths"]), dtype=bool)
        )
        local_names = z["label_names"]
        names.append(local_names[z["label_code"]])
    all_names = np.concatenate(names)
    uniq = sorted(set(all_names.tolist()))
    code = {v: i for i, v in enumerate(uniq)}
    return {
        "lengths": np.concatenate(L),
        "gaps": np.concatenate(G),
        "mask": np.concatenate(M),
        "is_udp": np.concatenate(udp),
        "label_code": np.asarray([code[v] for v in all_names], dtype=np.int64),
        "label_names": np.asarray(uniq, dtype=object),
    }
