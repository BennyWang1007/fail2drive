#!/usr/bin/env python3
"""Shared helpers for the pooled Fail2Drive evaluator
(scheduler: run_pooled_evals.py, worker: f2d_worker.py, servers: submit_model_servers.py)."""
import json
import os
import random
import socket
import time
import urllib.request
from pathlib import Path

# Same rule set as slurm_evaluate.py
RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}

# eval_mode -> request path appended to the server base url
SERVER_PATH = {"vlm": "/generate", "depth_ttc": "/score"}


def retry_reason(data):
    """None -> result is final. str -> why the route must be (re)run. (copy of slurm_evaluate._retry_reason)"""
    if not isinstance(data, dict):
        return "json top level is not a dict"
    cp = data.get("_checkpoint")
    if not isinstance(cp, dict):
        return "no _checkpoint"
    progress, records = cp.get("progress"), cp.get("records")
    if not isinstance(progress, list) or not isinstance(records, list):
        return "progress/records malformed"
    if len(progress) < 2:
        return f"progress too short {progress}"
    if progress[0] < progress[1]:
        return f"progress incomplete {progress}"
    if len(records) == 0:
        return "records empty"
    for rec in records:
        if not isinstance(rec, dict):
            return "record is not a dict"
        if rec.get("status") in RETRYABLE_STATUSES:
            return f"status {rec.get('status')!r} is retryable"
    return None


def statuses_of(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cp = json.load(f).get("_checkpoint", {})
        return [r.get("status") if isinstance(r, dict) else repr(r) for r in cp.get("records", [])]
    except Exception:
        return []


class FinalityCache:
    """is_final(path) with a (mtime,size) signature cache -> unchanged files are never re-parsed."""

    def __init__(self):
        self._c = {}

    def is_final(self, path) -> bool:
        path = str(path)
        try:
            st = os.stat(path)
        except OSError:
            return False
        sig = (st.st_mtime_ns, st.st_size)
        hit = self._c.get(path)
        if hit and hit[0] == sig:
            return hit[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                val = retry_reason(json.load(f)) is None
        except (OSError, ValueError):
            val = False
        self._c[path] = (sig, val)
        return val


# ---------------------------------------------------------------- paths
def route_id_of(xml_name: str) -> str:
    return xml_name.split("_")[-1][:-4]


def route_seed(route_id: str, seed: int) -> int:
    # Fail2Drive specific, identical to slurm_evaluate.py
    return int(route_id) % 1000 + 10000 * int(seed)


def seed_dir(task: dict) -> Path:
    return Path(task["out_root"]) / str(task["seed"])


def result_path(task: dict, rid: str) -> Path:
    return seed_dir(task) / "res" / f"{rid}_res.json"


def claim_path(state_dir, task: dict, rid: str) -> Path:
    return Path(state_dir) / "claims" / task["name"] / rid


# ---------------------------------------------------------------- model-server registry
def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def list_servers(registry_dir, kind: str):
    """Base urls (http://host:port) of all HEALTHY servers of this kind, in stable order."""
    d = Path(registry_dir) / kind
    out = []
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            base = json.loads(f.read_text())["base"]
        except Exception:
            continue
        if http_ok(base + "/health"):
            out.append(base)
    return out


class ServerWatch:
    """list_servers() with a small TTL cache."""

    def __init__(self, registry_dir, ttl: float = 15.0):
        self.registry_dir, self.ttl = registry_dir, ttl
        self._c = {}

    def get(self, kind: str):
        t, val = self._c.get(kind, (0.0, []))
        if time.time() - t > self.ttl:
            val = list_servers(self.registry_dir, kind)
            self._c[kind] = (time.time(), val)
        return val


# ---------------------------------------------------------------- misc
def free_port(lo: int, hi: int, span: int = 1, tries: int = 500) -> int:
    """Random port p in [lo,hi) such that p..p+span-1 can all be bound right now on this node."""
    for _ in range(tries):
        p = random.randint(lo, hi - span)
        socks = []
        try:
            for k in range(span):
                s = socket.socket()
                socks.append(s)
                s.bind(("0.0.0.0", p + k))
            return p
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError(f"no free port in [{lo},{hi})")


def parse_hms(s: str) -> int:
    """Slurm time string -> seconds (mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm, d-hh:mm:ss)."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        parts = [int(x) for x in s.split(":")]
        h, m, sec = (parts + [0, 0])[:3]
    else:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        else:
            h, m, sec = parts[:3]
    return days * 86400 + h * 3600 + m * 60 + sec
