#!/usr/bin/env python3
"""Cross-user, flock-protected server slots for slurm_evaluate.py.

One lease covers a CARLA batch (its routes run sequentially). Jobs connect to
their reserved backend directly, so the LB cannot break the configured ratio.
All launchers sharing a pool must use this protocol and the same state_dir.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import signal
import urllib.request
import uuid

SERVER_STATE_ROOT = os.environ.get("F2D_SERVER_STATE_ROOT", "results_shome/logs")
DEFAULT_SERVER_INFO_FILES = {
    "vlm": os.path.join(SERVER_STATE_ROOT, "alpamayo_server/active_server.json"),
    "depth_ttc": os.path.join(SERVER_STATE_ROOT, "depth_ttc_server/active_server.json"),
}


def normalize_gpu_type(value):
    if value == "4090":
        return "nvidia_geforce_rtx_4090"
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError("GPU type must be 'any', '4090', or a Slurm GPU type name")
    return value


def gpu_gres(value):
    value = normalize_gpu_type(value)
    return "gpu:1" if value == "any" else f"gpu:{value}:1"


class ServerSlots:
    SERVER_SPARE = 2  # Shared server cap = running route jobs + this spare count.
    HEARTBEAT_INTERVAL = 30
    HEARTBEAT_TIMEOUT = 120
    SERVER_STARTUP_TIMEOUT = 900
    UNHEALTHY_GRACE = 120
    ROUTE_WAIT_GRACE = 300
    TERMINAL_STATES = {"ended", "failed", "cancelled", "completed", "timeout",
                       "out_of_memory", "node_fail", "boot_fail", "preempted", "completing"}
    def __init__(self, state_dir, server_type, routes_per_server=None,
                 server_gpu_type="nvidia_geforce_rtx_4090", auto_server=True):
        self.root = Path(state_dir).resolve()
        self.server_type = server_type
        self.ratio = routes_per_server
        self.server_gpu_type = normalize_gpu_type(server_gpu_type)
        self.auto_server = auto_server
        if self.ratio is not None and self.ratio < 1:
            raise ValueError("F2D_ROUTES_PER_SERVER must be a positive integer")
        self.path = self.root / "route_slots.json"

    @contextmanager
    def locked(self):
        try:
            self.root.mkdir(parents=True, mode=0o2775)
        except FileExistsError:
            pass
        else:
            self.root.chmod(0o2775)
        # Never unlink this inode, including during stop/cleanup.
        try:
            fd = os.open(self.root / "route_slots.flock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o664)
        except FileExistsError:
            fd = os.open(self.root / "route_slots.flock", os.O_RDWR)
        else:
            os.fchmod(fd, 0o664)  # independent of the submitter's umask
        with os.fdopen(fd, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())  # corruption fails closed
            else:
                data = {"version": 1, "server_type": self.server_type,
                        "routes_per_server": self.ratio or 1,
                        "leases": {}, "managed_servers": {}}
            if data["server_type"] != self.server_type:
                raise RuntimeError("VLM and Depth-TTC must use separate state directories")
            yield data

    def save(self, data):
        fd, name = tempfile.mkstemp(prefix=".route_slots.", dir=self.root)
        try:
            with os.fdopen(fd, "w") as f:
                os.fchmod(f.fileno(), 0o664)
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def queue():
        # PrivateData=jobs hides other users even with --all. This snapshot is
        # authoritative ONLY for jobs owned by our own Unix UID.
        env = {k: v for k, v in os.environ.items() if not k.startswith("SQUEUE_")}
        out = subprocess.check_output(
            ["squeue", "--me", "--all", "--states=all", "--noheader", "--format", "%i|%T|%k|%M|%r"],
            text=True, env=env, timeout=15)
        jobs = {}
        for line in out.splitlines():
            fields = line.strip().split("|", 4)
            jid, state, comment = fields[:3]
            elapsed = None
            if len(fields) >= 4:
                duration = fields[3].strip()
                days, _, clock = duration.rpartition("-")
                try:
                    parts = [int(p) for p in clock.split(":")]
                    elapsed = int(days or 0) * 86400
                    for scale, value in zip((1, 60, 3600), reversed(parts)):
                        elapsed += scale * value
                except ValueError:
                    elapsed = None
            jobs[jid.strip()] = {"state": state.strip(), "comment": comment.strip(),
                                "elapsed_sec": elapsed, "reason": fields[4].strip() if len(fields) > 4 else ""}
        return jobs

    def reconcile(self, data, jobs):
        comments = {j["comment"]: jid for jid, j in jobs.items()}
        for token, lease in list(data["leases"].items()):
            if lease.get("owner_uid") != os.getuid():
                # A stale foreign heartbeat is suspicion, never permission to
                # reassign its slot. Only its owner can confirm job completion.
                if lease.get("heartbeat_at") and time.time() - lease["heartbeat_at"] > self.HEARTBEAT_TIMEOUT:
                    lease["state"] = "suspect"
                continue
            jid = lease.get("route_job_id") or comments.get("f2d-slot-" + token)
            if jid:
                lease["route_job_id"] = jid
                # --states=all also returns finished jobs retained by Slurm.
                # COMPLETING still runs cleanup, so keep its reservation until
                # it reaches a final state or disappears from the queue.
                if (jid not in jobs or
                        jobs[jid]["state"].lower() in self.TERMINAL_STATES - {"completing"}):
                    del data["leases"][token]
                else:
                    # Publish our authoritative queue state for other users,
                    # whose Slurm queries cannot see these jobs.
                    lease["queue_state"] = jobs[jid]["state"].lower()
                    if jobs[jid]["state"] == "RUNNING" and lease.get("held"):
                        lease["held"] = False
                        lease.setdefault("release_requested_at", time.time())
            elif time.time() - lease["created_at"] > 600:
                # Crash before sbatch: keep the reservation briefly. If sbatch
                # succeeded before a crash, its comment above recovers the job.
                del data["leases"][token]
        for jid, server in data["managed_servers"].items():
            if server.get("owner_uid") == os.getuid():
                server["state"] = jobs[jid]["state"].lower() if jid in jobs else "ended"

    def maintain(self, repair=False):
        with self.locked() as data:
            jobs = self.queue()
            self.reconcile(data, jobs)
            cancelled = self.repair(data, jobs) if repair else set()
            self.save(data)
            return cancelled

    def server_file(self, jid):
        if not str(jid).isdigit():
            raise ValueError("Expected numeric Slurm job ID")
        return self.root / "server_jobs" / (str(jid) + ".json")

    def server_status(self, jid):
        try:
            return json.loads(self.server_file(jid).read_text())
        except FileNotFoundError:
            return {}

    def server_ended(self, data, jid):
        server = data["managed_servers"].get(jid, {})
        return bool(server.get("state") in self.TERMINAL_STATES or server.get("cancel_requested_at") or
                    self.server_status(jid).get("state") == "ended")

    @staticmethod
    def control(command):
        try:
            return subprocess.run(command, stdout=sys.stderr, timeout=15).returncode == 0
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"[Slots] Control deferred ({' '.join(command)}): {exc}", file=sys.stderr)
            return False

    def match_held_routes(self, data, workers, health):
        """Match fewer active routes first, FIFO on ties, under the pool lock.

        Only jobs still held by the protocol can move. Release intent is saved
        before scontrol, so a controller crash cannot make a started job movable.
        """
        waiting = [(token, lease) for token, lease in data["leases"].items()
                   if lease.get("held") and lease.get("route_job_id") and
                   lease.get("state", "pending") == "pending" and
                   not lease.get("release_requested_at") and not lease.get("cancel_requested_at")]
        movable = {token for token, _ in waiting}
        used = {}
        for token, lease in data["leases"].items():
            if token in movable:
                continue
            for w in workers:
                if (lease.get("worker_id") == w["worker_id"] or
                        (not lease.get("worker_id") and lease["server_job_id"] == str(w["slurm_job_id"]))):
                    used[w["worker_id"]] = used.get(w["worker_id"], 0) + 1
        available = [w for w in workers if health[w["worker_id"]] and
                     not self.server_ended(data, str(w["slurm_job_id"]))]
        route_running = {}
        for lease in data["leases"].values():
            uid = lease.get("owner_uid")
            route_running.setdefault(uid, 0)
            if (lease.get("queue_state", lease.get("state")) == "running" or
                    lease.get("release_requested_at") or not lease.get("held")):
                route_running[uid] += 1
        while waiting:
            item = min(waiting, key=lambda item: (
                route_running[item[1].get("owner_uid")], item[1]["created_at"], item[0]))
            waiting.remove(item)
            _, lease = item
            lease.setdefault("provision_server_job_id", lease["server_job_id"])
            # Job ID 0 means no ready binding. Keep procurement separate from
            # binding, including for old readers of this shared ledger.
            lease.update(server_job_id="0", worker_id=None)
            for w in available:
                wid = w["worker_id"]
                if used.get(wid, 0) < data["routes_per_server"]:
                    lease.update(server_job_id=str(w["slurm_job_id"]), worker_id=wid)
                    used[wid] = used.get(wid, 0) + 1
                    route_running[lease.get("owner_uid")] += 1
                    break

    def server_limit(self, data):
        running = sum(l.get("queue_state", l.get("state")) == "running"
                      for l in data["leases"].values())
        return running + self.SERVER_SPARE

    def active_server_ids(self, data):
        # A successful scancel is not confirmation that GPU cleanup has ended.
        final = self.TERMINAL_STATES - {"completing"}
        active = {jid for jid, server in data["managed_servers"].items()
                  if server.get("state", "pending") not in final}
        active.update(str(w["slurm_job_id"]) for w in self.workers()
                      if not self.server_ended(data, str(w["slurm_job_id"])))
        return active

    def preferred_server_owner(self, data):
        """Choose procurement owner globally; each controller submits as itself."""
        waiting = set()
        for lease in data["leases"].values():
            uid = lease["owner_uid"]
            if lease.get("held") and not lease.get("release_requested_at") and not lease.get("cancel_requested_at"):
                waiting.add(uid)
        if not waiting:
            return None
        counts = {uid: 0 for uid in waiting}
        running = {uid: 0 for uid in waiting}
        for jid in self.active_server_ids(data):
            uid = data["managed_servers"].get(jid, {}).get("owner_uid")
            if uid in counts:
                counts[uid] += 1
                if data["managed_servers"].get(jid, {}).get("state") == "running":
                    running[uid] += 1
        imbalance = max(running.values()) - min(running.values()) > 2
        return min(waiting, key=lambda uid: (
            running[uid] if imbalance else counts[uid], counts[uid], uid))

    def cancel_newest_nonready_server(self, data, jobs, workers, health):
        running = {jid: s for jid, s in data["managed_servers"].items()
                   if s.get("state") == "running"}
        if (len(running) < self.server_limit(data) or
                any(s.get("cancel_requested_at") for s in running.values())):
            return
        ready = {str(w["slurm_job_id"]) for w in workers if health[w["worker_id"]]}
        protected = {l["server_job_id"] for l in data["leases"].values()
                     if not l.get("held") or l.get("release_requested_at") or
                     l.get("state") in ("running", "suspect")}
        candidates = set(running) - ready - protected
        if not candidates:
            return
        # Pick globally, then let only that job's owner perform the cancellation.
        jid = max(candidates, key=lambda j: (running[j].get("created_at", 0), int(j)))
        server = running[jid]
        if server.get("owner_uid") != os.getuid() or jobs.get(jid, {}).get("state") != "RUNNING":
            return
        if self.control(["scancel", jid]):
            server.update(cancel_requested_at=time.time(), repair_reason="running servers >= running routes + spare")
            for lease in data["leases"].values():
                if lease.get("provision_server_job_id") == jid:
                    lease.pop("provision_server_job_id", None)
                if lease["server_job_id"] == jid:
                    lease.update(server_job_id="0", worker_id=None)
            print(f"[Slots] Cancelling newest non-ready own server {jid}: "
                  f"running servers {len(running)} >= running routes + {self.SERVER_SPARE}", flush=True)

    def enforce_server_limit(self, data, jobs, workers, health):
        active = self.active_server_ids(data)
        limit = self.server_limit(data)
        protected = {l["server_job_id"] for l in data["leases"].values()
                     if not l.get("held") or l.get("release_requested_at") or
                     l.get("state") in ("running", "suspect")}
        ready = {str(w["slurm_job_id"]) for w in workers if health[w["worker_id"]]}
        # All owners compute the same keep set under the shared lock. Keep
        # committed bindings, then healthy servers, then the oldest starters.
        candidates = active - protected
        ordered = sorted(candidates, key=lambda jid: (
            bool(data["managed_servers"].get(jid, {}).get("cancel_requested_at")),
            jid not in ready,
            data["managed_servers"].get(jid, {}).get("state") != "running", int(jid)))
        keep = protected | set(ordered[:max(0, limit - len(active & protected))])
        for jid in sorted(active - keep, key=int, reverse=True):
            server = data["managed_servers"].get(jid, {})
            if (server.get("owner_uid") != os.getuid() or jid not in jobs or
                    server.get("cancel_requested_at")):
                continue
            if self.control(["scancel", jid]):
                server.update(cancel_requested_at=time.time(), repair_reason="shared server limit")
                for lease in data["leases"].values():
                    # Retain held route jobs; they can wait for a pooled worker.
                    if lease.get("provision_server_job_id") == jid:
                        lease.pop("provision_server_job_id", None)
                    if lease["server_job_id"] == jid:
                        lease.update(server_job_id="0", worker_id=None)
                print(f"[Slots] Cancelling excess own server {jid}: shared limit {limit} (running routes + {self.SERVER_SPARE})", flush=True)

    def repair(self, data, jobs):
        """Act only on our own queue/UID. Foreign jobs use shared lifecycle data.

        Never release a slot just because scancel succeeded: keep it until the
        owner's next queue snapshot confirms the route actually disappeared.
        """
        now = time.time()
        workers = self.workers()
        health = {}
        for worker in workers:
            health[worker["worker_id"]] = self.healthy(worker)
        self.match_held_routes(data, workers, health)
        self.enforce_server_limit(data, jobs, workers, health)
        self.cancel_newest_nonready_server(data, jobs, workers, health)
        def ready(lease):
            return any(str(w["slurm_job_id"]) == lease["server_job_id"] and
                       (not lease.get("worker_id") or w["worker_id"] == lease["worker_id"]) and
                       health[w["worker_id"]] for w in workers)

        for jid, server in data["managed_servers"].items():
            if server.get("owner_uid") != os.getuid() or jid not in jobs or self.server_ended(data, jid):
                continue
            if jobs[jid]["state"] != "RUNNING":
                continue
            if ready({"server_job_id": jid}):
                server["ever_ready"] = True
                server.pop("unhealthy_since", None)
                continue
            server.setdefault("running_since", now)
            server.setdefault("unhealthy_since", now)
            elapsed = jobs[jid].get("elapsed_sec")
            if elapsed is None:
                elapsed = now - server["running_since"]
            failed_start = not server.get("ever_ready") and elapsed >= self.SERVER_STARTUP_TIMEOUT
            failed_health = server.get("ever_ready") and now - server["unhealthy_since"] >= self.UNHEALTHY_GRACE
            if failed_start or failed_health:
                reason = "startup deadline exceeded" if failed_start else "backend health failed persistently"
                if self.control(["scancel", jid]):
                    server.update(cancel_requested_at=now, repair_reason=reason)
                    print(f"[Slots] Cancelling own server {jid}: {reason}", flush=True)

        # No hardcoded QoS job limit: react to Slurm's actual reason. Without
        # this, idle server jobs alone could fill MaxJobs and starve every route.
        own = [l for l in data["leases"].values() if l.get("owner_uid") == os.getuid()]
        pressure = any(jobs.get(l.get("route_job_id"), {}).get("state") == "PENDING" and
                       not l.get("held") and
                       jobs[l["route_job_id"]].get("reason", "").startswith(("QOS", "Assoc")) and
                       "Job" in jobs[l["route_job_id"]].get("reason", "") for l in own)
        controller = data.setdefault("controllers", {}).setdefault(str(os.getuid()), {})
        if pressure:
            controller["server_resume_after"] = now + 300
            for jid, server in data["managed_servers"].items():
                if (server.get("owner_uid") == os.getuid() and jid in jobs and
                        jobs[jid]["state"] == "PENDING" and not server.get("scheduler_held")):
                    if self.control(["scontrol", "hold", jid]):
                        server["scheduler_held"] = True
            candidates = []
            for jid, server in data["managed_servers"].items():
                bound = [l for l in data["leases"].values() if l["server_job_id"] == jid]
                if (server.get("owner_uid") == os.getuid() and jobs.get(jid, {}).get("state") == "RUNNING" and
                        not self.server_ended(data, jid) and ready({"server_job_id": jid}) and bound and
                        all(l.get("owner_uid") == os.getuid() and
                            jobs.get(l.get("route_job_id"), {}).get("state") == "PENDING" for l in bound)):
                    candidates.append(jid)
            if candidates:
                jid = max(candidates, key=int)
                if self.control(["scancel", jid]):
                    data["managed_servers"][jid].update(cancel_requested_at=now, repair_reason="QoS job slots needed by routes")
                    print(f"[Slots] Cancelling idle own server {jid}: free a QoS slot for routes", flush=True)
        elif now >= controller.get("server_resume_after", 0):
            ready_waiting = any(ready(l) and jobs.get(l.get("route_job_id"), {}).get("state") == "PENDING"
                                and not l.get("held") for l in own)
            starting = any(s.get("owner_uid") == os.getuid() and not s.get("scheduler_held") and
                           jobs.get(jid, {}).get("state") in ("PENDING", "CONFIGURING", "RUNNING") and
                           not self.server_ended(data, jid) and not ready({"server_job_id": jid})
                           for jid, s in data["managed_servers"].items())
            if not ready_waiting and not starting:
                for jid, server in data["managed_servers"].items():
                    if (server.get("owner_uid") == os.getuid() and server.get("scheduler_held") and
                            jobs.get(jid, {}).get("state") == "PENDING"):
                        if self.control(["scontrol", "release", jid]):
                            server["scheduler_held"] = False
                            print(f"[Slots] Released deferred server {jid}", flush=True)
                        break

        cancelled = set()
        for token, lease in data["leases"].items():
            jid = lease.get("route_job_id")
            if lease.get("owner_uid") != os.getuid() or jid not in jobs:
                continue
            # A previous successful scancel may still be completing in Slurm.
            if lease.get("cancel_requested_at"):
                continue
            dead = self.server_ended(data, lease["server_job_id"])
            is_ready = not dead and ready(lease)
            reason = "reserved server ended" if dead else None
            if lease.get("held") and is_ready:
                # Write ahead of release: even a timeout/crash may mean Slurm
                # already started the job. Never rebind an uncertain release.
                lease["release_requested_at"] = now
                self.save(data)
                if self.control(["scontrol", "release", jid]):
                    lease["held"] = False
                    lease.pop("waiting_since", None)
                    print(f"[Slots] Released route {jid}: server {lease['server_job_id']} ready", flush=True)
                continue
            if lease.get("held") and lease["server_job_id"] == "0":
                # Procurement failures do not require cancelling a held route.
                # The capped pool replenisher below will replace failed servers.
                provision = lease.get("provision_server_job_id")
                if provision and self.server_ended(data, provision):
                    lease.pop("provision_server_job_id", None)
            if not is_ready and jobs[jid]["state"] == "RUNNING" and not reason:
                # Existing unheld jobs must not consume GPUs indefinitely while
                # their backend is pending, dead, or unreachable.
                lease.setdefault("waiting_since", now)
                if now - lease["waiting_since"] >= self.ROUTE_WAIT_GRACE:
                    reason = "waiting for backend without progress"
            elif is_ready:
                lease.pop("waiting_since", None)
            if reason and jobs[jid]["state"].lower() not in self.TERMINAL_STATES:
                if self.control(["scancel", jid]):
                    lease.update(cancel_requested_at=now, repair_reason=reason)
                    cancelled.add(jid)
                    print(f"[Slots] Cancelling own route {jid}: {reason}; scheduler will retry unfinished routes", flush=True)
        # Held routes can outnumber servers. Replenish a failed pool without
        # cancelling/resubmitting those route jobs just to procure a backend.
        waiting = sum(l.get("held", False) and l["server_job_id"] == "0"
                      for l in data["leases"].values())
        active = self.active_server_ids(data)
        desired = min(self.server_limit(data), (len(data["leases"]) + data["routes_per_server"] - 1)
                      // data["routes_per_server"])
        if (self.auto_server and waiting and len(active) < desired and
                self.preferred_server_owner(data) == os.getuid()):
            self.submit_server(data)
        return cancelled

    def workers(self, jobs=None):
        found = []
        for p in sorted((self.root / "workers").glob("worker_*.json")):
            try:
                w = json.loads(p.read_text())
                if w["server_type"] != self.server_type:
                    continue
                # The launcher refreshes registration mtime while its worker is
                # alive. Discovery must not depend on another user's queue.
                if time.time() - p.stat().st_mtime > self.HEARTBEAT_TIMEOUT:
                    continue
                if self.server_status(str(w["slurm_job_id"])).get("state") == "ended":
                    continue
                found.append(w)
            except (OSError, ValueError, KeyError):
                continue
        return found

    @staticmethod
    def healthy(worker):
        try:
            # Compute-node HTTP must bypass login/node proxy settings.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(worker["base_url"].rstrip("/") + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def available(self, data, jobs, exclude_job=None):
        workers = self.workers(jobs)
        leases = list(data["leases"].values())
        for w in workers:
            jid, wid = str(w["slurm_job_id"]), w["worker_id"]
            used = sum(l.get("worker_id") == wid or
                       (not l.get("worker_id") and l["server_job_id"] == jid)
                       for l in leases)
            if (jid != exclude_job and not self.server_ended(data, jid) and
                    used < data["routes_per_server"] and self.healthy(w)):
                return {"server_job_id": jid, "worker_id": wid}
        # Our auto-submitted jobs each contain exactly one GPU. Their remaining
        # slots can be reserved even before their registration file appears.
        registered = {str(w["slurm_job_id"]) for w in workers}
        for jid, server in data["managed_servers"].items():
            if jid == exclude_job or jid in registered or self.server_ended(data, jid):
                continue
            status = self.server_status(jid)
            if status.get("heartbeat_at") and time.time() - status["heartbeat_at"] > self.HEARTBEAT_TIMEOUT:
                continue
            if server.get("state", "pending") not in ("pending", "configuring", "running"):
                continue
            used = sum(l["server_job_id"] == jid for l in leases)
            used += sum(l.get("provision_server_job_id") == jid and l["server_job_id"] != jid for l in leases)
            if used < data["routes_per_server"]:
                return {"server_job_id": jid, "worker_id": None}
        return None

    def submit_server(self, data):
        # The Bash launcher remains Python-free on login nodes. Each invocation
        # has private script paths, so simultaneous users cannot overwrite them.
        fd, name = tempfile.mkstemp(prefix=".submitted_server.", dir=self.root)
        os.close(fd)
        try:
            paused = time.time() < data.get("controllers", {}).get(str(os.getuid()), {}).get("server_resume_after", 0)
            subprocess.run(
                ["bash", str(Path(__file__).with_name("f2d_ctl.sh")), "start-server",
                 "--type", self.server_type, "--gpus", "1", "--gpus-per-job", "1",
                 "--state-dir", str(self.root), "--no-lb", "--job-id-file", name,
                 "--server-gpu-type", self.server_gpu_type, *(["--hold"] if paused else [])],
                check=True, stdout=sys.stderr)
            jid = Path(name).read_text().strip()
            if not jid.isdigit():
                raise RuntimeError("Server launcher did not return exactly one Slurm job ID")
            data["managed_servers"][jid] = {"owner_uid": os.getuid(), "state": "pending",
                                             "created_at": time.time(), "scheduler_held": paused}
            self.save(data)
            return jid
        finally:
            os.unlink(name)

    def cancel_new_server(self, data, jid):
        # Only a newly submitted, unleased server owned by this process's user.
        if data["managed_servers"].get(jid, {}).get("owner_uid") != os.getuid():
            return
        if any(l["server_job_id"] == jid for l in data["leases"].values()):
            return
        if self.control(["scancel", jid]):
            data["managed_servers"].pop(jid, None)
        self.save(data)

    def submit(self, job_file, auto_server=True):
        """Submit backup server, reserve a slot, then submit its route job.

        Hold the pool lock through the transaction, including Slurm submission.
        Thus no other user can claim the new server before we cancel/reuse it.
        """
        with self.locked() as data:
            jobs = self.queue()
            self.reconcile(data, jobs)
            if self.ratio is not None and data["routes_per_server"] != self.ratio:
                if data["leases"]:
                    raise RuntimeError("Pool ratio differs from F2D_ROUTES_PER_SERVER; drain leases before changing it")
                data["routes_per_server"] = self.ratio
            self.save(data)
            preferred = self.preferred_server_owner(data)
            fresh = (self.submit_server(data) if auto_server and
                     len(self.active_server_ids(data)) < self.server_limit(data) and
                     preferred in (None, os.getuid()) else None)
            # Refresh after sbatch; use the previous snapshot only for cleanup.
            try:
                target = self.available(data, self.queue(), exclude_job=fresh)
            except Exception:
                if fresh:
                    self.cancel_new_server(data, fresh)
                raise
            if target is None:
                if fresh is None and not auto_server:
                    raise RuntimeError("No free server slot (--no-auto-server)")
                target = {"server_job_id": fresh or "0", "worker_id": None}
            elif fresh:
                self.cancel_new_server(data, fresh)
            token = uuid.uuid4().hex
            data["leases"][token] = dict(target, created_at=time.time(), owner_uid=os.getuid(),
                                         state="pending", held=True)
            self.save(data)
            try:
                out = subprocess.check_output(
                    ["sbatch", "--parsable", "--hold", "--comment=f2d-slot-" + token,
                     "--export=ALL,F2D_SERVER_SLOT=" + token, str(job_file)], text=True)
                jid = out.strip().split(";", 1)[0]
                if not jid.isdigit():
                    raise RuntimeError("Unrecognized sbatch result; reservation retained for recovery")
            except subprocess.CalledProcessError:
                # A definite rejected submission has no running route.
                del data["leases"][token]
                self.save(data)
                if fresh:
                    self.cancel_new_server(data, fresh)
                raise
            data["leases"][token]["route_job_id"] = jid
            self.save(data)
            return jid

    def wait(self, token, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.locked() as data:
                lease = data["leases"].get(token)
                if not lease or lease.get("route_job_id") != os.environ.get("SLURM_JOB_ID"):
                    raise RuntimeError("Missing server reservation or route job ID mismatch")
                # Query only our own jobs. A foreign server's lifecycle is
                # communicated through the shared ledger and server status file.
                self.reconcile(data, self.queue())
                self.save(data)
                if self.server_ended(data, lease["server_job_id"]):
                    raise RuntimeError("Reserved server job has exited; refusing to use an unreserved backend")
                for w in self.workers():
                    if str(w["slurm_job_id"]) != lease["server_job_id"]:
                        continue
                    if lease["worker_id"] and w["worker_id"] != lease["worker_id"]:
                        continue
                    if self.healthy(w):
                        lease["worker_id"] = w["worker_id"]
                        self.save(data)
                        suffix = "/generate" if self.server_type == "vlm" else "/score"
                        return w["base_url"].rstrip("/") + suffix
            time.sleep(5)
        raise TimeoutError("Reserved server did not become ready before server-wait-timeout")

    def route_heartbeat(self, token):
        with self.locked() as data:
            lease = data["leases"].get(token)
            if (not lease or lease.get("route_job_id") != os.environ.get("SLURM_JOB_ID") or
                    lease.get("owner_uid") != os.getuid()):
                raise RuntimeError("Missing or foreign route reservation")
            lease.update(state="running", heartbeat_at=time.time())
            self.save(data)

    def server_heartbeat(self, ended=False):
        jid = os.environ["SLURM_JOB_ID"]
        with self.locked() as data:
            path = self.server_file(jid)
            previous = self.server_status(jid)
            if previous and previous.get("owner_uid") != os.getuid():
                raise RuntimeError("Refusing to update another user's server status")
            if previous.get("state") == "ended" and not ended:
                return False
            try:
                path.parent.mkdir(mode=0o2775)
            except FileExistsError:
                pass
            else:
                path.parent.chmod(0o2775)
            # Atomic, group-writable status, separate from per-worker readiness.
            fd, name = tempfile.mkstemp(prefix=".server.", dir=path.parent)
            try:
                with os.fdopen(fd, "w") as f:
                    os.fchmod(f.fileno(), 0o664)
                    json.dump(dict(job_id=jid, owner_uid=os.getuid(), server_type=self.server_type,
                                   state="ended" if ended else "running", heartbeat_at=time.time()), f)
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            return True

    def watch(self, token, parent_pid, server=False):
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                return
            if server:
                if not self.server_heartbeat():
                    return
            else:
                self.route_heartbeat(token)
                # Also reap our own jobs while a route is waiting/running.
                # A queue error retains all existing reservations.
                try:
                    self.maintain()
                except (subprocess.SubprocessError, OSError) as exc:
                    print(f"[Slots] Owner reconciliation deferred: {exc}", file=sys.stderr)
            time.sleep(self.HEARTBEAT_INTERVAL)

    def release(self, token):
        with self.locked() as data:
            lease = data["leases"].get(token)
            if lease:
                if (lease.get("route_job_id") != os.environ.get("SLURM_JOB_ID") or
                        lease.get("owner_uid") != os.getuid()):
                    raise RuntimeError("Refusing to release another route job's slot")
                del data["leases"][token]
                self.save(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("wait", "release", "route-watch", "server-watch", "server-end", "reconcile"))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--server-type", choices=("vlm", "depth_ttc"), required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    token = os.environ.get("F2D_SERVER_SLOT")
    if args.action in ("wait", "release", "route-watch") and not token:
        parser.error("F2D_SERVER_SLOT is required; launch through slurm_evaluate.py")
    pool = ServerSlots(args.state_dir, args.server_type)
    if args.action == "wait":
        print(pool.wait(token, args.timeout))
    elif args.action == "release":
        pool.release(token)
    elif args.action == "reconcile":
        pool.maintain()
    elif args.action == "server-end":
        pool.server_heartbeat(ended=True)
    else:
        if not args.parent_pid or args.parent_pid <= 1:
            parser.error("--parent-pid must identify the launcher process")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        pool.watch(token, args.parent_pid, server=args.action == "server-watch")


if __name__ == "__main__":
    main()
