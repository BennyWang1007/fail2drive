"""Depth-TTC server client for selecting among activation-steered trajectories."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
import time
import threading
import copy
from typing import Optional
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


@dataclass
class DepthTTCDecision:
    frame_id: int
    action: str = "none"
    selected_trajectory: str = "original"
    scores: dict | None = None
    reason: str = ""
    raw_response: str = ""
    timestamp: float = 0.0
    error: str = ""
    elapsed_ms: float = 0.0


class DepthTTCClient:
    """Synchronous HTTP client matching the Alpamayo server request style."""

    ACTION_TO_TRAJECTORY = {
        "none": "original",
        "original": "original",
        "brake": "brake",
        "left": "left",
        "right": "right",
    }

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8766/score",
        timeout_s: float = 30.0,
        horizon_s: float = 2.0,
        verbose: bool = False,
    ):
        self.server_url = server_url
        self.timeout_s = float(timeout_s)
        self.horizon_s = float(horizon_s)
        self.verbose = bool(verbose)

    @classmethod
    def from_env(cls) -> "DepthTTCClient":
        return cls(
            server_url=os.environ.get(
                "DEPTH_TTC_SERVER_URL", "http://127.0.0.1:8766/score"),
            timeout_s=float(os.environ.get("DEPTH_TTC_TIMEOUT_S", 30.0)),
            horizon_s=float(os.environ.get("DEPTH_TTC_HORIZON_S", 2.0)),
            verbose=str(os.environ.get("DEPTH_TTC_VERBOSE", "0")
                        ).lower() in ("1", "true", "yes", "y"),
        )

    def score(
        self,
        frame_id: int,
        rgb_image: np.ndarray,
        speed: float,
        trajectories: Optional[dict[str, list]] = None,
        camera: Optional[dict] = None,
        ego_extent: Optional[dict] = None,
        trajectory_times: Optional[dict[str, list]] = None,
        planner_candidates: Optional[dict] = None,
        radar: Optional[list | np.ndarray] = None,
    ) -> DepthTTCDecision:

        # 處理 radar 格式確保可轉為 JSON
        radar_payload = None
        # if radar is not None:
        #     if hasattr(radar, "tolist"):
        #         radar_payload = radar.tolist()
        #     else:
        #         radar_payload = radar

        payload = {
            "frame_id": int(frame_id),
            "image_base64": self._image_to_base64_png(rgb_image),
            "trajectories": trajectories or {},
            "trajectory_times": trajectory_times or {},
            "planner_candidates": planner_candidates or {},
            "speed": float(speed),
            "horizon_s": self.horizon_s,
            "camera": camera or {},
            "ego_extent": ego_extent or {},
            "radar": radar_payload,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.server_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST")

        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            return self._fallback(frame_id, f"Depth TTC server HTTP {exc.code}: {error_body}")
        except Exception as exc:  # pylint: disable=broad-except
            return self._fallback(frame_id, f"{type(exc).__name__}: {exc}")

        selected = str(response_data.get("selected_trajectory",
                       response_data.get("action", "original"))).lower()
        action = str(response_data.get("action", selected)).lower()
        selected = self.ACTION_TO_TRAJECTORY.get(selected, "original")
        if action == "original":
            action = "none"
        if action not in ("none", "brake", "left", "right"):
            action = "none" if selected == "original" else selected
        decision = DepthTTCDecision(
            frame_id=int(response_data.get("frame_id", frame_id)),
            action=action,
            selected_trajectory=selected,
            scores=response_data.get("scores", {}),
            reason=str(response_data.get("reason", "")),
            raw_response=json.dumps(response_data),
            timestamp=time.time(),
        )
        if self.verbose:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            print(
                f"[DepthTTC] frame={frame_id} selected={decision.selected_trajectory} "
                f"action={decision.action} elapsed_ms={elapsed_ms:.1f} reason={decision.reason}",
                flush=True)
        return decision

    def _fallback(self, frame_id: int, error: str) -> DepthTTCDecision:
        if self.verbose:
            print(
                f"[DepthTTC] fallback original frame={frame_id} error={error}", flush=True)
        return DepthTTCDecision(
            frame_id=int(frame_id),
            action="none",
            selected_trajectory="original",
            scores={},
            reason="Depth TTC scoring failed; using original trajectory",
            timestamp=time.time(),
            error=error,
        )

    @staticmethod
    def _image_to_base64_png(image: np.ndarray) -> str:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


class AsyncDepthTTCClient(DepthTTCClient):
    """One background HTTP request per agent; planner threads only submit/poll."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._latest = None
        self._thread = None
        if self.verbose:
            print(f"[DepthTTCAsync] enabled url={self.server_url} max_inflight=1", flush=True)

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def latest(self):
        with self._lock:
            return self._latest

    def submit(self, **payload):
        with self._lock:
            if self._busy or self._closed:
                return False
            self._busy = True
        try:
            # Never hand mutable sensor buffers to a background worker.
            payload = copy.deepcopy(payload)
            self._thread = threading.Thread(
                target=self._run_request, args=(payload,),
                name="AsyncDepthTTC", daemon=True)
            self._thread.start()
        except Exception:
            with self._lock:
                self._busy = False
            raise
        return True

    def _run_request(self, payload):
        started = time.perf_counter()
        try:
            decision = self.score(**payload)
        except Exception as exc:
            decision = self._fallback(payload['frame_id'], f"{type(exc).__name__}: {exc}")
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            if not self._closed:
                self._latest = decision
            self._busy = False
        if self.verbose:
            print(f"[DepthTTCAsync] result decision_frame={decision.frame_id} "
                  f"elapsed_ms={decision.elapsed_ms:.1f} "
                  f"selected={decision.selected_trajectory} error={decision.error}", flush=True)

    def stop(self):
        with self._lock:
            self._closed = True
            self._latest = None
        if self._thread is not None:
            self._thread.join(timeout=0.2)
