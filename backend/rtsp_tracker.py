#!/usr/bin/env python3
"""
High-performance RTSP person tracker with Event Management System.
YOLOv8 Nano + ByteTrack + Polygon ROI Dwell Detection + Async Event Pipeline.

Architecture
────────────
  FrameReader (daemon thread)
      └── queue.Queue(maxsize=1)  ← always holds the LATEST frame only
  FrameBuffer
      └── deque(maxlen=60)        ← rolling window of raw frames for snapshots
  Main thread
      ├── PersonDetector  (YOLOv8 Nano)
      ├── PersonTracker   (ByteTrack)
      ├── EntityRegistry  (per-ID state: dwell frames, cooldown, ROI status)
      └── EventManager    (asyncio loop thread — JPEG extraction + dispatch)

Event lifecycle
───────────────
  1. A TrackedEntity's centroid enters the PolygonROI.
  2. consecutive_roi_frames increments each frame.
  3. At frame 45 (configurable) and cooldown elapsed → Event fires.
  4. FrameBuffer snapshot is taken synchronously (O(n) list copy).
  5. An asyncio coroutine is scheduled on the EventManager's dedicated loop
     thread; CPU-bound JPEG encoding runs in a ThreadPoolExecutor.
  6. Event is dispatched to the user-supplied callback (default: logger).
  7. Per-ID 15 s cooldown blocks re-triggering.

Usage
─────
  python rtsp_tracker.py rtsp://user:pass@192.168.1.100:554/stream1
  python rtsp_tracker.py rtsp://... --device cuda --imgsz 416 --conf 0.45
  python rtsp_tracker.py rtsp://... \\
      --roi '[[100,200],[800,200],[800,600],[100,600]]' \\
      --dwell-frames 45 --cooldown 15 --display

Dependencies
────────────
  pip install ultralytics opencv-python-headless
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import cv2
import httpx
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rtsp_tracker")


# ─── Configuration dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class StreamConfig:
    """RTSP stream and reconnection parameters."""

    rtsp_url: str
    reconnect_delay: float = 2.0     # base back-off interval (seconds)
    max_reconnect_attempts: int = 0  # 0 = retry forever
    capture_backend: int = cv2.CAP_FFMPEG


@dataclass(frozen=True)
class DetectorConfig:
    """YOLO11 pose inference parameters (TensorRT on Jetson Orin Nano)."""

    model_path: str = "yolov11s-pose.engine"
    confidence: float = 0.40
    iou: float = 0.45
    device: str = "cuda:0"    # TensorRT engines are CUDA-only
    imgsz: int = 1280         # MUST match the engine's build-time input size
    person_class_id: int = 0  # COCO class 0 = person


@dataclass(frozen=True)
class TrackerConfig:
    """ByteTrack hyper-parameters."""

    track_high_thresh: float = 0.50
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.60
    track_buffer: int = 30   # frames to keep a lost track alive
    match_thresh: float = 0.80
    frame_rate: int = 30


@dataclass(frozen=True)
class EventConfig:
    """
    Dwell detection and event pipeline parameters.

    The ROI polygon and PPE ``active_rules`` are no longer defined here —
    they are fetched per-camera from Supabase by ``ConfigManager`` and hot-
    reloaded while the pipeline runs.
    """

    dwell_frames: int = 45           # consecutive in-ROI frames before trigger
    cooldown_seconds: float = 15.0   # per-ID minimum gap between events
    buffer_maxlen: int = 60          # rolling frame buffer depth
    snapshot_count: int = 10         # evenly-spaced frames extracted per event
    jpeg_quality: int = 60           # 0–100; lower = smaller in-memory streams
    entity_ttl_seconds: float = 30.0 # prune entities unseen for this long


# ─── Internal data types ─────────────────────────────────────────────────────


@dataclass
class Detections:
    """
    Container for per-frame person detections that satisfies the full
    attribute-and-subscript contract required by ``BYTETracker.update()``.

    Attribute contract
    ──────────────────
    ``BYTETracker`` reads four named attributes on whatever object is passed
    as its first argument:

        results.xyxy   → (N, 4) float32  [x1, y1, x2, y2]
        results.xywh   → (N, 4) float32  [cx, cy, w, h]   used by init_track
        results.conf   → (N,)   float32
        results.cls    → (N,)   float32

    Subscript contract
    ──────────────────
    Internally, ``BYTETracker`` performs index-based sub-selection on the
    detection object, e.g.::

        remain_inds  = scores > self.track_thresh        # boolean mask
        detections   = results[remain_inds]              # calls __getitem__
        u_detection  = results[u_detection_ind]          # integer-array index

    ``__getitem__`` therefore applies the provided index (boolean mask,
    integer array/list, or slice) uniformly to all three internal arrays
    and returns a new ``Detections`` instance — keeping our dataclass as the
    single representation throughout the pipeline instead of converting to
    a raw ``(N, 6)`` array.
    """

    xyxy: np.ndarray  # (N, 4)  absolute pixel coords [x1 y1 x2 y2]
    conf: np.ndarray  # (N,)    confidence in [0, 1]
    cls:  np.ndarray  # (N,)    integer class id

    # ── Pose extension ────────────────────────────────────────────────────────
    # Shape (N, 17, 3) — the 17 COCO skeleton keypoints per detection.
    # Each row is [x_px, y_px, confidence].  The field defaults to an empty
    # (0, 17, 3) sentinel so every caller that existed before pose support was
    # added continues to work without modification: BYTETracker only reads
    # .xyxy / .xywh / .conf / .cls and never touches this field.
    keypoints: np.ndarray = field(
        default_factory=lambda: np.empty((0, 17, 3), dtype=np.float32)
    )

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            xyxy=np.empty((0, 4),    dtype=np.float32),
            conf=np.empty((0,),      dtype=np.float32),
            cls =np.empty((0,),      dtype=np.float32),
            keypoints=np.empty((0, 17, 3), dtype=np.float32),
        )

    @property
    def xywh(self) -> np.ndarray:
        """
        Return bounding boxes in ``[cx, cy, w, h]`` format as a float32
        array of shape ``(N, 4)``.

        ``BYTETracker.init_track()`` passes this format directly into the
        Kalman filter's state vector, so the property must exist on whatever
        object is handed to ``BYTETracker.update()``.

        Derived on-the-fly from ``self.xyxy`` — no extra storage is needed::

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w  =  x2 - x1
            h  =  y2 - y1
        """
        x1, y1, x2, y2 = (self.xyxy[:, i] for i in range(4))
        return np.stack(
            [
                (x1 + x2) / 2.0,  # cx
                (y1 + y2) / 2.0,  # cy
                x2 - x1,          # w
                y2 - y1,          # h
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: "np.ndarray | list[int] | slice") -> "Detections":
        """
        Return a new ``Detections`` containing only the rows selected by
        *index*.

        NumPy advanced indexing handles all three flavours transparently:

        * **Boolean mask** (shape ``(N,)``): selects rows where the mask is
          ``True``.  Applied to ``xyxy`` (2-D) this selects matching *rows*;
          applied to ``conf`` / ``cls`` (1-D) it selects matching *elements*.
          Both yield arrays of the same length K.
        * **Integer array / list** (e.g. ``[0, 3, 5]``): selects the rows at
          those positions, in order.
        * **Slice** (e.g. ``slice(0, 4)``): selects a contiguous sub-range.

        The returned instance is a *view* where possible (slice) and a *copy*
        where NumPy must materialise a new array (fancy indexing).  In either
        case the caller owns the result independently of this object.
        """
        # Guard: only index into keypoints when the array is populated.
        # The (0, 17, 3) empty sentinel produced by Detections.empty() has
        # no rows, so applying an (N,)-shaped mask to it would raise a shape
        # mismatch error.  This branch keeps BYTETracker's internal masking
        # calls safe even when pose data is absent.
        kp = self.keypoints
        if kp.shape[0] > 0:
            kp = kp[index]  # (N,17,3)[index] → (K,17,3)
        return Detections(
            xyxy=self.xyxy[index],  # (N,4)[index] → (K,4)
            conf=self.conf[index],  # (N,)[index]  → (K,)
            cls =self.cls[index],   # (N,)[index]  → (K,)
            keypoints=kp,           # (K,17,3) or empty sentinel
        )


@dataclass
class TrackedEntity:
    """
    Per-track mutable state maintained across frames.

    Only the main thread reads/writes this object; no locking needed.
    """

    track_id: int
    bbox: np.ndarray                            # current [x1, y1, x2, y2]
    centroid: tuple[float, float]               # (cx, cy) derived from bbox
    frames_visible: int = 0                     # total frames track was active
    consecutive_roi_frames: int = 0             # resets when centroid leaves ROI
    in_roi: bool = False                        # current ROI membership
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp:  float = field(default_factory=time.time)
    last_event_timestamp: float = 0.0           # 0.0 = event never fired


@dataclass
class Event:
    """
    Immutable event record produced when a dwell threshold is crossed.

    ``jpeg_streams`` is a list of ``io.BytesIO`` objects, each containing
    a single highly-compressed JPEG frame.  Streams are seeked to position 0
    before delivery so callers can ``.read()`` immediately.
    """

    track_id: int
    triggered_at: float
    dwell_frames_at_trigger: int
    jpeg_streams: list[io.BytesIO]
    # PPE rule set active for the source camera at trigger time, as fetched
    # from Supabase by ConfigManager (e.g. ["hardhat", "vest"]).  Forwarded
    # to the VLM escalation layer to scope the violation check.
    active_rules: list[str] = field(default_factory=list)

    @property
    def triggered_at_str(self) -> str:
        return datetime.fromtimestamp(self.triggered_at).strftime("%H:%M:%S.%f")[:-3]

    @property
    def total_jpeg_bytes(self) -> int:
        total = 0
        for s in self.jpeg_streams:
            pos = s.tell()
            s.seek(0, 2)
            total += s.tell()
            s.seek(pos)
        return total


# ─── Rolling FPS counter ──────────────────────────────────────────────────────


class FPSCounter:
    """Thread-safe rolling-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def tick(self) -> float:
        """Record a new frame and return the current FPS estimate."""
        now = time.perf_counter()
        with self._lock:
            self._timestamps.append(now)
            if len(self._timestamps) > self._window:
                self._timestamps.pop(0)
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Thread-safe rolling frame buffer ────────────────────────────────────────


class FrameBuffer:
    """
    Thread-safe rolling buffer of raw BGR frames backed by ``collections.deque``.

    The main thread pushes frames; the asyncio event-handling coroutine (on a
    different thread) calls ``snapshot()`` to obtain a point-in-time copy.
    A ``threading.Lock`` guards both operations.

    Memory note
    ───────────
    At 1080p, 60 frames ≈ 360 MB.  If memory is constrained, reduce
    ``EventConfig.buffer_maxlen`` or add frame downscaling here.
    """

    def __init__(self, maxlen: int = 60) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        """Append *frame* to the buffer, evicting the oldest if full."""
        with self._lock:
            self._buf.append(frame)

    def snapshot(self) -> list[np.ndarray]:
        """
        Return a shallow copy of all buffered frames, oldest first.

        The returned list is independent of the deque; callers may hold it
        arbitrarily long without blocking the main thread.
        """
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ─── Frame Reader (daemon thread) ────────────────────────────────────────────


class FrameReader:
    """
    Daemon thread that continuously reads frames from an RTSP stream and
    exposes only the *latest* frame via a Queue of size 1.

    Older frames are explicitly discarded so the main thread never processes
    a stale frame — eliminating latency build-up.

    Disconnections trigger an exponential back-off retry loop; the thread
    keeps running until ``stop()`` is called.
    """

    def __init__(self, config: StreamConfig) -> None:
        self._cfg = config
        # maxsize=1: the queue always holds at most one frame (the freshest).
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="FrameReader",
            daemon=True,
        )
        # True for any source that is NOT a network stream.
        # Used in two places:
        #   1. _open_capture() — skips RTSP-specific buffer/timeout tuning that
        #      breaks local-file metadata scanning (moov atom, index tables).
        #   2. inner read loop — injects a 1/30 s sleep to simulate camera pacing
        #      instead of decoding at full CPU speed.
        _url_lower = config.rtsp_url.lower()
        self._is_local: bool = not (
            _url_lower.startswith("rtsp://")
            or _url_lower.startswith("http://")
            or _url_lower.startswith("https://")
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "FrameReader":
        self._thread.start()
        logger.info("FrameReader started → %s", self._cfg.rtsp_url)
        return self

    def stop(self) -> None:
        self._stop_event.set()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Block up to *timeout* seconds for a fresh frame; returns None on stall."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        if self._is_local:
            return self._open_local()
        return self._open_network()

    def _open_local(self) -> cv2.VideoCapture:
        """
        Open a local file (MP4, AVI, …) with zero network-oriented constraints.

        Why each restriction is lifted
        ───────────────────────────────
        • **No backend hint** — omitting ``cv2.CAP_FFMPEG`` lets OpenCV choose
          the best available backend.  Forcing ``CAP_FFMPEG`` can activate
          network-transport code paths that don't apply to files on disk.

        • **No ``CAP_PROP_BUFFERSIZE = 1``** — this cap tells FFmpeg to keep
          only one decoded frame in its internal queue.  For container formats
          (MP4, MOV, MKV) FFmpeg must first locate and parse the *moov atom*
          (or equivalent metadata/index block) before it can seek to any frame.
          A buffer size of 1 races against that initial scan and causes
          ``moov atom not found`` / ``Stream timeout`` errors on the very first
          ``cap.read()`` call.

        • **``OPENCV_FFMPEG_CAPTURE_OPTIONS`` cleared** — if this environment
          variable is set (e.g., by a parent process, a previous test run, or a
          system-wide profile) it injects FFmpeg AVOptions such as
          ``rtsp_transport`` or ``stimeout`` that are meaningless — and
          potentially fatal — for local file I/O.  We pop it for the duration
          of the constructor call only and restore it immediately afterwards so
          no other thread is affected.
        """
        saved = os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        try:
            cap = cv2.VideoCapture(self._cfg.rtsp_url)
        finally:
            # Restore unconditionally so the env is always left consistent,
            # even if VideoCapture raises an unexpected exception.
            if saved is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = saved
        return cap

    def _open_network(self) -> cv2.VideoCapture:
        """
        Open an RTSP / HTTP stream with low-latency network optimizations.

        • ``capture_backend`` (default ``cv2.CAP_FFMPEG``) is passed explicitly
          so OpenCV uses the FFmpeg demuxer, which handles RTSP negotiation and
          the full range of transport protocols (TCP/UDP/multicast).
        • ``CAP_PROP_BUFFERSIZE = 1`` minimises the internal decoded-frame
          queue so the main thread always receives the *freshest* frame rather
          than one that has been queued for several hundred milliseconds.
        """
        cap = cv2.VideoCapture(self._cfg.rtsp_url, self._cfg.capture_backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish(self, frame: np.ndarray) -> None:
        """Drop the stale queued frame (if any) and publish the new one."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass  # extremely unlikely race; ignore

    def _run(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            cap = self._open_capture()
            if not cap.isOpened():
                attempt += 1
                wait = min(self._cfg.reconnect_delay * (2 ** min(attempt - 1, 5)), 60.0)
                logger.warning(
                    "Cannot open stream (attempt %d). Retrying in %.1f s …", attempt, wait
                )
                if (
                    self._cfg.max_reconnect_attempts > 0
                    and attempt >= self._cfg.max_reconnect_attempts
                ):
                    logger.error("Max reconnect attempts reached. Stopping reader.")
                    break
                self._stop_event.wait(wait)
                continue

            logger.info("Stream opened (attempt %d).", attempt + 1)
            attempt = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self._is_local:
                        logger.info("Local file ended — reopening for loop playback.")
                    else:
                        logger.warning("Frame read failed — stream disconnected.")
                    break
                self._publish(frame)
                # Throttle local file decoding to ≈30 FPS so downstream
                # processing sees the same inter-frame cadence as a live camera.
                # The sleep is skipped entirely for real RTSP streams where the
                # network itself provides natural pacing.
                if self._is_local:
                    time.sleep(1.0 / 30.0)

            cap.release()
            if not self._stop_event.is_set():
                logger.info("Reconnecting in %.1f s …", self._cfg.reconnect_delay)
                self._stop_event.wait(self._cfg.reconnect_delay)

        logger.info("FrameReader stopped.")


# ─── Supabase-backed runtime config ──────────────────────────────────────────


class ConfigManager:
    """
    Fetches per-camera runtime config (ROI polygon + PPE ``active_rules``)
    from the Supabase ``camera_configs`` table and hot-reloads it every
    ``poll_interval`` seconds in a daemon thread.

    Schema (Supabase table ``camera_configs``)
    ──────────────────────────────────────────
        camera_id     text  primary key
        roi_polygon   jsonb  -- [[x1,y1],[x2,y2],...]
        active_rules  jsonb  -- ["hardhat","vest",...]

    Threading model
    ───────────────
    The main loop reads ``config_mgr.roi`` and ``config_mgr.active_rules``
    each frame.  A single background poller replaces these under a
    ``threading.Lock``; attribute reads return the latest consistent
    snapshot.  Reference assignment is atomic in CPython so the main loop
    never observes a partially-constructed ``PolygonROI``.
    """

    _SUPABASE_URL_ENV = "SUPABASE_URL"
    _SUPABASE_KEY_ENV = "SUPABASE_SERVICE_KEY"

    def __init__(self, camera_id: str, poll_interval: float = 60.0) -> None:
        url = os.environ.get(self._SUPABASE_URL_ENV)
        key = os.environ.get(self._SUPABASE_KEY_ENV)
        if not url or not key:
            raise RuntimeError(
                f"{self._SUPABASE_URL_ENV} and {self._SUPABASE_KEY_ENV} "
                "must be set in the environment."
            )

        self._camera_id = camera_id
        self._poll_interval = poll_interval
        self._client = httpx.Client(
            base_url=f"{url.rstrip('/')}/rest/v1",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
            timeout=10.0,
        )

        self._lock = threading.Lock()
        self._roi: Optional[PolygonROI] = None
        self._active_rules: list[str] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Initial synchronous fetch — fail fast if the camera is unknown or
        # Supabase is unreachable, rather than starting the pipeline with no
        # ROI and discovering the problem 60 s later.
        self._fetch_and_apply()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def roi(self) -> "PolygonROI":
        with self._lock:
            assert self._roi is not None  # guaranteed by __init__ fetch
            return self._roi

    @property
    def active_rules(self) -> list[str]:
        with self._lock:
            return list(self._active_rules)

    def start(self) -> "ConfigManager":
        """Launch the background polling thread (daemon)."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._poll_loop, name="ConfigPoller", daemon=True
        )
        self._thread.start()
        logger.info(
            "ConfigManager polling camera_id=%s every %.0fs.",
            self._camera_id, self._poll_interval,
        )
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._client.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_and_apply(self) -> None:
        """Single REST call → parse → atomic swap of the cached snapshot."""
        resp = self._client.get(
            "/camera_configs",
            params={
                "camera_id": f"eq.{self._camera_id}",
                "select": "roi_polygon,active_rules",
                "limit": 1,
            },
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            raise RuntimeError(
                f"No camera_configs row for camera_id={self._camera_id!r}."
            )
        row = rows[0]

        raw_polygon = row.get("roi_polygon") or []
        vertices = [(int(v[0]), int(v[1])) for v in raw_polygon]
        rules = [str(r) for r in (row.get("active_rules") or [])]

        new_roi = PolygonROI(vertices)
        with self._lock:
            self._roi = new_roi
            self._active_rules = rules
        logger.info(
            "ConfigManager updated: %d ROI vertices, active_rules=%s",
            len(vertices), rules,
        )

    def _poll_loop(self) -> None:
        # Interruptible sleep: wait() returns True when stop() is called.
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._fetch_and_apply()
            except Exception as exc:  # noqa: BLE001 — poll must never die
                logger.warning(
                    "ConfigManager poll failed (keeping last config): %s", exc
                )


# ─── Polygon Region of Interest ──────────────────────────────────────────────


class PolygonROI:
    """
    Closed polygon region of interest defined by integer pixel vertices.

    Uses ``cv2.pointPolygonTest`` for sub-pixel accurate containment checks
    in O(n) time relative to the number of polygon vertices.
    """

    def __init__(self, vertices: list[tuple[int, int]]) -> None:
        if len(vertices) < 3:
            raise ValueError("A polygon ROI requires at least 3 vertices.")
        self._pts = np.array(vertices, dtype=np.int32)

    def contains_point(self, point: tuple[float, float]) -> bool:
        """
        Return True if *point* (x, y) lies inside or on the polygon boundary.
        """
        # measureDist=False returns +1 (inside), 0 (on boundary), -1 (outside)
        return cv2.pointPolygonTest(self._pts, point, measureDist=False) >= 0

    def draw(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        fill_alpha: float = 0.10,
    ) -> None:
        """Overlay the polygon on *frame* in-place with an optional filled tint."""
        if fill_alpha > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self._pts], color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
        cv2.polylines(frame, [self._pts], isClosed=True, color=color, thickness=thickness)

    @property
    def vertices(self) -> np.ndarray:
        return self._pts


# ─── Person Detector (YOLOv8 Nano) ───────────────────────────────────────────


class PersonDetector:
    """
    Wraps Ultralytics YOLO11s-Pose running as a TensorRT engine on Jetson.

    Filters to the *person* class only and returns a ``Detections`` container
    whose attributes satisfy ``BYTETracker.update()``'s contract.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._cfg = config
        logger.info(
            "Loading model '%s' on device='%s' …", config.model_path, config.device
        )
        # task='pose' is required when loading a serialized TensorRT .engine:
        # the Ultralytics loader cannot always recover the task head from engine
        # metadata, and without it will silently initialise as a detect model,
        # leaving results[0].keypoints == None.
        is_engine = str(config.model_path).lower().endswith(".engine")
        self._model = (
            YOLO(config.model_path, task="pose") if is_engine
            else YOLO(config.model_path)
        )
        self._warmup()

    def _warmup(self) -> None:
        """One dummy inference pass to absorb JIT / kernel-launch overhead."""
        dummy = np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), dtype=np.uint8)
        self._infer(dummy)
        logger.info("Model warm-up complete.")

    def _infer(self, frame: np.ndarray):  # type: ignore[return]
        return self._model.predict(
            source=frame,
            classes=[self._cfg.person_class_id],
            conf=self._cfg.confidence,
            iou=self._cfg.iou,
            device=self._cfg.device,
            imgsz=self._cfg.imgsz,
            verbose=False,
        )

    def detect(self, frame: np.ndarray) -> Detections:
        """
        Run a single-pass inference on *frame* and return person detections
        with pose data.

        Multi-model data flow — Stage 1: Inference
        ───────────────────────────────────────────
        YOLO11s-Pose (TensorRT engine) runs a single forward pass on the full
        frame and jointly predicts:
          • Bounding boxes  → results[0].boxes   (used by BYTETracker)
          • 17 COCO keypoints per person → results[0].keypoints
            Shape: (N, 17, 3) where the last axis is [x_px, y_px, confidence].

        COCO keypoint index map (used downstream for anonymisation / skeleton):
          0  nose        1  left_eye    2  right_eye
          3  left_ear    4  right_ear
          5  left_shoulder   6  right_shoulder
          7  left_elbow      8  right_elbow
          9  left_wrist     10  right_wrist
         11  left_hip       12  right_hip
         13  left_knee      14  right_knee
         15  left_ankle     16  right_ankle

        Returns:
            ``Detections`` with arrays of shape (N,4), (N,), (N,), (N,17,3).
        """
        results = self._infer(frame)
        boxes   = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()

        n = len(boxes)

        # ── Extract keypoints ─────────────────────────────────────────────────
        # results[0].keypoints.data is a tensor of shape (N, 17, 3).
        # Fall back to an all-zeros array when the model returns no keypoint
        # data (e.g. if a non-pose checkpoint is accidentally loaded).
        kp_result = results[0].keypoints
        if kp_result is not None and kp_result.data is not None and len(kp_result.data) > 0:
            keypoints = kp_result.data.cpu().numpy().astype(np.float32)
        else:
            # Zero-filled sentinel: confidence channel = 0 means every
            # downstream confidence gate will treat all keypoints as invisible.
            keypoints = np.zeros((n, 17, 3), dtype=np.float32)

        return Detections(
            xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            conf=boxes.conf.cpu().numpy().astype(np.float32),
            cls =boxes.cls.cpu().numpy().astype(np.float32),
            keypoints=keypoints,
        )


# ─── ByteTrack wrapper ────────────────────────────────────────────────────────


class PersonTracker:
    """
    Thin wrapper around Ultralytics' ``BYTETracker``.

    Accepts a ``Detections`` object and returns ``(track_id, bbox_xyxy)`` pairs.
    """

    def __init__(self, config: TrackerConfig) -> None:
        args = IterableSimpleNamespace(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh =config.track_low_thresh,
            new_track_thresh =config.new_track_thresh,
            track_buffer     =config.track_buffer,
            match_thresh     =config.match_thresh,
            fuse_score       =True,
        )
        self._tracker = BYTETracker(args, frame_rate=config.frame_rate)

    def update(
        self,
        detections: Detections,
        frame: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Feed detections into ByteTrack and return active tracks.

        ``BYTETracker`` accesses detections via named attributes (``.xyxy``,
        ``.conf``, ``.cls``) **and** via subscript (``detections[mask]``).
        ``Detections`` satisfies both contracts natively, so we pass it
        through without any conversion.

        Returns:
            List of ``(track_id, [x1, y1, x2, y2])`` for every live track.
        """
        tracks = self._tracker.update(detections, frame)
        if tracks is None or len(tracks) == 0:
            return []
        result: list[tuple[int, np.ndarray]] = []
        for t in tracks:
            if hasattr(t, "track_id"):
                # Standard STrack object returned by most Ultralytics versions
                result.append((int(t.track_id), t.tlbr))
            else:
                # Raw numpy row: [x1, y1, x2, y2, track_id, ...]
                result.append((int(t[4]), t[:4]))
        return result


# ─── Entity Registry ─────────────────────────────────────────────────────────


class EntityRegistry:
    """
    Maintains a ``TrackedEntity`` for every ByteTrack ID seen so far.

    Only the main thread accesses this object — no locking required.
    Entities are kept alive after a track is lost (for cooldown tracking)
    and pruned by the main loop via ``prune_stale()``.
    """

    def __init__(self) -> None:
        self._entities: dict[int, TrackedEntity] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        active_tracks: list[tuple[int, np.ndarray]],
        roi: PolygonROI,
        now: float,
    ) -> None:
        """
        Reconcile the registry with the current frame's active tracks.

        For each active track:
          - Create a new ``TrackedEntity`` if this ID is first seen.
          - Update position, timestamp, and ROI membership.
          - Increment ``consecutive_roi_frames`` if centroid is in ROI,
            otherwise reset it to 0.

        For IDs no longer in ``active_tracks`` the entity is left intact
        (to preserve cooldown state) but ``consecutive_roi_frames`` is reset.
        """
        active_ids: set[int] = set()

        for tid, bbox in active_tracks:
            active_ids.add(tid)
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float((bbox[1] + bbox[3]) / 2.0)
            centroid: tuple[float, float] = (cx, cy)
            in_roi = roi.contains_point(centroid)

            if tid not in self._entities:
                self._entities[tid] = TrackedEntity(
                    track_id=tid,
                    bbox=bbox.copy(),
                    centroid=centroid,
                    first_seen_timestamp=now,
                    last_seen_timestamp=now,
                )

            entity = self._entities[tid]
            entity.bbox = bbox.copy()
            entity.centroid = centroid
            entity.last_seen_timestamp = now
            entity.frames_visible += 1
            entity.in_roi = in_roi

            if in_roi:
                entity.consecutive_roi_frames += 1
            else:
                entity.consecutive_roi_frames = 0

        # Reset dwell counter for tracks that vanished this frame
        for tid, entity in self._entities.items():
            if tid not in active_ids:
                entity.in_roi = False
                entity.consecutive_roi_frames = 0

    def all_entities(self) -> list[TrackedEntity]:
        """Return all known entities (including recently lost ones)."""
        return list(self._entities.values())

    def active_entities(
        self, active_ids: set[int]
    ) -> list[TrackedEntity]:
        """Return only entities whose track ID is currently active."""
        return [e for e in self._entities.values() if e.track_id in active_ids]

    def prune_stale(self, ttl: float, now: float) -> int:
        """
        Remove entities not seen for more than *ttl* seconds.

        Returns:
            Number of entities pruned.
        """
        stale = [
            tid
            for tid, e in self._entities.items()
            if now - e.last_seen_timestamp > ttl
        ]
        for tid in stale:
            del self._entities[tid]
        return len(stale)

    def __len__(self) -> int:
        return len(self._entities)


# ─── Async Event Manager ─────────────────────────────────────────────────────


class EventManager:
    """
    Evaluates per-entity dwell conditions and fires ``Event`` objects
    asynchronously when all conditions are met.

    Threading model
    ───────────────
    A dedicated daemon thread runs a private ``asyncio`` event loop.
    The main thread calls ``maybe_trigger()`` (synchronous); if conditions
    are met it schedules a coroutine on the async loop via
    ``asyncio.run_coroutine_threadsafe()``.  Inside the coroutine,
    CPU-bound JPEG encoding is further off-loaded to the loop's default
    ``ThreadPoolExecutor`` via ``loop.run_in_executor()``, keeping the
    async loop free for other coroutines.

    Callback
    ────────
    Supply ``on_event`` to handle events (e.g., push to message queue,
    HTTP webhook, database insert).  The default handler logs a summary.
    The callback is invoked from the asyncio thread pool — it must be
    thread-safe.
    """

    def __init__(
        self,
        config: EventConfig,
        on_event: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self._cfg = config
        self._on_event = on_event or self._default_dispatch

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="EventLoop",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info("EventManager async loop started.")

    # ── Public API (main thread) ──────────────────────────────────────────────

    def maybe_trigger(
        self,
        entity: TrackedEntity,
        frame_buffer: FrameBuffer,
        now: float,
        active_rules: Optional[list[str]] = None,
    ) -> bool:
        """
        Check whether *entity* should fire an event this frame.

        Conditions (all must hold):
          1. ``consecutive_roi_frames`` ≥ ``dwell_frames``
          2. Cooldown has elapsed since the last event for this ID

        If triggered:
          - ``last_event_timestamp`` is updated immediately (main thread) to
            block concurrent triggers before the async task completes.
          - ``consecutive_roi_frames`` is reset to 0.
          - A ``FrameBuffer`` snapshot is taken synchronously.
          - An async coroutine is scheduled for JPEG extraction + dispatch.

        Returns:
            True if an event was triggered this call.
        """
        if entity.consecutive_roi_frames < self._cfg.dwell_frames:
            return False

        elapsed_since_last = now - entity.last_event_timestamp
        if elapsed_since_last < self._cfg.cooldown_seconds:
            remaining = self._cfg.cooldown_seconds - elapsed_since_last
            logger.debug(
                "ID %d cooldown active — %.1f s remaining.", entity.track_id, remaining
            )
            return False

        # ── Arm the event ──────────────────────────────────────────────────
        dwell_count = entity.consecutive_roi_frames  # capture before reset
        entity.last_event_timestamp = now            # block re-trigger NOW
        entity.consecutive_roi_frames = 0            # reset dwell counter

        # Snapshot must be taken on the main thread (O(n) deque copy, very fast)
        snapshot: list[np.ndarray] = frame_buffer.snapshot()

        asyncio.run_coroutine_threadsafe(
            self._process_event(
                entity.track_id, dwell_count, snapshot, now,
                list(active_rules) if active_rules else [],
            ),
            self._loop,
        )
        logger.info(
            "Event queued for ID %d (dwell=%d frames, buffer=%d frames).",
            entity.track_id,
            dwell_count,
            len(snapshot),
        )
        return True

    def shutdown(self) -> None:
        """Signal the async loop to stop and wait for it to exit."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5.0)
        logger.info("EventManager shut down.")

    # ── Async pipeline (event loop thread) ───────────────────────────────────

    async def _process_event(
        self,
        track_id: int,
        dwell_frames: int,
        snapshot: list[np.ndarray],
        triggered_at: float,
        active_rules: list[str],
    ) -> None:
        """
        Async coroutine: encode JPEG snapshots then dispatch the Event.

        JPEG encoding is CPU-bound → runs in the default ThreadPoolExecutor
        so this coroutine yields the event loop while waiting.
        """
        loop = asyncio.get_running_loop()
        jpeg_streams: list[io.BytesIO] = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            self._encode_jpegs,
            snapshot,
        )

        event = Event(
            track_id=track_id,
            triggered_at=triggered_at,
            dwell_frames_at_trigger=dwell_frames,
            jpeg_streams=jpeg_streams,
            active_rules=active_rules,
        )

        # Support both plain sync callbacks and async callables transparently.
        # Sync callbacks are run in the executor to avoid blocking the event loop.
        #
        # Why two checks instead of one
        # ───────────────────────────────
        # asyncio.iscoroutinefunction (and the inspect variant it wraps) only
        # returns True when the object itself is a function or bound method with
        # the CO_COROUTINE flag.  A *callable instance* — i.e. any object whose
        # class defines ``async def __call__`` — is neither a function nor a
        # method, so the first check alone silently returns False.  The runtime
        # then dispatches to run_in_executor, which calls __call__ in a thread
        # pool, receives a coroutine object (not a result), and discards it →
        # "RuntimeWarning: coroutine '...__call__' was never awaited".
        #
        # The second check, inspect.iscoroutinefunction(type(cb).__call__),
        # catches exactly that pattern: callable class instances whose __call__
        # is declared with ``async def``.  Together the two checks cover every
        # supported callback shape:
        #
        #   • async def fn(event): ...              → first check  ✓
        #   • async def method on bound obj         → first check  ✓
        #   • class with async def __call__(self, event) → second check ✓
        #   • plain sync lambda / function          → neither → executor ✓
        cb = self._on_event
        if inspect.iscoroutinefunction(cb) or inspect.iscoroutinefunction(
            getattr(type(cb), "__call__", None)
        ):
            await cb(event)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, cb, event)

    def _select_frames(self, snapshot: list[np.ndarray]) -> list[np.ndarray]:
        """
        Select ``snapshot_count`` evenly-spaced frames from *snapshot*.

        Uses ``np.linspace`` to compute indices so the selection spans the
        full temporal extent of the buffer regardless of its current length.
        """
        n = len(snapshot)
        if n == 0:
            return []
        count = min(self._cfg.snapshot_count, n)
        indices = np.linspace(0, n - 1, count, dtype=int)
        return [snapshot[int(i)] for i in indices]

    def _encode_jpegs(self, snapshot: list[np.ndarray]) -> list[io.BytesIO]:
        """
        Encode selected frames as highly-compressed JPEGs into ``io.BytesIO``.

        Runs synchronously inside a thread pool worker — never call from the
        async loop directly.

        Returns:
            List of ``io.BytesIO`` streams, each seeked to position 0.
        """
        frames = self._select_frames(snapshot)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.jpeg_quality]
        streams: list[io.BytesIO] = []

        for idx, frame in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                logger.warning("JPEG encode failed for snapshot frame %d — skipping.", idx)
                continue
            stream = io.BytesIO(buf.tobytes())
            stream.seek(0)
            streams.append(stream)

        return streams

    # ── Default event handler ─────────────────────────────────────────────────

    @staticmethod
    def _default_dispatch(event: Event) -> None:
        """Log a structured summary of the fired event."""
        size_kb = event.total_jpeg_bytes / 1024
        logger.info(
            "┌─ EVENT FIRED ─────────────────────────────────────────\n"
            "│  Track ID  : %d\n"
            "│  Time      : %s\n"
            "│  Dwell     : %d consecutive frames in ROI\n"
            "│  Snapshots : %d JPEG frames\n"
            "│  JPEG size : %.1f KB (quality=%s)\n"
            "└───────────────────────────────────────────────────────",
            event.track_id,
            event.triggered_at_str,
            event.dwell_frames_at_trigger,
            len(event.jpeg_streams),
            size_kb,
            "varies",  # quality comes from config; shown for info
        )


# ─── Annotation helper ───────────────────────────────────────────────────────


def _annotate_and_show(
    frame: np.ndarray,
    tracks: list[tuple[int, np.ndarray]],
    fps: float,
    roi: PolygonROI,
    registry: EntityRegistry,
) -> None:
    """
    Render bounding boxes, track IDs, ROI polygon, and dwell counters
    onto a copy of *frame* and display it in an OpenCV window.
    """
    vis = frame.copy()

    # Draw ROI polygon (semi-transparent tint + outline)
    roi.draw(vis, color=(0, 255, 255), thickness=2, fill_alpha=0.10)

    # Draw per-track annotations
    entity_map: dict[int, TrackedEntity] = {
        e.track_id: e for e in registry.all_entities()
    }

    for tid, box in tracks:
        x1, y1, x2, y2 = map(int, box)
        entity = entity_map.get(tid)

        # Green when in ROI, grey otherwise
        color = (0, 220, 0) if (entity and entity.in_roi) else (160, 160, 160)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid}"
        if entity:
            label += f"  dwell:{entity.consecutive_roi_frames}"
        cv2.putText(
            vis, label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    # HUD: FPS + active count
    cv2.putText(
        vis, f"FPS: {fps:.1f}  |  People: {len(tracks)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2,
    )
    cv2.imshow("RTSP Tracker", vis)


# ─── Pose processing: anonymisation + skeletal annotation ────────────────────
#
# Multi-model data flow overview
# ──────────────────────────────
# Stage 1 — Inference   (PersonDetector.detect)
#   YOLOv8s-Pose produces bounding boxes AND 17 COCO keypoints per person.
#
# Stage 2 — Anonymisation   (_apply_face_blur)
#   Facial keypoints 0-4 (nose, eyes, ears) are used to mathematically infer
#   a tight bounding box around the worker's face.  A heavy GaussianBlur is
#   applied to that region IN-PLACE before the frame is pushed to the rolling
#   FrameBuffer, ensuring that every JPEG snapshot sent to the VLM and stored
#   in cloud telemetry is permanently anonymised at the source.
#
# Stage 3 — Annotation   (_draw_skeleton)
#   Body keypoints 5-16 (shoulders through ankles) are connected by coloured
#   limb lines to produce a 2-D skeletal overlay on the same frame.  This
#   gives the VLM explicit structural information about joint alignment that
#   would otherwise require it to interpret pixel-level clothing texture alone.
#
# Stage 4 — VLM Reasoning   (GeminiClient.analyse in vlm_escalation.py)
#   The Gemini 2.5 Flash model receives the 3×3 storyboard grid of
#   anonymised, skeleton-annotated frames and evaluates both vest presence
#   (colour/texture) and vest fit (shoulder/torso joint alignment).
#
# The face-to-track association is performed via IoU matching between the
# BYTETracker output bboxes and the original detection bboxes, because
# BYTETracker does not expose the detection indices it consumed internally.

# Indices of the five face keypoints in the COCO 17-point schema.
_FACE_KP_IDX: list[int] = [0, 1, 2, 3, 4]  # nose, l_eye, r_eye, l_ear, r_ear

# Minimum keypoint confidence to treat a joint as visible.
_KP_CONF_THRESH: float = 0.30

# Skeleton edge table: (from_kp_idx, to_kp_idx, BGR_colour).
# Left-side limbs → warm orange | Right-side limbs → blue | Centre → green.
_SKELETON_EDGES: list[tuple[int, int, tuple[int, int, int]]] = [
    (5,  6,  (0,   220,  0)),   # L-shoulder  ↔  R-shoulder  (collar bar)
    (5,  7,  (30,  140, 255)),  # L-shoulder  →  L-elbow
    (7,  9,  (30,  140, 255)),  # L-elbow     →  L-wrist
    (6,  8,  (255, 100,  30)),  # R-shoulder  →  R-elbow
    (8,  10, (255, 100,  30)),  # R-elbow     →  R-wrist
    (5,  11, (0,   220, 220)),  # L-shoulder  →  L-hip       (torso left)
    (6,  12, (0,   220, 220)),  # R-shoulder  →  R-hip       (torso right)
    (11, 12, (0,   220,  0)),   # L-hip       ↔  R-hip       (pelvis)
    (11, 13, (60,  220, 220)),  # L-hip       →  L-knee
    (13, 15, (60,  220, 220)),  # L-knee      →  L-ankle
    (12, 14, (220, 180,  60)),  # R-hip       →  R-knee
    (14, 16, (220, 180,  60)),  # R-knee      →  R-ankle
]


def _infer_face_bbox(
    kpts: np.ndarray,
    frame_h: int,
    frame_w: int,
    conf_thresh: float = 0.20,
    pad_frac_x: float = 0.65,
    pad_frac_y_up: float = 0.10,
    pad_frac_y_down: float = 0.65,
) -> Optional[tuple[int, int, int, int]]:
    """
    Derive a face bounding box from COCO keypoints 0-4 (nose + eyes + ears).

    Asymmetric vertical padding
    ───────────────────────────
    The COCO face keypoints cluster around the eyes, nose, and ears.
    ``y_min`` therefore lands at approximately eye level — the forehead and
    any PPE worn on top of the head (hard hats, bump caps) sit *above* that
    line.

    To keep hard-hat crowns fully visible to the VLM:

    • ``pad_frac_y_up``   (default 0.10) — tiny upward expansion, just enough
      to cover the eyebrows which sit fractionally above the eye keypoints.
      This deliberately stops well short of the forehead so no upward PPE is
      ever blurred.

    • ``pad_frac_y_down`` (default 0.65) — generous downward expansion toward
      the chin; the mouth and jaw sit below all five keypoints, so this pad
      is still needed for complete facial anonymisation.

    • ``pad_frac_x``      (default 0.65) — symmetric horizontal expansion
      from ear to ear to cover cheeks and temples.

    All fractions are proportional to the tight keypoint span so the blur
    region scales naturally with subject distance from the camera.

    Args:
        kpts:            (17, 3) float32 array [x, y, confidence].
        frame_h/w:       Frame dimensions used to clamp the box in-bounds.
        conf_thresh:     Minimum keypoint confidence to include in the fit.
        pad_frac_x:      Fractional horizontal outward expansion.
        pad_frac_y_up:   Fractional upward expansion (eyebrows only).
        pad_frac_y_down: Fractional downward expansion (chin coverage).

    Returns:
        (x1, y1, x2, y2) integer pixel coords clamped to the frame, or
        None when fewer than 2 face keypoints are visible.
    """
    # Isolate the five face keypoints and keep only high-confidence ones.
    face_kpts = kpts[_FACE_KP_IDX]          # shape (5, 3)
    visible   = face_kpts[face_kpts[:, 2] > conf_thresh]

    # Need at least two visible face keypoints to form a meaningful box.
    if len(visible) < 2:
        return None

    x_min, y_min = float(visible[:, 0].min()), float(visible[:, 1].min())
    x_max, y_max = float(visible[:, 0].max()), float(visible[:, 1].max())

    # Use the tighter of width/height as the reference span for each axis so
    # the padding stays proportional regardless of viewing angle.
    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)

    pad_x    = span_x * pad_frac_x
    pad_up   = span_y * pad_frac_y_up    # small — eyebrows only, hard hat safe
    pad_down = span_y * pad_frac_y_down  # generous — reaches chin/jaw

    x1 = int(max(0,           x_min - pad_x))
    y1 = int(max(0,           y_min - pad_up))    # top edge stays near eye level
    x2 = int(min(frame_w - 1, x_max + pad_x))
    y2 = int(min(frame_h - 1, y_max + pad_down))  # bottom edge covers chin

    # Degenerate box guard (can occur when all keypoints overlap exactly).
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _apply_face_blur(frame: np.ndarray, kpts: np.ndarray) -> None:
    """
    Permanently anonymise the worker's face in *frame* using GaussianBlur.

    Multi-model data flow — Stage 2: Anonymisation
    ───────────────────────────────────────────────
    Called IN-PLACE before the frame is pushed to the FrameBuffer so that
    every downstream consumer — JPEG extraction, Storyboard builder, VLM,
    cloud storage — only ever receives the anonymised version.

    The Gaussian kernel size is derived from the face bounding-box dimensions
    and forced to an odd integer (required by OpenCV) so the blur intensity
    scales with subject proximity rather than being a fixed pixel count.

    Args:
        frame: BGR frame to modify in-place.
        kpts:  (17, 3) keypoint array for a single tracked person.
    """
    h, w = frame.shape[:2]
    face_bbox = _infer_face_bbox(kpts, h, w)
    if face_bbox is None:
        # No reliable face keypoints detected — skip rather than guess.
        return

    x1, y1, x2, y2 = face_bbox
    face_roi = frame[y1:y2, x1:x2]

    # Kernel must be odd and at least 21 px for a visually heavy blur that
    # cannot be reversed by simple deconvolution.
    bw, bh  = x2 - x1, y2 - y1
    k_size  = max(21, int(max(bw, bh) * 0.35))
    k_size  = k_size if k_size % 2 == 1 else k_size + 1  # force odd

    # Write the blurred region back into the frame in-place.
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (k_size, k_size), 0)


def _draw_skeleton(frame: np.ndarray, kpts: np.ndarray) -> None:
    """
    Draw a high-contrast 2-D skeletal overlay for body joints 5-16.

    Multi-model data flow — Stage 3: Annotation
    ────────────────────────────────────────────
    Limb lines and joint dots are drawn only when BOTH endpoints have a
    confidence score above _KP_CONF_THRESH, preventing spurious lines from
    low-confidence predictions.

    Face keypoints (0-4) are intentionally skipped because that region has
    already been blurred by the anonymisation layer; drawing landmarks on a
    blurred region would provide no useful information to the VLM.

    The colour coding follows the _SKELETON_EDGES table:
      • Warm orange  — left-side limbs (from the subject's perspective)
      • Blue         — right-side limbs
      • Bright green — collar bar and pelvis (central axis)

    Args:
        frame: BGR frame to annotate in-place.
        kpts:  (17, 3) keypoint array for a single tracked person.
    """
    # ── Limb lines ────────────────────────────────────────────────────────────
    for idx_a, idx_b, colour in _SKELETON_EDGES:
        xa, ya, ca = kpts[idx_a]
        xb, yb, cb = kpts[idx_b]
        # Skip this limb if either endpoint is below the confidence threshold.
        if ca < _KP_CONF_THRESH or cb < _KP_CONF_THRESH:
            continue
        cv2.line(
            frame,
            (int(xa), int(ya)),
            (int(xb), int(yb)),
            colour, 2, cv2.LINE_AA,
        )

    # ── Joint dots (body only; face joints omitted — already blurred) ─────────
    for idx in range(5, 17):
        x, y, c = kpts[idx]
        if c < _KP_CONF_THRESH:
            continue
        pt = (int(x), int(y))
        # White-filled circle with a thin dark outline for visibility on any
        # background colour.
        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (30,   30,  30),  1, cv2.LINE_AA)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection-over-Union for two [x1, y1, x2, y2] boxes.

    Used exclusively by _map_track_keypoints to re-associate BYTETracker
    output bboxes with the detection keypoints that spawned them.
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _map_track_keypoints(
    active_tracks: list[tuple[int, np.ndarray]],
    detections: Detections,
    iou_threshold: float = 0.30,
) -> dict[int, np.ndarray]:
    """
    Re-associate BYTETracker track bboxes with per-detection keypoints.

    BYTETracker does not expose the detection indices it consumed, so this
    function greedily matches each track's bbox to the detection with the
    highest IoU.  Tracks with no match above iou_threshold are omitted from
    the returned dict, meaning anonymisation and skeleton drawing will simply
    be skipped for that individual on this frame.

    Args:
        active_tracks: List of (track_id, [x1,y1,x2,y2]) from PersonTracker.
        detections:    The Detections object that was passed to the tracker
                       this frame, carrying the (N, 17, 3) keypoints array.
        iou_threshold: Minimum IoU to accept a detection as the match for a
                       given track; prevents false associations when detections
                       are close together.

    Returns:
        {track_id: keypoints (17, 3)} for every successfully matched track.
    """
    kp_map: dict[int, np.ndarray] = {}

    # Early exit when no pose data is available (e.g. fallback to plain bbox
    # model) to avoid unnecessary inner-loop work.
    if len(detections) == 0 or detections.keypoints.shape[0] == 0:
        return kp_map

    for track_id, track_bbox in active_tracks:
        best_iou:  float              = iou_threshold
        best_kpts: Optional[np.ndarray] = None

        for det_idx in range(len(detections)):
            score = _iou(track_bbox, detections.xyxy[det_idx])
            if score > best_iou:
                best_iou  = score
                best_kpts = detections.keypoints[det_idx]  # (17, 3)

        if best_kpts is not None:
            kp_map[track_id] = best_kpts

    return kp_map


# ─── Main pipeline ────────────────────────────────────────────────────────────


def run(
    stream_cfg: StreamConfig,
    detector_cfg: DetectorConfig,
    tracker_cfg: TrackerConfig,
    event_cfg: EventConfig,
    config_mgr: ConfigManager,
    on_event: Optional[Callable[[Event], None]] = None,
    display: bool = False,
) -> None:
    """
    Wire together all subsystems into a single real-time inference loop.

    Per-frame execution order (multi-model architecture)
    ─────────────────────────────────────────────────────
    1.  Acquire latest frame from FrameReader queue (blocks up to 2 s).
    2.  Run YOLOv8s-Pose: joint bbox + 17-keypoint inference  [Stage 1]
    3.  Update ByteTrack; receive active (track_id, bbox) pairs.
    4.  Re-associate track bboxes → detection keypoints via IoU matching.
    5.  Anonymisation layer: for every tracked person, infer a face bbox
        from keypoints 0-4 and apply GaussianBlur in-place.         [Stage 2]
    6.  Pose layer: draw 2-D skeletal overlay (joints 5-16) on the same
        processed frame.                                             [Stage 3]
    7.  Push the anonymised + skeleton-annotated frame to FrameBuffer.
        All downstream consumers (JPEG extractor, Storyboard, VLM, cloud
        storage) receive only this privacy-safe version.
    8.  Update EntityRegistry: recompute ROI membership and dwell counters.
    9.  For each entity, call EventManager.maybe_trigger() (synchronous
        check; async JPEG work is scheduled if triggered).
    10. Prune stale entities.
    11. Compute FPS; print console status line.
    12. (Optional) Render bounding boxes + ROI overlay on processed frame.
    """
    reader    = FrameReader(stream_cfg).start()
    detector  = PersonDetector(detector_cfg)
    tracker   = PersonTracker(tracker_cfg)
    fps_ctr   = FPSCounter(window=30)
    buf       = FrameBuffer(maxlen=event_cfg.buffer_maxlen)
    registry  = EntityRegistry()
    event_mgr = EventManager(event_cfg, on_event=on_event)
    config_mgr.start()

    logger.info(
        "Pipeline running.  dwell=%d frames  cooldown=%.0f s  "
        "(ROI + active_rules pulled from Supabase)",
        event_cfg.dwell_frames,
        event_cfg.cooldown_seconds,
    )
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            frame = reader.get_frame(timeout=2.0)
            if frame is None:
                logger.debug("Waiting for frame …")
                continue

            now = time.time()

            # Pull the latest ROI + PPE rule set from Supabase cache.  Both
            # may have just been swapped by the ConfigManager poll thread;
            # reading through the property is lock-guarded.
            roi = config_mgr.roi
            active_rules = config_mgr.active_rules

            # ── 1. Detect (Stage 1 — YOLOv8s-Pose inference) ─────────────────
            # Produces bounding boxes AND (N, 17, 3) keypoint arrays in one
            # forward pass.  The raw frame is never stored in the buffer so
            # un-anonymised faces never reach any downstream consumer.
            detections = detector.detect(frame)

            # ── 2. Track ──────────────────────────────────────────────────────
            active_tracks = tracker.update(detections, frame)
            active_ids = {tid for tid, _ in active_tracks}

            # ── 3. Associate keypoints → tracks via IoU ───────────────────────
            # BYTETracker returns (track_id, bbox) pairs without carrying the
            # originating detection indices.  _map_track_keypoints greedy-
            # matches each track bbox to the highest-IoU detection so we can
            # look up that person's 17 keypoints by track ID.
            kp_map = _map_track_keypoints(active_tracks, detections)

            # ── 4. Anonymisation + Pose annotation (Stages 2 & 3) ────────────
            # Work on a copy so the raw frame is never modified or buffered.
            # Face blur is applied first (in-place on proc_frame) so the
            # skeleton draw that follows never re-exposes facial landmarks.
            proc_frame = frame.copy()
            for track_id, _bbox in active_tracks:
                kpts = kp_map.get(track_id)
                if kpts is not None:
                    # Stage 2 — Anonymisation: blur face region derived from
                    # nose / eye / ear keypoints (COCO indices 0-4).
                    _apply_face_blur(proc_frame, kpts)
                    # Stage 3 — Pose annotation: draw coloured limb lines and
                    # joint dots for body keypoints (COCO indices 5-16).
                    _draw_skeleton(proc_frame, kpts)

            # ── 5. Buffer (anonymised + annotated frame only) ─────────────────
            # Every JPEG snapshot extracted by EventManager._encode_jpegs,
            # every Storyboard grid cell, and every image uploaded to cloud
            # storage will contain this privacy-safe, skeleton-annotated frame.
            buf.push(proc_frame)

            # ── 6. Update entity states ───────────────────────────────────────
            registry.update(active_tracks, roi, now)

            # ── 5. Evaluate event conditions ──────────────────────────────────
            triggered: list[int] = []
            for entity in registry.active_entities(active_ids):
                if event_mgr.maybe_trigger(entity, buf, now, active_rules):
                    triggered.append(entity.track_id)

            # ── 6. Prune old entities ─────────────────────────────────────────
            registry.prune_stale(ttl=event_cfg.entity_ttl_seconds, now=now)

            # ── 7. Console status ─────────────────────────────────────────────
            fps        = fps_ctr.tick()
            ids        = sorted(active_ids)
            roi_ids    = sorted(e.track_id for e in registry.active_entities(active_ids) if e.in_roi)
            dwell_info = {
                e.track_id: e.consecutive_roi_frames
                for e in registry.active_entities(active_ids)
                if e.in_roi
            }
            print(
                f"\rFPS:{fps:5.1f}  people:{len(ids):3d}  "
                f"IDs:{ids!s:<25}  "
                f"in-ROI:{roi_ids!s:<20}  "
                f"dwell:{dwell_info!s:<25}  "
                f"events:{triggered!s:<15}",
                end="",
                flush=True,
            )

            # ── 8. Optional display ───────────────────────────────────────────
            # proc_frame already carries the face blur and skeleton overlay;
            # _annotate_and_show adds bounding boxes and the ROI polygon on
            # top of that for the live monitor window only.
            if display:
                _annotate_and_show(proc_frame, active_tracks, fps, roi, registry)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print()
        logger.info("Interrupted by user.")
    finally:
        reader.stop()
        event_mgr.shutdown()
        config_mgr.stop()
        # If the caller passed a VLMEscalationHandler (or any handler with a
        # close_sync() hook), give it a chance to drain in-flight async work
        # and close network connections before the process exits.
        if on_event is not None and hasattr(on_event, "close_sync"):
            on_event.close_sync()  # type: ignore[union-attr]
        if display:
            cv2.destroyAllWindows()
        logger.info("Pipeline shut down.")


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RTSP Person Tracker — YOLOv8 Nano + ByteTrack + ROI Events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stream ────────────────────────────────────────────────────────────────
    p.add_argument("rtsp_url", help="RTSP stream URL (e.g. rtsp://user:pw@host/path)")
    p.add_argument(
        "--reconnect-delay", type=float, default=2.0, metavar="SEC",
        help="Base reconnect back-off (doubles on each failure, max 60 s)",
    )
    p.add_argument(
        "--max-retries", type=int, default=0, metavar="N",
        help="Max reconnect attempts (0 = infinite)",
    )

    # ── Detector ──────────────────────────────────────────────────────────────
    p.add_argument("--model",  default="yolov8n.pt")
    p.add_argument("--conf",   type=float, default=0.40, help="Detection confidence threshold")
    p.add_argument("--iou",    type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", default="cpu",            help="cuda | mps | cpu")
    p.add_argument("--imgsz",  type=int,   default=640)

    # ── Tracker ───────────────────────────────────────────────────────────────
    p.add_argument("--fps",          type=int,   default=30)
    p.add_argument("--track-high",   type=float, default=0.50)
    p.add_argument("--track-low",    type=float, default=0.10)
    p.add_argument("--new-track",    type=float, default=0.60)
    p.add_argument("--track-buffer", type=int,   default=30)
    p.add_argument("--match-thresh", type=float, default=0.80)

    # ── Event ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--camera-id",
        required=True,
        help=(
            "Supabase camera_configs.camera_id whose roi_polygon and "
            "active_rules drive this tracker instance."
        ),
    )
    p.add_argument(
        "--config-poll-interval", type=float, default=60.0, metavar="SEC",
        help="How often ConfigManager re-queries Supabase for ROI/rule updates",
    )
    p.add_argument(
        "--dwell-frames", type=int, default=45,
        help="Consecutive in-ROI frames required to fire an event",
    )
    p.add_argument(
        "--cooldown", type=float, default=15.0,
        help="Per-ID minimum gap between events (seconds)",
    )
    p.add_argument(
        "--buffer-len", type=int, default=60,
        help="Rolling frame buffer depth (frames)",
    )
    p.add_argument(
        "--snapshots", type=int, default=10,
        help="Number of evenly-spaced JPEG snapshots extracted per event",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=60,
        help="JPEG quality for in-memory snapshots (0–100)",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--display", action="store_true", help="Show annotated OpenCV window")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_mgr = ConfigManager(
        camera_id=args.camera_id,
        poll_interval=args.config_poll_interval,
    )

    run(
        stream_cfg=StreamConfig(
            rtsp_url=args.rtsp_url,
            reconnect_delay=args.reconnect_delay,
            max_reconnect_attempts=args.max_retries,
        ),
        detector_cfg=DetectorConfig(
            model_path=args.model,
            confidence=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
        ),
        tracker_cfg=TrackerConfig(
            frame_rate=args.fps,
            track_high_thresh=args.track_high,
            track_low_thresh=args.track_low,
            new_track_thresh=args.new_track,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
        ),
        event_cfg=EventConfig(
            dwell_frames=args.dwell_frames,
            cooldown_seconds=args.cooldown,
            buffer_maxlen=args.buffer_len,
            snapshot_count=args.snapshots,
            jpeg_quality=args.jpeg_quality,
        ),
        config_mgr=config_mgr,
        display=args.display,
    )


if __name__ == "__main__":
    # ── Local test mode ───────────────────────────────────────────────────────
    # Run directly (`python rtsp_tracker.py`) to process the local test video.
    # FrameReader will throttle to 30 FPS automatically because the path does
    # not start with "rtsp://".  Pass CLI arguments to use production settings:
    #   python rtsp_tracker.py rtsp://user:pass@host/stream
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Deferred import — vlm_escalation imports Event from this module at its
        # top level, so we must not import it at rtsp_tracker's module level or
        # we'd create a circular dependency.  Importing here (after this module
        # is fully loaded) is the standard Python pattern for breaking cycles.
        from vlm_escalation import VLMEscalationHandler, GeminiConfig

        # ── ANSI colours ──────────────────────────────────────────────────────
        _CYAN  = "\033[96m"   # bright cyan — stands out from the FPS log line
        _BOLD  = "\033[1m"
        _RESET = "\033[0m"

        def print_vlm_verdict(result) -> None:
            """Print the VLM JSON verdict to the console in bright cyan."""
            flag = "⚠  VIOLATION DETECTED" if result.analysis.violation_detected \
                   else "✓  No violation"
            print(
                f"\n{_CYAN}{_BOLD}"
                f"╔══ VLM VERDICT ════════════════════════════════════════════╗\n"
                f"║  {flag}\n"
                f"║  violation_type   : {result.analysis.violation_type}\n"
                f"║  confidence_score : {result.analysis.confidence_score:.2f}\n"
                f"║  reasoning        : {result.analysis.reasoning}\n"
                f"╚═══════════════════════════════════════════════════════════╝"
                f"{_RESET}",
                flush=True,
            )

        _vlm_handler = VLMEscalationHandler(
            gemini_cfg=GeminiConfig(),  # project/location default to Vertex AI config
            on_analysis=print_vlm_verdict,
        )

        # ── Video Selection ──
        _VIDEO_DIR = "/Users/cprakash/Documents/MY_AI/Luna_V2/Luna_V2_Test_Video/"
        
        _TEST_VIDEOS = [
            _VIDEO_DIR + "Video1.mp4",  # Index 0
            _VIDEO_DIR + "Video2.mp4",  # Index 1
            _VIDEO_DIR + "Video3.mp4",  # Index 2
        ]
        
        # Change this index (0, 1, or 2) to swap test videos
        _CURRENT_VIDEO = _TEST_VIDEOS[1] 
        
        print(f"\n--- LOADING VIDEO: {_CURRENT_VIDEO} ---\n")

        # Local test runs still need a camera_id in Supabase; override via
        # the CAMERA_ID env var.  ROI + active_rules come from that row.
        _test_camera_id = os.environ.get("CAMERA_ID", "local-test")

        run(
            stream_cfg=StreamConfig(rtsp_url=_CURRENT_VIDEO), # <-- Make sure this says _CURRENT_VIDEO
            detector_cfg=DetectorConfig(),
            tracker_cfg=TrackerConfig(),
            event_cfg=EventConfig(
                dwell_frames=5,
                snapshot_count=5,
            ),
            config_mgr=ConfigManager(camera_id=_test_camera_id),
            on_event=_vlm_handler,
            display=True,
        )
