#!/usr/bin/env python3
"""
VLM Escalation Module — Gemini 1.5 Flash safety-violation analyser.

Pipeline per event
──────────────────
  Event (list[io.BytesIO])
      │
      ▼
  Storyboard.build()
      • Select 9 frames evenly from the stream list (np.linspace)
      • Decode each JPEG stream → NumPy BGR array
      • Resize every cell to cfg.cell_w × cfg.cell_h
      • Annotate each cell: frame label (T1-T9), final-frame marker
      • np.hstack each row → np.vstack rows into 3×3 grid
      • cv2.imencode → JPEG bytes → base64 string
      │
      ▼
  GeminiClient.analyse()
      • POST to generativelanguage.googleapis.com (aiohttp)
      • response_schema enforces ViolationAnalysis JSON structure
      • Exponential back-off on 429 / 5xx; immediate fail on other 4xx
      │
      ▼
  ViolationAnalysis (Pydantic, frozen)
      • violation_detected: bool
      • violation_type:     str   ("No Hardhat", "None", …)
      • confidence_score:   float [0.0, 1.0]
      │
      ▼
  AnalysisResult dispatched to on_analysis callback

Non-blocking guarantee
──────────────────────
  VLMEscalationHandler.__call__ is an async coroutine.
  EventManager already awaits it on its own dedicated asyncio loop thread,
  so the main video-processing loop is never touched.

Usage
─────
  from vlm_escalation import VLMEscalationHandler, GeminiConfig, StoryboardConfig

  handler = VLMEscalationHandler(
      gemini_cfg=GeminiConfig(),  # uses ADC; project/location default to Vertex AI config
      on_analysis=lambda r: print(r.analysis),
  )
  # Pass as the on_event callback to rtsp_tracker.run()
  run(..., on_event=handler)
  # At shutdown (called automatically by rtsp_tracker.run()):
  handler.close_sync()

Dependencies
────────────
  pip install google-genai pydantic
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from slack_sdk import WebClient as SlackWebClient

from google import genai
from google.genai import types
import cv2
import numpy as np
from pydantic import BaseModel, Field, create_model, field_validator

# rtsp_tracker is imported for the Event type only; no circular dependency
# because rtsp_tracker never imports from this module.
from rtsp_tracker import Event

logger = logging.getLogger("vlm_escalation")


# ─── Pydantic response schema ─────────────────────────────────────────────────


class ViolationAnalysis(BaseModel):
    """
    Strict schema for the Gemini structured-output response.

    Marked ``frozen=True`` so instances are hashable and safe to share
    across threads without mutation risk.
    """

    model_config = {"frozen": True}

    violation_detected: bool = Field(
        description="True when a safety violation is present in the storyboard."
    )
    # Base-class description is deliberately generic.  The concrete schema
    # used for each Gemini call is built by ``_build_violation_schema`` and
    # narrows both the type (Literal[...]) and the description to the
    # ``active_rules`` for that specific camera.
    violation_type: str = Field(
        description="A concise label for the most significant violation observed, or 'None'."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the analysis, as a float in [0.0, 1.0].",
    )
    reasoning: str = Field(
        default="",
        description=(
            "One or two sentences explaining which visual evidence in the storyboard "
            "supports the conclusion.  Must be non-empty when violation_detected is true."
        ),
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> float:
        """Clamp silently in case the model returns a fractionally out-of-range value."""
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]

    @field_validator("violation_type", mode="before")
    @classmethod
    def _normalise_none(cls, v: object) -> str:
        """Map empty / 'N/A' / 'no violation' variants → canonical 'None'."""
        cleaned = str(v).strip()
        if cleaned.lower() in ("none", "n/a", "no violation", "no violations", ""):
            return "None"
        return cleaned


# ─── Configuration dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class StoryboardConfig:
    """
    Controls the layout and quality of the 3×3 temporal storyboard image.

    Memory / token trade-off
    ────────────────────────
    Each cell is resized to ``cell_w × cell_h`` before stitching.
    The final grid is ``(cols * cell_w) × (rows * cell_h)`` pixels.
    Reducing ``storyboard_jpeg_quality`` shrinks the base64 payload and
    lowers Gemini token consumption at the cost of image fidelity.
    """

    rows: int = 3
    cols: int = 3
    cell_w: int = 320                  # pixels per cell, width
    cell_h: int = 240                  # pixels per cell, height
    storyboard_jpeg_quality: int = 82  # higher than event snapshots for VLM clarity
    label_font_scale: float = 0.65
    label_thickness: int = 2


@dataclass(frozen=True)
class GeminiConfig:
    """
    Gemini SDK connection and generation parameters — Vertex AI backend.

    Authentication is handled by Application Default Credentials (ADC).
    No API key is required.  Before running, ensure the environment is
    authenticated::

        gcloud auth application-default login

    Or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path.
    """

    project: str = "luna-ai-490605"
    location: str = "us-central1"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.10           # low = deterministic safety judgements

    max_retries: int = 3
    retry_base_delay: float = 1.0       # seconds; doubles on each retry


# ─── Result container ─────────────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """
    Combines the original event with the VLM's structured analysis.

    ``storyboard_jpeg_bytes`` holds the raw (not base64) JPEG grid so
    downstream consumers can forward or store it without re-decoding.
    """

    event: Event
    analysis: ViolationAnalysis
    storyboard_jpeg_bytes: bytes       # raw JPEG of the 3×3 grid
    api_latency_ms: float
    model: str

    @property
    def track_id(self) -> int:
        return self.event.track_id

    @property
    def triggered_at_str(self) -> str:
        return self.event.triggered_at_str

    def summary(self) -> str:
        flag = "VIOLATION" if self.analysis.violation_detected else "clear"
        return (
            f"[{flag}] ID={self.track_id} | "
            f"type='{self.analysis.violation_type}' | "
            f"conf={self.analysis.confidence_score:.2f} | "
            f"latency={self.api_latency_ms:.0f}ms | "
            f"t={self.triggered_at_str}"
        )


# ─── Dynamic system prompt + schema (rule-scoped) ─────────────────────────────
#
# The factory manager configures each camera with an ``active_rules`` list in
# Supabase (e.g. ["hardhat", "vest"]).  Only those rules are flagged as
# violations; every other PPE category MUST be ignored — both to match
# customer intent and to prevent Gemini from hallucinating safety issues the
# site does not care about.
#
# For every escalation we rebuild:
#   1. The natural-language prompt — active rules get strict CHECK instructions,
#      inactive rules get explicit IGNORE instructions.
#   2. The Pydantic response_schema — ``violation_type`` is narrowed to a
#      ``Literal`` of just the labels that correspond to the active rules
#      (plus ``"None"``), and the field description enumerates that exact set.

# Per-rule prompt + label metadata.  Extend this table to onboard new PPE
# categories without touching the prompt assembler.
_RULE_SPECS: dict[str, dict] = {
    "hardhat": {
        "active_instruction": (
            "HARD HAT (ACTIVE RULE): You MUST verify that every visible worker "
            "is wearing a hard hat / safety helmet on the head.  Use the "
            "skeletal overlay's nose/ear keypoints (which anchor the head "
            "location even though the face pixels are blurred) to locate the "
            "head region, then inspect the pixels immediately above it for "
            "helmet geometry and colour.  If the head region is bare, "
            "flag 'Missing Hard Hat'."
        ),
        "ignore_instruction": (
            "HARD HAT (INACTIVE): Do NOT flag missing or absent hard hats.  "
            "Ignore head-protection status entirely — this site is not "
            "enforcing hard hat compliance."
        ),
        "violation_labels": ["Missing Hard Hat"],
    },
    "vest": {
        "active_instruction": (
            "HI-VIS VEST (ACTIVE RULE): You MUST verify that every visible "
            "worker is wearing a high-visibility safety vest properly on the "
            "torso.  Use the skeletal overlay's shoulder (5, 6), hip (11, 12) "
            "and collar bar to define the torso region, then check whether "
            "that region is covered by a bright hi-vis garment.  If the vest "
            "is absent flag 'No Hi-Vis Vest'; if it is present but the "
            "skeletal geometry shows it draped, held, or hanging off one "
            "shoulder rather than worn, flag 'Vest Not Worn Properly'."
        ),
        "ignore_instruction": (
            "HI-VIS VEST (INACTIVE): Do NOT flag missing, absent, or "
            "improperly worn hi-vis vests.  Ignore torso-garment status "
            "entirely — this site is not enforcing vest compliance."
        ),
        "violation_labels": ["No Hi-Vis Vest", "Vest Not Worn Properly"],
    },
}


def _allowed_violation_types(active_rules: list[str]) -> list[str]:
    """
    Return the ordered list of labels the model is allowed to emit for
    ``violation_type``.  Always includes ``"None"``.  If two or more rules
    are active, ``"Missing Multiple PPE"`` is also allowed.
    """
    allowed: list[str] = ["None"]
    normalised = [r.lower() for r in active_rules]
    for rule in normalised:
        spec = _RULE_SPECS.get(rule)
        if spec is None:
            continue
        for label in spec["violation_labels"]:
            if label not in allowed:
                allowed.append(label)
    if len(normalised) >= 2:
        allowed.append("Missing Multiple PPE")
    return allowed


def _build_prompt(active_rules: list[str]) -> str:
    """
    Assemble the Gemini system prompt for the given ``active_rules``.

    Active rules get strict CHECK blocks; inactive rules get explicit IGNORE
    blocks so Gemini cannot invent violations outside the customer's scope.
    """
    normalised = [r.lower() for r in active_rules]
    active_blocks: list[str] = []
    ignore_blocks: list[str] = []
    for rule, spec in _RULE_SPECS.items():
        if rule in normalised:
            active_blocks.append(f"  • {spec['active_instruction']}")
        else:
            ignore_blocks.append(f"  • {spec['ignore_instruction']}")

    if active_blocks:
        checks_section = (
            "Compliance checks for THIS camera (only these matter):\n"
            + "\n".join(active_blocks)
        )
    else:
        checks_section = (
            "Compliance checks for THIS camera: NONE configured.  Treat every "
            "worker as compliant and set violation_detected=false."
        )

    ignore_section = (
        "Rules that are NOT active for this camera — you MUST ignore them:\n"
        + "\n".join(ignore_blocks)
    ) if ignore_blocks else ""

    allowed = _allowed_violation_types(active_rules)
    allowed_str = ", ".join(f'"{t}"' for t in allowed)

    multi_clause = (
        f' Use "Missing Multiple PPE" only when two or more ACTIVE rules '
        f'are simultaneously violated.'
        if "Missing Multiple PPE" in allowed else ""
    )

    parts = [
        "You are an enterprise safety AI performing multi-modal reasoning on a "
        "temporal 3×3 storyboard grid produced by a dual-model computer-vision "
        "pipeline.",
        "",
        "Image pre-processing context (critical for correct interpretation):",
        "  • The faces of all workers have been blurred for privacy using a "
        "Gaussian filter derived from facial landmark keypoints.  Do not flag "
        "blurred face regions as anomalies.",
        "  • A 2-D skeletal pose overlay has been drawn over each worker using "
        "17 COCO keypoints.  Limb lines are colour-coded: warm orange = left "
        "side, blue = right side, green = collar and pelvis bars.  Use this "
        "overlay to reason about body posture and joint alignment.",
        "",
        "Reading order: left-to-right, top-to-bottom.",
        "  T1 (top-left) → earliest frame",
        "  T9 (bottom-right) → most recent frame",
        "",
        checks_section,
    ]
    if ignore_section:
        parts.extend(["", ignore_section])

    parts.extend([
        "",
        "If and ONLY IF one of the active rules above is violated, set "
        "violation_detected=true and detail which rule(s) failed in the "
        "reasoning string.  Never flag a violation that falls outside the "
        "active rule list.",
        "",
        f"Set violation_type to EXACTLY one of: {allowed_str}.{multi_clause} "
        "Any other string is forbidden.",
        "Set confidence_score in [0.0, 1.0] reflecting your certainty across "
        "the full temporal sequence.",
        "Set reasoning to one or two sentences citing the specific frame "
        "numbers (T1–T9), the visual evidence, and the skeletal joint "
        "evidence that support your conclusion.",
        "",
        "Respond with a single valid JSON object — no markdown fence, no "
        "explanation, no trailing text outside the JSON.",
    ])
    return "\n".join(parts)


def _build_violation_schema(active_rules: list[str]) -> type[ViolationAnalysis]:
    """
    Build a subclass of ``ViolationAnalysis`` whose ``violation_type`` is a
    ``Literal`` over exactly the labels corresponding to ``active_rules``.

    Using ``pydantic.create_model`` (rather than class-body annotations)
    matters here because the module uses ``from __future__ import annotations``
    — class-body annotations would be captured as strings and the closed-over
    ``allowed`` list would not survive ``get_type_hints`` resolution.
    """
    allowed = _allowed_violation_types(active_rules)
    # Literal accepts a tuple; this is equivalent to Literal[allowed[0], allowed[1], ...].
    literal_type = Literal[tuple(allowed)]  # type: ignore[valid-type]
    description = (
        "A concise label for the most significant violation observed, or "
        f"'None'. Use EXACTLY one of: {', '.join(repr(t) for t in allowed)}. "
        "Any other label is forbidden."
    )

    return create_model(  # type: ignore[call-overload]
        "ViolationAnalysisScoped",
        __base__=ViolationAnalysis,
        violation_type=(literal_type, Field(description=description)),
    )


# ─── Temporal storyboard builder ─────────────────────────────────────────────


class Storyboard:
    """
    Builds a 3×3 temporal grid image from up to N JPEG byte streams.

    Frame selection
    ───────────────
    If more than ``rows * cols`` streams are available, exactly
    ``rows * cols`` are selected using ``np.linspace`` so the chosen frames
    span the full temporal window uniformly.  If fewer are available, all
    are used and the remaining cells are filled with black padding frames.

    All I/O (stream decoding, NumPy ops, cv2.imencode) is synchronous and
    CPU-bound; callers should run this inside ``loop.run_in_executor``.
    """

    def __init__(self, cfg: StoryboardConfig = StoryboardConfig()) -> None:
        self._cfg = cfg
        self._n_cells = cfg.rows * cfg.cols

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self, jpeg_streams: list[io.BytesIO]) -> tuple[np.ndarray, bytes, str]:
        """
        Construct the storyboard from *jpeg_streams*.

        Args:
            jpeg_streams: JPEG-encoded frames from ``EventManager``, oldest first.

        Returns:
            A 3-tuple of:
              - grid_bgr:   ``np.ndarray`` (H×W×3, uint8) — the stitched grid
              - jpeg_bytes: raw JPEG bytes of the grid (for storage / forwarding)
              - b64_str:    base64-encoded JPEG string (for Gemini inline payload)
        """
        selected = self._select_streams(jpeg_streams)
        cells = self._decode_cells(selected)
        grid = self._stitch(cells)
        jpeg_bytes = self._encode_jpeg(grid)
        b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
        return grid, jpeg_bytes, b64_str

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_streams(self, streams: list[io.BytesIO]) -> list[Optional[io.BytesIO]]:
        """
        Choose ``n_cells`` streams via evenly-spaced linspace indices.
        Returns a list of length ``n_cells``; missing slots are ``None``.
        """
        n = len(streams)
        if n == 0:
            return [None] * self._n_cells

        if n >= self._n_cells:
            indices = np.linspace(0, n - 1, self._n_cells, dtype=int)
            return [streams[int(i)] for i in indices]

        # Fewer frames than cells: use all, pad the rest
        selected: list[Optional[io.BytesIO]] = list(streams)
        selected += [None] * (self._n_cells - n)
        return selected

    def _decode_cells(
        self, slots: list[Optional[io.BytesIO]]
    ) -> list[np.ndarray]:
        """
        Decode each slot into a resized BGR cell.
        ``None`` slots (padding) become solid-black cells.
        Frame labels (T1–T9) and a final-frame marker are drawn in-place.
        """
        cfg = self._cfg
        cells: list[np.ndarray] = []

        for idx, slot in enumerate(slots):
            cell_idx = idx + 1  # 1-based label

            if slot is not None:
                slot.seek(0)
                raw = np.frombuffer(slot.read(), dtype=np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Could not decode JPEG for cell T%d — using black.", cell_idx)
                    frame = self._black_cell()
                else:
                    frame = cv2.resize(
                        frame, (cfg.cell_w, cfg.cell_h), interpolation=cv2.INTER_AREA
                    )
            else:
                frame = self._black_cell()

            self._draw_label(frame, cell_idx, is_final=(cell_idx == self._n_cells and slot is not None))
            cells.append(frame)

        return cells

    def _black_cell(self) -> np.ndarray:
        return np.zeros((self._cfg.cell_h, self._cfg.cell_w, 3), dtype=np.uint8)

    def _draw_label(self, cell: np.ndarray, idx: int, is_final: bool) -> None:
        """Annotate cell with its temporal index and an optional 'LATEST' badge."""
        cfg = self._cfg
        label = f"T{idx}"
        color = (255, 220, 80)   # warm yellow for standard frames
        shadow = (0, 0, 0)

        # Shadow pass (readability over any background)
        cv2.putText(
            cell, label, (9, 25),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.label_font_scale,
            shadow, cfg.label_thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            cell, label, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.label_font_scale,
            color, cfg.label_thickness, cv2.LINE_AA,
        )

        if is_final:
            # Draw a small red "LATEST" badge in the bottom-right corner
            badge = "LATEST"
            bx = cfg.cell_w - 76
            by = cfg.cell_h - 8
            cv2.putText(cell, badge, (bx + 1, by + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(cell, badge, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 80, 255), 1, cv2.LINE_AA)

    def _stitch(self, cells: list[np.ndarray]) -> np.ndarray:
        """Stack cells into rows then stack rows into a full grid."""
        cfg = self._cfg
        rows_imgs: list[np.ndarray] = []
        for r in range(cfg.rows):
            row_cells = cells[r * cfg.cols : (r + 1) * cfg.cols]
            rows_imgs.append(np.hstack(row_cells))
        return np.vstack(rows_imgs)

    def _encode_jpeg(self, grid: np.ndarray) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.storyboard_jpeg_quality]
        ok, buf = cv2.imencode(".jpg", grid, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed for storyboard grid.")
        return buf.tobytes()


# ─── Async Gemini API client ──────────────────────────────────────────────────


class GeminiClient:
    """
    Async Gemini client backed by the new ``google-genai`` SDK
    (``from google import genai``).

    Client lifecycle
    ────────────────
    ``genai.Client`` is a scoped object — no global state is mutated via
    ``configure()``.  Multiple clients can coexist safely in the same process
    (e.g. multiple cameras, unit tests).  The client is cheap to construct
    and reused across all ``analyse()`` calls on this instance.

    Async inference
    ───────────────
    Calls go through ``client.aio.models.generate_content`` — the ``aio``
    sub-namespace exposes native ``async/await`` coroutines, keeping the
    EventManager's dedicated loop fully non-blocking.

    Schema enforcement
    ──────────────────
    ``response_mime_type="application/json"`` plus
    ``response_schema=ViolationAnalysis`` inside ``types.GenerateContentConfig``
    constrains the model's output to a JSON object matching the Pydantic schema.
    The raw text is then validated a second time by Pydantic so all custom
    validators (confidence clamping, ``None`` normalisation) are always applied.

    Retry policy
    ────────────
    Any SDK-level exception (quota, transient server error, network failure)
    is retried up to ``GeminiConfig.max_retries`` times with non-blocking
    exponential back-off via ``asyncio.sleep``.
    """

    def __init__(self, cfg: GeminiConfig) -> None:
        self._cfg = cfg
        # Route all requests through Vertex AI.  Authentication is resolved
        # automatically from Application Default Credentials (ADC) — no API
        # key is passed or stored.
        self._client = genai.Client(
            vertexai=True,
            project=cfg.project,
            location=cfg.location,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    async def analyse(
        self,
        jpeg_bytes: bytes,
        active_rules: list[str],
    ) -> ViolationAnalysis:
        """
        Send the storyboard JPEG to Gemini and return a validated
        ``ViolationAnalysis`` scoped to ``active_rules``.

        Args:
            jpeg_bytes:   Raw JPEG bytes of the 3×3 temporal storyboard grid.
            active_rules: PPE rule IDs enabled for this camera (e.g.
                          ``["hardhat", "vest"]``).  Drives both the prompt
                          and the response_schema so Gemini can only flag
                          violations the factory manager cares about.

        Returns:
            A validated ``ViolationAnalysis`` Pydantic model (actually an
            instance of the dynamic subclass returned by
            ``_build_violation_schema``).

        Raises:
            GeminiAPIError: When all retries are exhausted.
        """
        # types.Part.from_bytes attaches the mime-type explicitly so the SDK
        # never has to guess the format from magic bytes.
        grid_image = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

        prompt = _build_prompt(active_rules)
        scoped_schema = _build_violation_schema(active_rules)

        last_exc: Optional[Exception] = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._cfg.model,
                    contents=[grid_image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        # Scoped subclass: violation_type is a Literal over
                        # exactly the labels allowed for this camera.
                        response_schema=scoped_schema,
                        temperature=self._cfg.temperature,
                    ),
                )
                # Double-validate through Pydantic to enforce custom validators
                # (confidence clamping, None normalisation) plus the dynamic
                # Literal narrowing on violation_type.
                return scoped_schema.model_validate_json(response.text)

            except Exception as exc:  # noqa: BLE001
                delay = self._cfg.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Gemini SDK error (attempt %d/%d): %s. Retrying in %.1f s.",
                    attempt + 1, self._cfg.max_retries + 1, exc, delay,
                )
                last_exc = exc
                if attempt < self._cfg.max_retries:
                    await asyncio.sleep(delay)

        raise GeminiAPIError(
            f"Gemini request failed after {self._cfg.max_retries + 1} attempts."
        ) from last_exc

    async def close(self) -> None:
        """No-op: the SDK manages its own transport lifecycle."""


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API returns an unrecoverable error."""


# ─── Slack alerting ───────────────────────────────────────────────────────────

_CAMERA_LOCATION = "Zone A - Packing Line"


def _send_slack_alert(
    analysis: ViolationAnalysis,
    timestamp: str,
    track_id: int,
    jpeg_bytes: bytes,
) -> None:
    """
    Upload the storyboard image and violation details to Slack via the
    Bot Token API (``slack_sdk.WebClient``).

    Execution model
    ───────────────
    This function is fully synchronous (file I/O + blocking HTTP).  It is
    always scheduled via ``asyncio.create_task(asyncio.to_thread(...))`` so
    the upload never touches the event loop and cannot stall the tracking
    pipeline, even on a slow network.

    Image evidence
    ──────────────
    The storyboard JPEG is written to ``/tmp/violation_{track_id}.jpg``,
    uploaded with ``files_upload_v2``, then deleted from disk regardless of
    whether the upload succeeded or failed.

    Required environment variables
    ───────────────────────────────
    ``SLACK_BOT_TOKEN``  — OAuth Bot Token (``xoxb-…``) with the
                           ``files:write`` and ``chat:write`` scopes.
    ``SLACK_CHANNEL``    — Channel ID or name to post into (e.g. ``C01AB2CD3``
                           or ``#safety-alerts``).

    Failure handling
    ────────────────
    Any SDK or I/O error is caught and logged as a warning so a Slack
    outage can never crash the VLM pipeline.
    """
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    channel   = os.environ.get("SLACK_CHANNEL")
    if not bot_token or not channel:
        logger.warning(
            "SLACK_BOT_TOKEN or SLACK_CHANNEL not set — Slack alert skipped."
        )
        return

    # ── Write storyboard to a temp file ──────────────────────────────────────
    tmp_path = f"/tmp/violation_{track_id}.jpg"
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(jpeg_bytes)
    except OSError as exc:
        logger.warning("Could not write storyboard image to disk: %s", exc)
        return

    # ── Build the message text that accompanies the image ────────────────────
    initial_comment = (
        f"🚨 *Safety Violation Detected* — {_CAMERA_LOCATION}\n"
        f"*Time:* {timestamp}  |  "
        f"*Type:* {analysis.violation_type}  |  "
        f"*Confidence:* {analysis.confidence_score:.0%}\n"
        f"*Reasoning:* {analysis.reasoning}"
    )

    # ── Upload image + metadata in a single API call ──────────────────────────
    client = SlackWebClient(token=bot_token)
    try:
        client.files_upload_v2(
            channel=channel,
            file=tmp_path,
            filename=f"violation_{track_id}.jpg",
            title=f"PPE Violation — {analysis.violation_type}",
            initial_comment=initial_comment,
        )
        logger.info(
            "Slack image alert uploaded for track ID %d (type='%s').",
            track_id,
            analysis.violation_type,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Slack upload failed: %s", exc)
    finally:
        # Always remove the temp file, whether the upload succeeded or not.
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ─── VLM Escalation Handler ───────────────────────────────────────────────────


class VLMEscalationHandler:
    """
    Async callable used as the ``on_event`` callback for ``EventManager``.

    Threading model
    ───────────────
    ``__call__`` is a coroutine.  ``EventManager`` detects this via
    ``asyncio.iscoroutinefunction`` and awaits it directly on its dedicated
    asyncio loop thread — the main video-processing loop is never touched.

    The storyboard build (NumPy + cv2, CPU-bound) is offloaded to a thread
    pool executor so the event loop stays responsive to concurrent events
    from multiple tracked IDs.

    Lifecycle
    ─────────
    Call ``close_sync()`` from the main thread after ``EventManager.shutdown()``
    to chain teardown to any downstream callbacks that own async resources.

    Example
    ───────
    ::

        handler = VLMEscalationHandler(
            gemini_cfg=GeminiConfig(),  # ADC auth; no API key required
            on_analysis=lambda r: print(r.summary()),
        )
        run(..., on_event=handler)
        # close_sync() is called automatically by rtsp_tracker.run()
    """

    def __init__(
        self,
        gemini_cfg: GeminiConfig,
        storyboard_cfg: StoryboardConfig = StoryboardConfig(),
        on_analysis: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> None:
        self._gemini_cfg = gemini_cfg
        self._storyboard = Storyboard(storyboard_cfg)
        self._on_analysis = on_analysis or self._default_on_analysis
        # GeminiClient and the asyncio loop reference are set lazily on first call
        self._client: Optional[GeminiClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Async callable (EventManager calls this) ──────────────────────────────

    async def __call__(self, event: Event) -> None:
        """
        Full async pipeline: storyboard → Gemini SDK → ViolationAnalysis.

        Args:
            event: The fired ``Event`` from ``EventManager``, carrying
                   the list of ``io.BytesIO`` JPEG streams.
        """
        # Lazily bind client to this event loop on first invocation
        if self._client is None:
            self._loop = asyncio.get_running_loop()
            self._client = GeminiClient(self._gemini_cfg)

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        try:
            # ── Build storyboard in thread pool (CPU-bound) ───────────────────
            grid_bgr, jpeg_bytes, b64_str = await loop.run_in_executor(
                None, self._storyboard.build, event.jpeg_streams
            )
            logger.debug(
                "Storyboard built for ID %d: grid=%s  JPEG=%.1f KB",
                event.track_id,
                f"{grid_bgr.shape[1]}×{grid_bgr.shape[0]}",
                len(jpeg_bytes) / 1024,
            )

            # ── Send to Gemini via SDK (async, non-blocking) ──────────────────
            # active_rules narrows both the prompt and the response_schema so
            # Gemini cannot hallucinate violations outside the configured set.
            analysis: ViolationAnalysis = await self._client.analyse(
                jpeg_bytes, event.active_rules
            )

            # ── Slack alert (fire-and-forget) ─────────────────────────────────
            # Scheduled as a concurrent Task so the blocking file write +
            # slack_sdk HTTP upload never stall the event loop.
            # _send_slack_alert runs in a thread pool via asyncio.to_thread;
            # all I/O and network errors are caught inside it and demoted to
            # warnings so a Slack outage cannot crash the VLM pipeline.
            if analysis.violation_detected:
                asyncio.create_task(
                    asyncio.to_thread(
                        _send_slack_alert,
                        analysis,
                        event.triggered_at_str,
                        event.track_id,
                        jpeg_bytes,
                    ),
                    name=f"slack-alert-{event.track_id}",
                )

            latency_ms = (time.perf_counter() - t0) * 1000
            result = AnalysisResult(
                event=event,
                analysis=analysis,
                storyboard_jpeg_bytes=jpeg_bytes,
                api_latency_ms=latency_ms,
                model=self._gemini_cfg.model,
            )

            # ── Dispatch result ───────────────────────────────────────────────
            if asyncio.iscoroutinefunction(self._on_analysis):
                await self._on_analysis(result)  # type: ignore[misc]
            else:
                self._on_analysis(result)

        except GeminiAPIError as exc:
            logger.error("Gemini API error for event ID %d: %s", event.track_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error in VLM pipeline for event ID %d: %s",
                event.track_id, exc,
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close_sync(self) -> None:
        """
        Thread-safe shutdown.

        The ``google-generativeai`` SDK manages its own transport lifecycle;
        there is no session object to drain here.  This method exists solely
        to chain teardown to any downstream ``on_analysis`` callback that
        owns async resources (e.g. ``CloudTelemetry``'s HTTP session).

        Safe to call from the main thread after ``EventManager.shutdown()``.
        Does nothing if the handler was never called.
        """
        if self._client is None:
            return
        logger.info("VLMEscalationHandler closed (SDK transport self-managed).")
        # Chain teardown: if the on_analysis callback (e.g. CloudTelemetry)
        # also owns async resources, let it clean up on the same loop.
        if hasattr(self._on_analysis, "close_sync"):
            self._on_analysis.close_sync()  # type: ignore[union-attr]

    # ── Default analysis dispatcher ───────────────────────────────────────────

    @staticmethod
    def _default_on_analysis(result: AnalysisResult) -> None:
        """
        Log a structured summary.  Replace with a webhook call, database write,
        alarm trigger, or any other downstream action.
        """
        flag_color = "\033[91m" if result.analysis.violation_detected else "\033[92m"
        reset = "\033[0m"
        logger.info(
            "%s┌─ VLM ANALYSIS ────────────────────────────────────────\n"
            "│  Track ID        : %d\n"
            "│  Time            : %s\n"
            "│  Violation       : %s\n"
            "│  Type            : %s\n"
            "│  Confidence      : %.0f%%\n"
            "│  Dwell at trigger: %d frames\n"
            "│  Storyboard      : %.1f KB JPEG\n"
            "│  API latency     : %.0f ms  [%s]\n"
            "└───────────────────────────────────────────────────────%s",
            flag_color,
            result.track_id,
            result.triggered_at_str,
            result.analysis.violation_detected,
            result.analysis.violation_type,
            result.analysis.confidence_score * 100,
            result.event.dwell_frames_at_trigger,
            len(result.storyboard_jpeg_bytes) / 1024,
            result.api_latency_ms,
            result.model,
            reset,
        )
