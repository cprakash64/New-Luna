# syntax=docker/dockerfile:1.6
#
# VeoTrex edge tracker — container image for NVIDIA Jetson Orin Nano.
#
# The image runs the Python tracking pipeline (YOLO11s-Pose via TensorRT,
# ByteTrack, Supabase config poller, VLM escalation) against an RTSP feed.
# All secrets (Supabase keys, Slack tokens, Gemini credentials) are injected
# at runtime via docker-compose's env_file — NONE are baked into this image.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

# ── System packages ──────────────────────────────────────────────────────────
# • python3 / python3-pip      → runtime
# • libgl1, libglib2.0-0, etc. → OpenCV shared-lib dependencies
# • ffmpeg                     → H.264/H.265 decode for RTSP streams
# • ca-certificates            → TLS for Supabase / Gemini / Slack
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
# Model engine files and test videos are NOT baked in — they mount as volumes
# via docker-compose so a single image can serve every camera.
COPY rtsp_tracker.py vlm_escalation.py cloud_telemetry.py ./

# ── Runtime ──────────────────────────────────────────────────────────────────
# Required at run time (supplied via compose env_file):
#   RTSP_URL, CAMERA_ID,
#   SUPABASE_URL, SUPABASE_SERVICE_KEY,
#   SLACK_BOT_TOKEN, SLACK_CHANNEL_ID (optional),
#   GOOGLE_APPLICATION_CREDENTIALS (mounted path; file mounted separately).
#
# tini is PID 1 so SIGTERM from `docker stop` cleanly reaches the Python
# process and the tracker's `finally:` block runs (stops FrameReader,
# EventManager, ConfigManager, closes the httpx/SDK clients).
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sh", "-c", "exec python rtsp_tracker.py \"$RTSP_URL\" --camera-id \"$CAMERA_ID\""]
