import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";
import ROIBuilder, { type Point } from "../components/ROIBuilder";
import RuleToggles, { type RuleId, RULES } from "../components/RuleToggles";

const SAMPLE_IMAGE =
  import.meta.env.VITE_SAMPLE_IMAGE_URL || "/sample-camera.jpg";

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "saving" }
  | { kind: "ok"; msg: string }
  | { kind: "err"; msg: string };

export default function CameraConfig() {
  const [cameraId, setCameraId] = useState("cam-01");
  const [points, setPoints] = useState<Point[]>([]);
  const [active, setActive] = useState<RuleId[]>(["hardhat", "vest"]);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  // Fetch the existing row for this camera_id so the dashboard opens on the
  // live deployed config rather than an empty canvas.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setStatus({ kind: "loading" });
      const { data, error } = await supabase
        .from("camera_configs")
        .select("roi_polygon, active_rules")
        .eq("camera_id", cameraId)
        .maybeSingle();

      if (cancelled) return;
      if (error) {
        setStatus({ kind: "err", msg: error.message });
        return;
      }
      if (data) {
        const poly = (data.roi_polygon ?? []) as Point[];
        const rules = (data.active_rules ?? []) as string[];
        setPoints(poly.map((p) => [Number(p[0]), Number(p[1])] as Point));
        setActive(
          rules.filter((r): r is RuleId => RULES.some((spec) => spec.id === r)),
        );
        setStatus({ kind: "idle" });
      } else {
        setStatus({ kind: "idle" });
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  async function deploy() {
    if (points.length < 3) {
      setStatus({ kind: "err", msg: "ROI needs at least 3 vertices." });
      return;
    }
    if (!cameraId.trim()) {
      setStatus({ kind: "err", msg: "Camera ID is required." });
      return;
    }
    setStatus({ kind: "saving" });
    const { error } = await supabase.from("camera_configs").upsert(
      {
        camera_id: cameraId.trim(),
        roi_polygon: points,
        active_rules: active,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "camera_id" },
    );
    if (error) {
      setStatus({ kind: "err", msg: error.message });
      return;
    }
    setStatus({
      kind: "ok",
      msg: `Deployed to ${cameraId.trim()} — edge agent will hot-reload within 60s.`,
    });
  }

  const canDeploy =
    points.length >= 3 && cameraId.trim().length > 0 && status.kind !== "saving";

  return (
    <div className="shell">
      <header className="bar">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">VEOTREX</span>
          <span className="brand-sub">Camera Configuration</span>
        </div>
        <div className="camera-id-field">
          <label htmlFor="camera-id">CAMERA_ID</label>
          <input
            id="camera-id"
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            spellCheck={false}
          />
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>Region of Interest</h2>
          <p className="sub">
            Click inside the frame to place polygon vertices. Three or more
            points define a closed zone — the tracker fires events only when
            a person dwells inside this region.
          </p>
          <ROIBuilder
            imageUrl={SAMPLE_IMAGE}
            points={points}
            onChange={setPoints}
          />
        </section>

        <section className="panel">
          <h2>PPE Rules</h2>
          <p className="sub">
            Only enabled rules are evaluated. The VLM is explicitly instructed
            to ignore disabled rules so it cannot hallucinate violations
            outside the configured scope.
          </p>
          <RuleToggles active={active} onChange={setActive} />

          <div className="deploy">
            <button
              type="button"
              className="deploy-btn"
              onClick={deploy}
              disabled={!canDeploy}
            >
              {status.kind === "saving" ? "DEPLOYING…" : "DEPLOY TO CAMERA"}
            </button>
            <StatusLine status={status} />
          </div>
        </section>
      </main>
    </div>
  );
}

function StatusLine({ status }: { status: Status }) {
  if (status.kind === "idle") return null;
  if (status.kind === "loading")
    return <div className="status status-info">Loading existing config…</div>;
  if (status.kind === "saving")
    return <div className="status status-info">Writing to Supabase…</div>;
  if (status.kind === "ok")
    return <div className="status status-ok">✓ {status.msg}</div>;
  return <div className="status status-err">✗ {status.msg}</div>;
}
