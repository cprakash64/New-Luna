import { useState, type MouseEvent } from "react";

export type Point = [number, number];

type Props = {
  imageUrl: string;
  points: Point[];
  onChange: (points: Point[]) => void;
};

/**
 * Clickable image canvas. Each click appends a vertex in the image's native
 * (natural) pixel coordinate space, so the persisted polygon is resolution-
 * independent regardless of how the <img> is laid out on screen.
 *
 * The wrapper hugs the image (inline-block, no fixed size) and the SVG overlay
 * stretches to its exact bounds, so currentTarget.getBoundingClientRect() on
 * the wrapper is identical to the image's rendered rect.
 */
export default function ROIBuilder({ imageUrl, points, onChange }: Props) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);

  function handleClick(e: MouseEvent<HTMLDivElement>) {
    if (!natural) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = natural.w / rect.width;
    const scaleY = natural.h / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    if (x < 0 || y < 0 || x > natural.w || y > natural.h) return;
    onChange([...points, [x, y]]);
  }

  function undo() {
    onChange(points.slice(0, -1));
  }

  function clear() {
    onChange([]);
  }

  const polyPoints = points.map((p) => p.join(",")).join(" ");
  const closed = points.length >= 3;

  return (
    <div className="roi">
      <div className="roi-canvas" onClick={handleClick}>
        <img
          src={imageUrl}
          alt="camera frame"
          draggable={false}
          onLoad={(e) => {
            const el = e.currentTarget;
            setNatural({ w: el.naturalWidth, h: el.naturalHeight });
          }}
        />
        {natural && (
          <svg
            className="roi-overlay"
            viewBox={`0 0 ${natural.w} ${natural.h}`}
            preserveAspectRatio="none"
          >
            {closed && (
              <polygon
                points={polyPoints}
                fill="rgba(0, 230, 150, 0.18)"
                stroke="#00e696"
                strokeWidth={Math.max(natural.w, natural.h) / 400}
              />
            )}
            {!closed && points.length > 1 && (
              <polyline
                points={polyPoints}
                fill="none"
                stroke="#00e696"
                strokeWidth={Math.max(natural.w, natural.h) / 400}
              />
            )}
            {points.map(([x, y], i) => (
              <g key={i}>
                <circle
                  cx={x}
                  cy={y}
                  r={Math.max(natural.w, natural.h) / 200}
                  fill="#00e696"
                  stroke="#0b0d10"
                  strokeWidth={Math.max(natural.w, natural.h) / 600}
                />
                <text
                  x={x + Math.max(natural.w, natural.h) / 120}
                  y={y - Math.max(natural.w, natural.h) / 160}
                  fill="#00e696"
                  fontSize={Math.max(natural.w, natural.h) / 50}
                  fontFamily="ui-monospace, monospace"
                >
                  {i + 1}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>

      <div className="roi-controls">
        <div className="roi-hint">
          {points.length === 0
            ? "Click the frame to place polygon vertices."
            : points.length < 3
              ? `${points.length} / 3 minimum vertices placed.`
              : `${points.length} vertices — polygon closed.`}
        </div>
        <div className="roi-buttons">
          <button type="button" onClick={undo} disabled={points.length === 0}>
            Undo
          </button>
          <button type="button" onClick={clear} disabled={points.length === 0}>
            Clear
          </button>
        </div>
      </div>

      {points.length > 0 && (
        <pre className="roi-coords">
          {JSON.stringify(points)}
        </pre>
      )}
    </div>
  );
}
