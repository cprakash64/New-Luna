export type RuleId = "hardhat" | "vest" | "glasses";

export const RULES: { id: RuleId; label: string; hint: string }[] = [
  { id: "hardhat", label: "Require Hard Hat", hint: "Flags workers with no helmet on the head." },
  { id: "vest", label: "Require High-Vis Vest", hint: "Flags missing or improperly worn hi-vis vests." },
  { id: "glasses", label: "Require Safety Glasses", hint: "Flags missing eye protection (best-effort)." },
];

type Props = {
  active: RuleId[];
  onChange: (active: RuleId[]) => void;
};

export default function RuleToggles({ active, onChange }: Props) {
  function toggle(id: RuleId) {
    onChange(active.includes(id) ? active.filter((r) => r !== id) : [...active, id]);
  }

  return (
    <div className="rules">
      {RULES.map((r) => {
        const on = active.includes(r.id);
        return (
          <label key={r.id} className={`rule ${on ? "rule-on" : ""}`}>
            <span className="rule-text">
              <span className="rule-label">{r.label}</span>
              <span className="rule-hint">{r.hint}</span>
            </span>
            <span className="switch" aria-hidden>
              <input
                type="checkbox"
                checked={on}
                onChange={() => toggle(r.id)}
              />
              <span className="switch-track">
                <span className="switch-thumb" />
              </span>
            </span>
          </label>
        );
      })}
    </div>
  );
}
