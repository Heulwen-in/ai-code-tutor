import { Icon } from "@/components/ui/Icon";

type Step = { icon: string; title: string; detail: string; parallel?: boolean };

const STEPS: Step[] = [
  { icon: "account_tree", title: "AST parse gate", detail: "Owns indentation — exact line, confidence 1.0" },
  { icon: "category", title: "Stage 1 · 4-class", detail: "CodeBERT coarse bug type + no-bug gate (0.60)" },
  { icon: "zoom_in", title: "Stage 2 · 14 subtypes", detail: "Masked to Stage 1 group + subtype gate (0.65)" },
  { icon: "my_location", title: "Line localiser", detail: "Token classification + line gate (0.50)" },
  { icon: "school", title: "Skill detector", detail: "Novice / Professional — runs in parallel", parallel: true },
  { icon: "forum", title: "Role feedback", detail: "Qwen2.5-Coder via Ollama, honest error path" },
  { icon: "menu_book", title: "Lesson recommender", detail: "Bug type + role → targeted lessons" },
  { icon: "autorenew", title: "Learning loop", detail: "Progress + Leitner spaced repetition" },
];

// Hand-built CSS flow of the inference pipeline (no image dependency).
export function PipelineDiagram() {
  return (
    <div className="proc-flow" role="img" aria-label="Inference pipeline: AST gate, Stage 1, Stage 2, line localiser, skill detector, feedback, lessons, learning loop">
      {STEPS.map((s, i) => (
        <div className="proc-flow-item" key={s.title}>
          <div className={s.parallel ? "proc-flow-step parallel" : "proc-flow-step"}>
            <span className="proc-flow-icon" aria-hidden>
              <Icon name={s.icon} size={22} />
            </span>
            <strong>{s.title}</strong>
            <span>{s.detail}</span>
          </div>
          {i < STEPS.length - 1 && (
            <span className="proc-flow-arrow material-symbols-rounded" aria-hidden>
              arrow_forward
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
