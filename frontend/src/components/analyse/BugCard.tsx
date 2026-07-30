import { BUG_CONFIG } from "@/lib/constants";
import type { AnalyzeResponse } from "@/lib/types";

// "swapped_comparison_operands" → "Swapped Comparison Operands"
function humanizeSubtype(subtype: string): string {
  return subtype
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function BugCard({ bug }: { bug: AnalyzeResponse["bug"] }) {
  const cfg = BUG_CONFIG[bug.bug_type];
  const pct = Math.round(bug.confidence * 100);

  return (
    <div className="result-card">
      <p className="section-label">Bug Detected</p>
      <div className="result-row">
        <span className="bug-badge" style={{ background: cfg.bg, color: cfg.color }}>
          {bug.bug_type}
        </span>
        {bug.line_number != null && <span className="line-ref">Line {bug.line_number}</span>}
      </div>

      {/* Stage-2 fine-grained subtype (absent for no_bug / indentation / Stage 2 off) */}
      {bug.bug_subtype && (
        <div className="subtype-row">
          <span className="subtype-key">Subtype</span>
          <span className="subtype-val" style={{ color: cfg.color }}>
            {humanizeSubtype(bug.bug_subtype)}
          </span>
          {bug.subtype_confidence != null && (
            <span className="subtype-conf">· {Math.round(bug.subtype_confidence * 100)}%</span>
          )}
        </div>
      )}

      <p className="desc">{bug.description}</p>

      <div className="conf-head">
        <span>Confidence</span>
        <span className="pct" style={{ color: cfg.color }}>
          {pct}%
        </span>
      </div>
      <div className="progress-bar">
        <span style={{ width: `${pct}%`, background: cfg.grad }} />
      </div>

      <p className="card-footer">CodeBERT (microsoft/codebert-base) · Macro F1: 0.9556</p>
    </div>
  );
}
