import type { AnalyzeResponse } from "@/lib/types";

type BugReportProps = {
  result: AnalyzeResponse | null;
};

const bugLabels: Record<string, string> = {
  syntax_error: "Syntax error",
  indentation_error: "Indentation error",
  logic_error: "Logic error",
  variable_misuse: "Variable misuse",
  no_bug: "No bug detected"
};

export function BugReport({ result }: BugReportProps) {
  if (!result) {
    return (
      <section className="panel empty-state">
        <h3>Bug Report</h3>
        <p>Run an analysis to see detected bug type, confidence, and line number.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Bug Report</p>
          <h3>{bugLabels[result.bug.bug_type]}</h3>
        </div>
        <strong>{Math.round(result.bug.confidence * 100)}%</strong>
      </div>
      <p className="muted">{result.bug.description || "The analyzer returned a classification."}</p>
      <div className="metric-row">
        <span>Line</span>
        <strong>{result.bug.line_number ?? "N/A"}</strong>
      </div>
    </section>
  );
}
