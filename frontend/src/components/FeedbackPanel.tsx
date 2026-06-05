import type { AnalyzeResponse } from "@/lib/types";

type FeedbackPanelProps = {
  result: AnalyzeResponse | null;
};

export function FeedbackPanel({ result }: FeedbackPanelProps) {
  if (!result) {
    return (
      <section className="panel empty-state">
        <h3>AI Feedback</h3>
        <p>Feedback will be personalised by role once code is analyzed.</p>
      </section>
    );
  }

  return (
    <section className="panel feedback-panel">
      <p className="section-label">AI Feedback</p>
      <h3>{result.feedback.summary}</h3>
      <p>{result.feedback.explanation}</p>
      <ul className="next-steps">
        {result.feedback.next_steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ul>
    </section>
  );
}
