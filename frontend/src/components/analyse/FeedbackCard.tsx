import type { AnalyzeResponse } from "@/lib/types";

export function FeedbackCard({ feedback }: { feedback: AnalyzeResponse["feedback"] }) {
  return (
    <div className="result-card">
      <p className="section-label">Personalised Feedback</p>
      <p className="desc" style={{ marginTop: 4 }}>
        {feedback.explanation || feedback.summary}
      </p>

      {feedback.next_steps.length > 0 && (
        <>
          <p className="next-steps-title">NEXT STEPS</p>
          <ul className="next-steps">
            {feedback.next_steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
