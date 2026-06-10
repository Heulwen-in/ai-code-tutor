"use client";

import { BugReport } from "./BugReport";
import { CodeEditor } from "./CodeEditor";
import { FeedbackPanel } from "./FeedbackPanel";
import { LessonCard } from "./LessonCard";
import { SkillBadge } from "./SkillBadge";
import { useAnalysis } from "@/lib/useAnalysis";

export function TutorWorkspace() {
  const { code, role, result, isAnalyzing, error, setCode, setRole, runAnalysis } = useAnalysis();

  return (
    <div className="dashboard-grid">
      <section className="tutor-workspace">
        <div className="tutor-controls">
          <div className="segmented-control" aria-label="Feedback role">
            <button className={role === "student" ? "active" : ""} onClick={() => setRole("student")}>
              Student
            </button>
            <button className={role === "worker" ? "active" : ""} onClick={() => setRole("worker")}>
              Worker
            </button>
          </div>
          <button className="primary-button" onClick={runAnalysis} disabled={isAnalyzing}>
            {isAnalyzing ? (
              <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <svg
                  width="16" height="16" viewBox="0 0 16 16"
                  style={{ animation: "spin 0.8s linear infinite" }}
                  fill="none" xmlns="http://www.w3.org/2000/svg"
                >
                  <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.3" strokeWidth="2" />
                  <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                Analyzing...
              </span>
            ) : "Analyze code"}
          </button>
        </div>
        <CodeEditor code={code} onChange={setCode} />
        {error ? <p className="error-message">{error}</p> : null}
      </section>

      <aside className="analysis-side">
        <BugReport result={result} />
        <SkillBadge result={result} />
        <FeedbackPanel result={result} />
        <section className="panel">
          <p className="section-label">Recommended Lessons</p>
          <div className="lesson-list">
            {(result?.lessons ?? []).map((lesson) => (
              <LessonCard key={lesson.lesson_id} lesson={lesson} compact />
            ))}
            {!result ? <p className="muted">Run analysis to receive lesson recommendations.</p> : null}
          </div>
        </section>
      </aside>
    </div>
  );
}
