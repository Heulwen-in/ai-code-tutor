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
            {isAnalyzing ? "Analyzing..." : "Analyze code"}
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
