import Link from "next/link";
import { notFound } from "next/navigation";

import { AppShell } from "@/components/AppShell";
import { LessonProgressActions } from "@/components/LessonProgressActions";
import { Icon } from "@/components/ui/Icon";
import { LESSONS } from "@/lib/lessons";

type Props = { params: Promise<{ id: string }> };

const LEVEL_CLASS: Record<string, string> = {
  beginner: "badge badge-beginner",
  intermediate: "badge badge-intermediate",
  advanced: "badge badge-advanced",
};

const labelStyle = { display: "inline-flex", alignItems: "center", gap: 6 } as const;

export default async function LessonDetailPage({ params }: Props) {
  const { id } = await params;
  const lesson = LESSONS[id];

  if (!lesson) notFound();

  return (
    <AppShell>
      <div className="lesson-detail">
        <Link
          href="/lessons"
          style={{ display: "inline-block", marginBottom: 16, color: "var(--primary)", fontSize: 14, fontWeight: 600 }}
        >
          ← Back to lessons
        </Link>

        {/* Header */}
        <div className="page-heading">
          <span className={LEVEL_CLASS[lesson.difficulty] ?? "badge"} style={{ marginBottom: 10 }}>
            {lesson.difficulty}
          </span>
          <h1 style={{ marginTop: 8 }}>{lesson.title}</h1>
          <p>{lesson.explanation}</p>
        </div>

        {/* Buggy code */}
        <section className="card settings-panel" style={{ marginBottom: 16 }}>
          <p className="section-label" style={{ ...labelStyle, color: "var(--danger)" }}>
            <Icon name="cancel" size={16} fill /> Buggy code
          </p>
          <pre className="code-block buggy">{lesson.buggyCode}</pre>
        </section>

        {/* Fixed code */}
        <section className="card settings-panel" style={{ marginBottom: 16 }}>
          <p className="section-label" style={{ ...labelStyle, color: "var(--success)" }}>
            <Icon name="check_circle" size={16} fill /> Corrected code
          </p>
          <pre className="code-block fixed">{lesson.fixedCode}</pre>
        </section>

        {/* Key takeaways */}
        <section className="card settings-panel" style={{ marginBottom: 16 }}>
          <p className="section-label">Key takeaways</p>
          <ul className="takeaways">
            {lesson.takeaways.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </section>

        {/* Bug-fit practice exercise */}
        <section className="card settings-panel" style={{ marginBottom: 16 }}>
          <p className="section-label" style={{ ...labelStyle, color: "var(--primary)" }}>
            <Icon name="fitness_center" size={16} /> Try it yourself
          </p>
          <p style={{ margin: "0 0 12px", color: "var(--text-body)", fontSize: 14 }}>
            {lesson.exercise.prompt}
          </p>
          <pre className="code-block">{lesson.exercise.starter}</pre>
          <Link href="/analyze" className="btn btn-outline" style={{ marginTop: 14 }}>
            Open the Analyser →
          </Link>
        </section>

        {/* Progress tracking */}
        <section className="card settings-panel">
          <p className="section-label">Your progress</p>
          <LessonProgressActions lessonId={id} />
        </section>
      </div>
    </AppShell>
  );
}

// Tell Next.js which static paths to pre-render
export function generateStaticParams() {
  return Object.keys(LESSONS).map((id) => ({ id }));
}
