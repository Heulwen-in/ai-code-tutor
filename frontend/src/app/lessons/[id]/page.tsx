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

export default async function LessonDetailPage({ params }: Props) {
  const { id } = await params;
  const lesson = LESSONS[id];

  if (!lesson) notFound();

  return (
    <AppShell>
      <Link href="/lessons" className="lesson-back">
        <Icon name="arrow_back" size={16} /> Back to lessons
      </Link>

      <div className="lesson-layout">
        {/* One unified card with hairline-divided sections */}
        <div className="card lesson-main">
          {/* Header */}
          <div className="lesson-section">
            <span className={LEVEL_CLASS[lesson.difficulty] ?? "badge"}>{lesson.difficulty}</span>
            <h1 className="lesson-title">{lesson.title}</h1>
            <p className="lesson-intro">{lesson.explanation}</p>
          </div>

          {/* Buggy + corrected code, side by side */}
          <div className="lesson-section">
            <div className="code-duo">
              <div className="code-col">
                <p className="lesson-card-label danger">
                  <Icon name="cancel" size={16} fill /> Buggy code
                </p>
                <pre className="code-block buggy">{lesson.buggyCode}</pre>
              </div>
              <div className="code-col">
                <p className="lesson-card-label success">
                  <Icon name="check_circle" size={16} fill /> Corrected code
                </p>
                <pre className="code-block fixed">{lesson.fixedCode}</pre>
              </div>
            </div>
          </div>

          {/* Key takeaways */}
          <div className="lesson-section">
            <p className="lesson-card-label">Key takeaways</p>
            <ul className="takeaways">
              {lesson.takeaways.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ul>
          </div>

          {/* Bug-fit practice exercise */}
          <div className="lesson-section">
            <p className="lesson-card-label primary">
              <Icon name="fitness_center" size={16} /> Try it yourself
            </p>
            <p className="exercise-prompt">{lesson.exercise.prompt}</p>
            <pre className="code-block">{lesson.exercise.starter}</pre>
            <Link href="/analyze" className="btn btn-gradient lesson-cta">
              Open the Analyser →
            </Link>
          </div>
        </div>

        {/* Sticky sidebar */}
        <aside className="lesson-aside">
          <div className="card lesson-card aside-progress">
            <p className="lesson-card-label">Your progress</p>
            <LessonProgressActions lessonId={id} />
          </div>

          <div className="card lesson-card aside-meta">
            <p className="lesson-card-label">In this lesson</p>
            <ul className="aside-meta-list">
              <li>
                <Icon name="signal_cellular_alt" size={18} />
                <span>Level</span>
                <b>{lesson.difficulty}</b>
              </li>
              <li>
                <Icon name="lightbulb" size={18} />
                <span>Key takeaways</span>
                <b>{lesson.takeaways.length}</b>
              </li>
              <li>
                <Icon name="fitness_center" size={18} />
                <span>Practice</span>
                <b>Included</b>
              </li>
            </ul>
          </div>
        </aside>
      </div>
    </AppShell>
  );
}

// Tell Next.js which static paths to pre-render
export function generateStaticParams() {
  return Object.keys(LESSONS).map((id) => ({ id }));
}
