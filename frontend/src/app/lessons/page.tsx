import { AppShell } from "@/components/AppShell";
import { LessonCard } from "@/components/LessonCard";
import { lessonModules } from "@/lib/mock-data";

export default function LessonsPage() {
  return (
    <AppShell
      title="Lesson Learning"
      description="Short W3Schools-style lessons linked to the bug types found by the analyzer."
    >
      <section className="panel">
        <p className="section-label">Learning Path</p>
        <h3>Recommended modules</h3>
        <p className="muted">
          Each lesson will later include explanation, runnable examples, quiz questions, and practice tasks.
        </p>
      </section>
      <div className="lesson-list" style={{ marginTop: 18 }}>
        {lessonModules.map((lesson) => (
          <LessonCard key={lesson.id} lesson={lesson} />
        ))}
      </div>
      <section className="panel" style={{ marginTop: 18 }}>
        <p className="section-label">Example Lesson Structure</p>
        <h3>Python indentation rules</h3>
        <p>
          Python groups code using indentation. After statements such as <strong>def</strong>,{" "}
          <strong>if</strong>, <strong>for</strong>, and <strong>while</strong>, the next block must be
          indented consistently.
        </p>
        <pre className="preview-code">if user_score &gt;= 50:{"\n"}    print("Passed")</pre>
      </section>
    </AppShell>
  );
}
