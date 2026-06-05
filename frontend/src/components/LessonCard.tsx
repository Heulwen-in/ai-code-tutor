import type { LessonItem, LessonModule } from "@/lib/types";

type LessonCardProps = {
  lesson: LessonItem | LessonModule;
  compact?: boolean;
};

export function LessonCard({ lesson, compact = false }: LessonCardProps) {
  const title = "title" in lesson ? lesson.title : "Lesson";
  const description = "description" in lesson ? lesson.description : lesson.summary;
  const difficulty = "difficulty" in lesson ? lesson.difficulty : lesson.level;
  const progress = "progress" in lesson ? lesson.progress : null;

  return (
    <article className={`lesson-card ${compact ? "compact" : ""}`} id={"id" in lesson ? lesson.id : lesson.lesson_id}>
      <div className="lesson-topline">
        <span>{difficulty}</span>
        {"duration" in lesson ? <span>{lesson.duration}</span> : null}
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      {progress !== null ? (
        <div className="progress-track" aria-label={`${progress}% complete`}>
          <span style={{ width: `${progress}%` }} />
        </div>
      ) : null}
    </article>
  );
}
