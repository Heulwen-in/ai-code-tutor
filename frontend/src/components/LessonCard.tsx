import Link from "next/link";
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
  const url = "url" in lesson ? lesson.url : null;
  const cardId = "id" in lesson ? lesson.id : lesson.lesson_id;
  const href = url ?? (cardId ? `/lessons/${cardId}` : null);

  const card = (
    <article className={`lesson-card ${compact ? "compact" : ""}`} id={cardId}>
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

  if (href) {
    return (
      <Link href={href} style={{ textDecoration: "none", color: "inherit", display: "block" }}>
        {card}
      </Link>
    );
  }

  return card;
}
