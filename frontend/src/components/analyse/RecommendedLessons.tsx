import Link from "next/link";

import { Icon } from "@/components/ui/Icon";
import type { LessonItem } from "@/lib/types";

// Material Symbols icon per difficulty.
const ICON: Record<string, string> = {
  beginner: "eco",
  intermediate: "trending_up",
  advanced: "rocket_launch",
};

export function RecommendedLessons({ lessons }: { lessons: LessonItem[] }) {
  if (lessons.length === 0) return null;

  return (
    <div className="result-card">
      <p className="section-label">Recommended Lessons</p>
      {lessons.slice(0, 2).map((lesson) => (
        <Link key={lesson.lesson_id} href={`/lessons/${lesson.lesson_id}`} className="rec-lesson">
          <Icon name={ICON[lesson.difficulty] ?? "menu_book"} size={20} className="ic" />

          <span className="meta">
            <strong>{lesson.title}</strong>
            <span style={{ textTransform: "capitalize" }}>{lesson.difficulty}</span>
          </span>
          <span className="chev" aria-hidden>
            ›
          </span>
        </Link>
      ))}
    </div>
  );
}
