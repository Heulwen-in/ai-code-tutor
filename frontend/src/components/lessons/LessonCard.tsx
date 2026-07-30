import { Icon } from "@/components/ui/Icon";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { BUG_CONFIG } from "@/lib/constants";
import type { Lesson } from "@/lib/lessons";

const DIFF_CLASS: Record<string, string> = {
  beginner: "badge badge-beginner",
  intermediate: "badge badge-intermediate",
  advanced: "badge badge-advanced",
};

type LessonStatus = "started" | "completed" | undefined;

const STATUS_LABEL: Record<string, string> = {
  completed: "Completed",
  started: "In progress",
};

// Map a persisted status to a bar percentage for a consistent visual.
export function statusPct(status: LessonStatus): number {
  return status === "completed" ? 100 : status === "started" ? 55 : 0;
}

type LessonCardProps = {
  lesson: Lesson;
  status?: LessonStatus;
  onOpen: (lesson: Lesson) => void;
};

export function LessonCard({ lesson, status, onOpen }: LessonCardProps) {
  const cfg = BUG_CONFIG[lesson.bugType];

  return (
    <button className="card lesson-card" onClick={() => onOpen(lesson)}>
      <div className="head">
        <span className="ic" style={{ background: cfg.bg, color: cfg.color }} aria-hidden>
          <Icon name={cfg.icon} size={20} />
        </span>
        <div style={{ minWidth: 0 }}>
          <h3>{lesson.title}</h3>
          <p className="desc">{lesson.summary}</p>
        </div>
      </div>
      <div className="lesson-badges">
        <span className={DIFF_CLASS[lesson.difficulty] ?? "badge"}>{lesson.difficulty}</span>
        <span className="badge">{lesson.duration}</span>
      </div>
      <ProgressBar value={statusPct(status)} gradient={cfg.grad} />
      <div className="pct">{status ? STATUS_LABEL[status] : "Not started"}</div>
    </button>
  );
}
