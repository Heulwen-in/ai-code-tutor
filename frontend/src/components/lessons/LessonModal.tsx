"use client";

import Link from "next/link";
import { useEffect } from "react";

import { Icon } from "@/components/ui/Icon";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { BUG_CONFIG } from "@/lib/constants";
import type { Lesson } from "@/lib/lessons";
import { statusPct } from "./LessonCard";

type LessonModalProps = {
  lesson: Lesson;
  status?: "started" | "completed";
  onClose: () => void;
};

export function LessonModal({ lesson, status, onClose }: LessonModalProps) {
  const cfg = BUG_CONFIG[lesson.bugType];
  const label = status === "completed" ? "Completed" : status === "started" ? "In progress" : "Not started";

  // Close on Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="ic" style={{ background: cfg.bg, color: cfg.color }} aria-hidden>
            <Icon name={cfg.icon} size={22} />
          </span>
          <div>
            <h3>{lesson.title}</h3>
            <div className="meta">
              {lesson.difficulty} · {lesson.duration}
            </div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="modal-body">
          <p>{lesson.summary}</p>
          <pre className="code-block">{lesson.buggyCode}</pre>

          <div className="conf-head" style={{ marginTop: 4 }}>
            <span>Status</span>
            <span className="pct" style={{ color: cfg.color }}>
              {label}
            </span>
          </div>
          <ProgressBar value={statusPct(status)} gradient={cfg.grad} />

          <Link
            href={`/lessons/${lesson.id}`}
            className="btn btn-gradient"
            style={{ width: "100%", marginTop: 18 }}
          >
            {status === "started" ? "Continue Lesson →" : "Start Lesson →"}
          </Link>
        </div>
      </div>
    </div>
  );
}
