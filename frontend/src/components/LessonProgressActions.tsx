"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { completeLesson, startLesson } from "@/lib/api";
import { getStoredUser } from "@/lib/auth";
import type { User } from "@/lib/types";

type Props = { lessonId: string };

export function LessonProgressActions({ lessonId }: Props) {
  const [user, setUser] = useState<User | null>(null);
  const [status, setStatus] = useState<"idle" | "saving" | "completed" | "error">("idle");

  // Record "started" once when a signed-in user opens the lesson
  useEffect(() => {
    const stored = getStoredUser();
    setUser(stored);
    if (stored) {
      startLesson(lessonId, stored.id).catch(() => {
        /* tracking is best-effort — never block the lesson content */
      });
    }
  }, [lessonId]);

  if (!user) {
    return (
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-muted)" }}>
        <Link href="/auth" style={{ color: "var(--primary)" }}>Sign in</Link> to track
        lesson progress and earn badges.
      </p>
    );
  }

  async function handleComplete() {
    setStatus("saving");
    try {
      await completeLesson(lessonId, user!.id);
      setStatus("completed");
    } catch {
      setStatus("error");
    }
  }

  if (status === "completed") {
    return (
      <p style={{ margin: 0, fontSize: 13, color: "var(--success)", fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
        <Icon name="check_circle" size={16} fill /> Lesson marked as completed — progress saved.
      </p>
    );
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <button
        type="button"
        className="btn btn-gradient"
        onClick={handleComplete}
        disabled={status === "saving"}
      >
        {status === "saving" ? "Saving…" : "Mark as completed"}
      </button>
      {status === "error" && (
        <span style={{ fontSize: 13, color: "var(--danger)" }}>
          Could not save — is the backend running?
        </span>
      )}
    </div>
  );
}
