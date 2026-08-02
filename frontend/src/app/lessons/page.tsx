"use client";

import { useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { LessonCard, statusPct } from "@/components/lessons/LessonCard";
import { LessonModal } from "@/components/lessons/LessonModal";
import { Icon } from "@/components/ui/Icon";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { getLessonProgress, type LessonStatusMap } from "@/lib/api";
import { getStoredUser } from "@/lib/auth";
import { BUG_CONFIG } from "@/lib/constants";
import { LESSON_LIST, type Lesson } from "@/lib/lessons";
import { useRole } from "@/lib/roleStore";

export default function LessonsPage() {
  const { role } = useRole();
  const [active, setActive] = useState<Lesson | null>(null);
  const [statuses, setStatuses] = useState<LessonStatusMap>({});
  const [showAll, setShowAll] = useState(false);

  // Real per-lesson status for the signed-in user (fail-soft in demo/offline).
  // Refetch when the tab/window regains focus so returning from a lesson (even
  // from the router cache) always reflects the latest completion state.
  useEffect(() => {
    const u = getStoredUser();
    if (!u) return;
    const refresh = () => getLessonProgress(u.id).then(setStatuses).catch(() => {});
    refresh();
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const visible = useMemo(
    () => (showAll ? LESSON_LIST : LESSON_LIST.filter((l) => l.role === role)),
    [role, showAll],
  );

  const completed = Object.values(statuses).filter((s) => s === "completed").length;
  const inProgress = Object.values(statuses).filter((s) => s === "started").length;

  // Feature the first in-progress lesson in the current view, else the first.
  const featured = visible.find((l) => statuses[l.id] === "started") ?? visible[0];
  const featuredCfg = featured ? BUG_CONFIG[featured.bugType] : null;

  return (
    <AppShell>
      <div className="page-heading">
        <h1>Learning Library</h1>
        <p>
          {visible.length} lessons · {completed} completed · {inProgress} in progress
        </p>
      </div>

      <div className="lessons-toolbar">
        <div className="seg-toggle" role="group" aria-label="Lesson track">
          <button className={!showAll ? "active" : ""} onClick={() => setShowAll(false)}>
            <Icon name={role === "worker" ? "work" : "school"} size={16} />
            {role === "worker" ? "Professional" : "Student"}
          </button>
          <button className={showAll ? "active" : ""} onClick={() => setShowAll(true)}>
            All lessons
          </button>
        </div>
      </div>

      {featured && featuredCfg && (
        <section className="lessons-featured">
          <span className="ic" aria-hidden>
            <Icon name={featuredCfg.icon} size={26} />
          </span>
          <div className="body">
            <div className="eyebrow">
              {statuses[featured.id] === "started" ? "CONTINUE" : "START HERE"}
            </div>
            <h3>{featured.title}</h3>
            <ProgressBar value={statusPct(statuses[featured.id])} />
          </div>
          <button className="btn btn-outline" onClick={() => setActive(featured)}>
            Open →
          </button>
        </section>
      )}

      <div className="lesson-grid">
        {visible.map((lesson) => (
          <LessonCard
            key={lesson.id}
            lesson={lesson}
            status={statuses[lesson.id]}
            onOpen={setActive}
          />
        ))}
      </div>

      {active && (
        <LessonModal lesson={active} status={statuses[active.id]} onClose={() => setActive(null)} />
      )}
    </AppShell>
  );
}
