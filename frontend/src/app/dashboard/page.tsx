"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { Icon } from "@/components/ui/Icon";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { StatCard } from "@/components/ui/StatCard";
import { statusPct } from "@/components/lessons/LessonCard";
import {
  getAchievements,
  getLessonProgress,
  getReviews,
  getUserProgress,
  getUserStats,
  type Achievement,
  type LessonStatusMap,
  type ReviewItem,
  type UserProgress,
  type UserStats,
} from "@/lib/api";
import { getStoredUser } from "@/lib/auth";
import { BUG_CONFIG } from "@/lib/constants";
import { LESSON_LIST } from "@/lib/lessons";
import { useRole } from "@/lib/roleStore";

export default function DashboardPage() {
  const { role } = useRole();
  const [stats, setStats] = useState<UserStats | null>(null);
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [lessonStatus, setLessonStatus] = useState<LessonStatusMap>({});
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [reviews, setReviews] = useState<ReviewItem[]>([]);

  useEffect(() => {
    Promise.all([getUserStats(), getUserProgress(), getAchievements(), getReviews()])
      .then(([s, p, a, r]) => {
        setStats(s);
        setProgress(p);
        setAchievements(a);
        setReviews(r);
      })
      .catch(console.error);
    const u = getStoredUser();
    if (u) getLessonProgress(u.id).then(setLessonStatus).catch(() => {});
  }, []);

  const user = getStoredUser();
  const name = user?.display_name || user?.email?.split("@")[0] || "Gia Han";
  const roleLabel = role === "worker" ? "Professional" : "Student";
  const badgesEarned = achievements.filter((a) => a.status === "earned").length;

  // Continue-learning: role-appropriate lessons, in-progress first.
  const continueList = LESSON_LIST.filter((l) => l.role === role)
    .sort((a, b) => (lessonStatus[b.id] === "started" ? 1 : 0) - (lessonStatus[a.id] === "started" ? 1 : 0))
    .slice(0, 2);

  return (
    <AppShell>
      <section className="welcome-banner">
        <div className="hello">
          Welcome back <Icon name="waving_hand" size={16} fill />
        </div>
        <h1>{name}</h1>
        <div className="banner-pills">
          <span className="banner-pill">
            <Icon name="local_fire_department" size={15} fill /> {stats?.day_streak ?? 0}-day streak!
          </span>
          <span className="banner-pill">
            <Icon name="bolt" size={15} fill /> {stats?.xp ?? 0} XP
          </span>
          <span className="banner-pill">
            <Icon name={role === "worker" ? "work" : "school"} size={15} /> {roleLabel}
          </span>
        </div>
      </section>

      <div className="stat-grid">
        <StatCard value={stats?.total ?? 0} label="Analyses" trend={`${stats?.xp ?? 0} XP total`} accent="var(--primary)" />
        <StatCard
          value={progress?.lessons_completed ?? 0}
          label="Lessons Done"
          trend={`${progress?.lessons_started ?? 0} started`}
          accent="var(--primary-violet)"
        />
        <StatCard
          value={stats?.day_streak ?? 0}
          label="Day Streak"
          trend={stats && stats.day_streak > 0 ? "Keep it going!" : "Start today"}
          accent="var(--warning)"
        />
        <StatCard
          value={badgesEarned}
          label="Badges Earned"
          trend={`${Math.max(0, achievements.length - badgesEarned)} to go`}
          accent="var(--success)"
        />
      </div>

      <div className="dash-grid">
        <section className="card panel-card">
          <h3>Continue Learning</h3>
          {continueList.map((lesson) => {
            const cfg = BUG_CONFIG[lesson.bugType];
            const pct = statusPct(lessonStatus[lesson.id]);
            return (
              <Link
                key={lesson.id}
                href={`/lessons/${lesson.id}`}
                className="continue-item"
                style={{ color: "inherit" }}
              >
                <span className="ic" style={{ background: cfg.bg, color: cfg.color }} aria-hidden>
                  <Icon name={cfg.icon} size={18} />
                </span>
                <div className="body">
                  <strong>{lesson.title}</strong>
                  <div className="sub">
                    {lessonStatus[lesson.id] === "completed"
                      ? "Completed"
                      : lessonStatus[lesson.id] === "started"
                        ? "In progress"
                        : `Not started · ${lesson.duration}`}
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <ProgressBar value={pct} gradient={cfg.grad} />
                  </div>
                </div>
              </Link>
            );
          })}
        </section>

        <section className="card panel-card">
          <h3>Quick Actions</h3>
          <Link href="/analyze" className="quick-action primary" style={{ color: "#fff" }}>
            <span className="ic" aria-hidden>
              <Icon name="bolt" size={20} fill />
            </span>
            <div>
              <strong>Analyse New Code</strong>
              <span>Paste Python and get feedback</span>
            </div>
          </Link>
          <Link href="/progress" className="quick-action secondary" style={{ color: "inherit" }}>
            <span className="ic" aria-hidden>
              <Icon name="insights" size={20} />
            </span>
            <div>
              <strong>View Progress</strong>
              <span>{stats?.xp ?? 0} XP earned so far</span>
            </div>
          </Link>
        </section>
      </div>

      <section className="card panel-card review-card">
        <div className="review-head">
          <h3>
            <Icon name="autorenew" size={18} /> Review Queue
          </h3>
          <span className="sub">Spaced repetition — bugs you made come back for review</span>
        </div>

        {reviews.length === 0 ? (
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Nothing scheduled yet — when the analyser finds a bug, that concept is
            queued for review the next day.
          </p>
        ) : (
          reviews.slice(0, 4).map((r) => {
            const cfg = BUG_CONFIG[r.bug_type as keyof typeof BUG_CONFIG];
            if (!cfg) return null;
            return (
              <div className="review-item" key={r.bug_type}>
                <span className="ic" style={{ background: cfg.bg, color: cfg.color }} aria-hidden>
                  <Icon name={cfg.icon} size={18} />
                </span>
                <div className="body">
                  <strong>{cfg.label}</strong>
                  <span className="sub">
                    {r.status === "mastered"
                      ? "Mastered — no further reviews"
                      : `Reviews every ${r.interval_days} day${r.interval_days > 1 ? "s" : ""}`}
                  </span>
                </div>
                {r.status === "mastered" ? (
                  <span className="review-badge mastered">
                    <Icon name="check_circle" size={14} fill /> Mastered
                  </span>
                ) : r.due ? (
                  <span className="review-badge due">Due now</span>
                ) : (
                  <span className="review-badge">Due {r.next_due.slice(0, 5)}</span>
                )}
                {r.lesson && r.status !== "mastered" && (
                  <Link href={r.lesson.url} className="btn btn-outline review-go">
                    Review →
                  </Link>
                )}
              </div>
            );
          })
        )}
      </section>
    </AppShell>
  );
}
