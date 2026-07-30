"use client";

import { useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { AchievementGrid } from "@/components/progress/AchievementGrid";
import { ScoreChart } from "@/components/progress/ScoreChart";
import { Icon } from "@/components/ui/Icon";
import {
  getAchievements,
  getUserProgress,
  type Achievement,
  type ScorePoint,
} from "@/lib/api";

export default function ProgressPage() {
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [scores, setScores] = useState<ScorePoint[]>([]);

  useEffect(() => {
    getUserProgress()
      .then((p) => setScores(p.score_over_time ?? []))
      .catch(console.error);
    getAchievements().then(setAchievements).catch(console.error);
  }, []);

  const totalPts = scores.reduce((sum, p) => sum + p.score, 0);
  const hasData = totalPts > 0;
  const weekLabel =
    scores.length >= 7
      ? `${scores[0].date.slice(0, 5)} – ${scores[6].date.slice(0, 5)}`
      : "";

  return (
    <AppShell>
      <div className="page-heading">
        <h1>Your Progress</h1>
      </div>

      <section className="card progress-card">
        <div className="head">
          <div>
            <h3>Learning Score Over Time</h3>
            <div className="sub">
              This week{weekLabel && ` · ${weekLabel}`} · +10 pts per analysis
            </div>
          </div>
          {totalPts > 0 && (
            <span className="pts-badge">
              <Icon name="trending_up" size={15} /> {totalPts} pts this week
            </span>
          )}
        </div>

        {hasData ? (
          <ScoreChart data={scores} />
        ) : (
          <p className="muted" style={{ padding: "32px 0", textAlign: "center" }}>
            No activity yet this week — analyse some code to earn points.
          </p>
        )}
      </section>

      <section className="card progress-card">
        <h3 style={{ marginBottom: 16 }}>Achievements</h3>
        <AchievementGrid achievements={achievements} />
      </section>
    </AppShell>
  );
}
