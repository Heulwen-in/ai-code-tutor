import { AchievementBadge } from "@/components/AchievementBadge";
import { AppShell } from "@/components/AppShell";
import { LessonCard } from "@/components/LessonCard";
import { ProgressChart } from "@/components/ProgressChart";
import { TutorWorkspace } from "@/components/TutorWorkspace";
import { achievements, lessonModules } from "@/lib/mock-data";

export default function DashboardPage() {
  return (
    <AppShell
      title="Learning Dashboard"
      description="Analyze Python code, follow lessons, track progress, and collect achievements."
    >
      <TutorWorkspace />
      <div className="content-grid" style={{ marginTop: 18 }}>
        <section className="panel">
          <p className="section-label">Lesson Learning</p>
          <h3>Continue recommended modules</h3>
          <div className="lesson-list" style={{ marginTop: 14 }}>
            {lessonModules.slice(0, 2).map((lesson) => (
              <LessonCard key={lesson.id} lesson={lesson} compact />
            ))}
          </div>
        </section>
        <ProgressChart />
        <section className="panel">
          <p className="section-label">Badges</p>
          <h3>Achievements</h3>
          <div className="achievement-grid" style={{ marginTop: 14 }}>
            {achievements.slice(0, 2).map((achievement) => (
              <AchievementBadge key={achievement.id} achievement={achievement} />
            ))}
          </div>
        </section>
        <section className="panel">
          <p className="section-label">Next Milestone</p>
          <h3>Prepare final demo flow</h3>
          <p className="muted">
            Use 6 to 8 scripted examples: syntax, indentation, logic, variable misuse, and clean-code
            cases once the clean-code gate is added.
          </p>
        </section>
      </div>
    </AppShell>
  );
}
