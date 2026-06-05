import { AppShell } from "@/components/AppShell";
import { ProgressChart } from "@/components/ProgressChart";

export default function ProgressPage() {
  return (
    <AppShell
      title="Progress Chart"
      description="Track improvement across analysis sessions, lessons, and repeated submissions."
    >
      <div className="content-grid">
        <ProgressChart />
        <section className="panel">
          <p className="section-label">Session Summary</p>
          <h3>Current week</h3>
          <div className="metric-row">
            <span>Analyses completed</span>
            <strong>14</strong>
          </div>
          <div className="metric-row">
            <span>Lessons completed</span>
            <strong>5</strong>
          </div>
          <div className="metric-row">
            <span>Most common bug</span>
            <strong>Logic</strong>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
