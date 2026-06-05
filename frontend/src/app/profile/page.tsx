import { AppShell } from "@/components/AppShell";

export default function ProfilePage() {
  return (
    <AppShell
      title="Profile"
      description="Learner information, preferred role, and saved progress summary."
    >
      <div className="profile-grid">
        <section className="profile-card">
          <h3>Nguyen Ngoc Gia Han</h3>
          <p className="muted">BSc Computer Science final year project demo learner.</p>
          <div className="metric-row">
            <span>Default role</span>
            <strong>Student</strong>
          </div>
          <div className="metric-row">
            <span>Skill signal</span>
            <strong>Novice</strong>
          </div>
        </section>
        <section className="profile-card">
          <h3>Learning snapshot</h3>
          <p className="muted">This data will come from CodeSession and BugDetection records later.</p>
          <div className="metric-row">
            <span>Total submissions</span>
            <strong>32</strong>
          </div>
          <div className="metric-row">
            <span>Completed lessons</span>
            <strong>8</strong>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
