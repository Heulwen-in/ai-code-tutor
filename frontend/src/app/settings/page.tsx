import { AppShell } from "@/components/AppShell";

export default function SettingsPage() {
  return (
    <AppShell
      title="Settings"
      description="Frontend controls for role, model mode, theme, and learning preferences."
    >
      <section className="settings-card">
        <h3>Learning preferences</h3>
        <div className="settings-row">
          <div>
            <strong>Default role</strong>
            <p className="muted">Used to adapt feedback tone.</p>
          </div>
          <select defaultValue="student">
            <option value="student">Student</option>
            <option value="worker">Professional worker</option>
          </select>
        </div>
        <div className="settings-row">
          <div>
            <strong>Mock API mode</strong>
            <p className="muted">Keep enabled until the frontend is wired to FastAPI.</p>
          </div>
          <button className="switch" aria-label="Mock API mode enabled" />
        </div>
        <div className="settings-row">
          <div>
            <strong>Lesson reminders</strong>
            <p className="muted">Suggest a lesson after every bug classification.</p>
          </div>
          <button className="switch" aria-label="Lesson reminders enabled" />
        </div>
      </section>
    </AppShell>
  );
}
