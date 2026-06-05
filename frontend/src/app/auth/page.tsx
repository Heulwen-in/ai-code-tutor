import Link from "next/link";

export default function AuthPage() {
  return (
    <main className="auth-page">
      <section className="auth-intro">
        <Link href="/" className="landing-brand">
          <span className="brand-mark">CT</span>
          <span>CodeTutor AI</span>
        </Link>
        <h1>Start a guided Python learning session.</h1>
        <p>
          Sign in to save code analysis history, lesson progress, badges, and profile settings. This
          screen is frontend-only for now and will be wired to the FastAPI auth router later.
        </p>
      </section>
      <section className="auth-card">
        <h2>Sign in</h2>
        <p className="muted">Use a demo account while the backend authentication is being completed.</p>
        <form className="form-stack">
          <label>
            Email
            <input type="email" placeholder="student@example.com" />
          </label>
          <label>
            Password
            <input type="password" placeholder="Enter password" />
          </label>
          <label>
            Learning role
            <select defaultValue="student">
              <option value="student">Student</option>
              <option value="worker">Professional worker</option>
            </select>
          </label>
          <Link href="/dashboard" className="primary-button">
            Continue to dashboard
          </Link>
        </form>
      </section>
    </main>
  );
}
