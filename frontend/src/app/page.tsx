import Link from "next/link";

import { sampleCode } from "@/lib/mock-data";

const features = [
  {
    title: "AI code analysis",
    text: "Classifies Python bug types with a CodeBERT-based backend and rule-based syntax checks."
  },
  {
    title: "Role-aware feedback",
    text: "Adapts explanation style for students and professional workers."
  },
  {
    title: "Lesson learning",
    text: "Recommends short, W3Schools-style lessons linked to the detected issue."
  },
  {
    title: "Progress tracking",
    text: "Stores sessions, charts improvement, and awards learning achievements."
  }
];

export default function LandingPage() {
  return (
    <main className="landing">
      <nav className="landing-nav">
        <Link href="/" className="landing-brand">
          <span className="brand-mark">CT</span>
          <span>CodeTutor AI</span>
        </Link>
        <div className="landing-links">
          <Link href="/lessons">Lessons</Link>
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/auth" className="primary-button">
            Sign in
          </Link>
        </div>
      </nav>

      <section className="landing-hero">
        <div className="landing-copy">
          <h1>AI-Based Personalised Programming Tutor</h1>
          <p>
            A bachelor final project that helps Python learners submit code, detect bug types, understand
            feedback, follow recommended lessons, and track progress over time.
          </p>
          <div className="hero-actions">
            <Link href="/dashboard" className="primary-button">
              Open dashboard
            </Link>
            <Link href="/auth" className="secondary-button">
              Create account
            </Link>
          </div>
          <div className="hero-proof">
            <div className="proof-item">
              <strong>0.9855</strong>
              <span>CodeBERT macro F1</span>
            </div>
            <div className="proof-item">
              <strong>4</strong>
              <span>bug classes plus no-bug plan</span>
            </div>
            <div className="proof-item">
              <strong>1</strong>
              <span>main analysis endpoint</span>
            </div>
          </div>
        </div>

        <div className="hero-preview" aria-label="AI tutor dashboard preview">
          <div className="mini-app-header">
            <span>AI Tutor Workspace</span>
            <span>Student mode</span>
          </div>
          <div className="preview-grid">
            <pre className="preview-code">{sampleCode}</pre>
            <div className="preview-panel">
              <div className="mini-card">
                <span className="section-label">Bug</span>
                <h3>Logic error</h3>
                <p>Confidence 91%. Review loop behaviour and edge cases.</p>
              </div>
              <div className="mini-card">
                <span className="section-label">Lesson</span>
                <h3>Trace loops</h3>
                <p>Learn how to dry-run code with small examples.</p>
              </div>
              <div className="mini-card">
                <span className="section-label">Progress</span>
                <h3>Improving</h3>
                <p>Weekly score increased after repeated submissions.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section">
        <div className="section-title">
          <h2>Final project scope</h2>
          <p>
            The frontend is structured around the same flow as the machine learning and backend work:
            analyze, teach, track, and motivate.
          </p>
        </div>
        <div className="feature-grid">
          {features.map((feature) => (
            <article className="feature-card" key={feature.title}>
              <h3>{feature.title}</h3>
              <p>{feature.text}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
