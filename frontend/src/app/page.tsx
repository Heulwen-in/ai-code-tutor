import Link from "next/link";

import { Icon } from "@/components/ui/Icon";

const features = [
  {
    icon: "search",
    title: "AI Bug Detection",
    text: "4 bug classes: syntax, indentation, logic, variable misuse.",
  },
  {
    icon: "tune",
    title: "Role-Adaptive Feedback",
    text: "Student or Professional — different tone, same accuracy.",
  },
  {
    icon: "menu_book",
    title: "Lesson Recommendations",
    text: "Bug type + role, targeted lessons to close the gap.",
  },
  {
    icon: "insights",
    title: "Progress Tracking",
    text: "Score over time, streaks, badges, learning loop.",
  },
];

// Honest, defensible model metrics (grouped-split evaluation).
const stats = [
  { value: "0.9556", label: "CodeBERT F1" },
  { value: "4 + 1", label: "Bug Classes" },
  { value: "0.8542", label: "Skill F1" },
  { value: "2", label: "Roles" },
];

export default function LandingPage() {
  return (
    <main className="landing">
      <nav className="landing-nav">
        <Link href="/" className="brand">
          <span className="brand-mark"><Icon name="smart_toy" size={18} /></span>
          <span>PyTutor</span>
        </Link>
        <div className="landing-links">
          <a href="#features">Features</a>
          <Link href="/lessons">Lessons</Link>
          <Link href="/process">How It&apos;s Built</Link>
          <Link href="/auth" className="btn btn-ghost" style={{ minHeight: 38 }}>
            Log in
          </Link>
          <Link href="/auth" className="btn btn-gradient" style={{ minHeight: 38 }}>
            Get Started Free
          </Link>
        </div>
      </nav>

      <section className="landing-hero">
        <div className="hero-copy">
          <h1>
            Debug <span className="grad-text">Smarter.</span>
            <br />
            Learn <span className="grad-text">Faster.</span>
          </h1>
          <p>
            Paste Python code → AI detects the bug type → personalised feedback for your role
            (Student or Professional) → recommended lessons to fix the gap.
          </p>
          <div className="hero-actions">
            <Link href="/auth" className="btn btn-gradient">
              Start Analysing →
            </Link>
            <Link href="/analyze?demo=1" className="btn btn-outline">
              View Demo
            </Link>
            <Link href="/process" className="btn btn-outline">
              See How It&apos;s Built
            </Link>
          </div>
          <div className="stat-pills">
            {stats.map((s) => (
              <div className="stat-pill" key={s.label}>
                <strong>{s.value}</strong>
                <span>{s.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="preview-card" aria-label="Live analysis preview">
          <div className="preview-titlebar">
            <span className="traffic" aria-hidden>
              <i />
              <i />
              <i />
            </span>
            <span className="preview-tab">two_sum.py</span>
          </div>
          <pre className="preview-code">
            <span className="ln">1</span>
            <span className="kw">def </span>
            <span className="fn">two_sum</span>(nums, target):{"\n"}
            <span className="ln">2</span>
            {"    "}
            <span className="kw">for</span> i <span className="kw">in</span> range(len(nums)):{"\n"}
            <span className="ln">3</span>
            {"        "}
            <span className="kw">for</span> j <span className="kw">in</span> range(i+1, len(nums)):
            {"\n"}
            <span className="ln">4</span>
            {"            "}
            <span className="kw">if</span> nums[i]+nums[j] == target:{"\n"}
            <span className="hl">
              <span className="ln">5</span>
              {"                "}
              <span className="kw">return</span> [i,j] <span className="cm"># bug: logic_error</span>
            </span>
            <span className="ln">6</span>
            {"    "}
            <span className="kw">return</span> []
          </pre>
          <div className="preview-result">
            <span
              className="bug-badge"
              style={{ background: "#FEF9C3", color: "#B45309", fontSize: 11 }}
            >
              logic_error
            </span>
            Conf: 91% · Line 5 · Novice · CodeBERT F1: 0.9556
          </div>
        </div>
      </section>

      <section className="landing-features" id="features">
        {features.map((f) => (
          <article className="card feature-card" key={f.title}>
            <Icon name={f.icon} className="emoji" style={{ color: "var(--primary)" }} />
            <h3>{f.title}</h3>
            <p>{f.text}</p>
          </article>
        ))}
      </section>
    </main>
  );
}
