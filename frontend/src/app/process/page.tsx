import Link from "next/link";
import type { Metadata } from "next";

import { Icon } from "@/components/ui/Icon";
import { FigureSlot } from "@/components/process/FigureSlot";
import { MiniTable } from "@/components/process/MiniTable";
import { PipelineDiagram } from "@/components/process/PipelineDiagram";
import { ProcessSection } from "@/components/process/ProcessSection";
import { SectionNav } from "@/components/process/SectionNav";
import { StatGrid } from "@/components/process/StatGrid";
import { HERO_PILLS, SECTIONS } from "@/lib/processContent";

export const metadata: Metadata = {
  title: "How It's Built · PyTutor",
  description: "The end-to-end machine-learning process behind PyTutor — from raw buggy code to a hierarchical CodeBERT tutor.",
};

export default function ProcessPage() {
  return (
    <main className="landing proc">
      <nav className="landing-nav">
        <Link href="/" className="brand">
          <span className="brand-mark"><Icon name="smart_toy" size={18} /></span>
          <span>PyTutor</span>
        </Link>
        <div className="landing-links">
          <Link href="/#features">Features</Link>
          <Link href="/lessons">Lessons</Link>
          <Link href="/process" aria-current="page">How It&apos;s Built</Link>
          <Link href="/auth" className="btn btn-ghost" style={{ minHeight: 38 }}>
            Log in
          </Link>
          <Link href="/auth" className="btn btn-gradient" style={{ minHeight: 38 }}>
            Get Started Free
          </Link>
        </div>
      </nav>

      <header className="proc-hero">
        <p className="proc-kicker">Implementation Process</p>
        <h1>
          From raw code to a <span className="grad-text">working AI tutor</span>
        </h1>
        <p className="proc-hero-sub">
          A walkthrough of the machine-learning pipeline behind PyTutor — the data, the analysis, the
          leakage-free evaluation, and the hierarchical CodeBERT model that powers it.
        </p>
        <div className="stat-pills">
          {HERO_PILLS.map((p) => (
            <div className="stat-pill" key={p.label}>
              <strong>{p.value}</strong>
              <span>{p.label}</span>
            </div>
          ))}
        </div>
      </header>

      <SectionNav />

      <div className="proc-body">
        {SECTIONS.map((section) => (
          <ProcessSection
            key={section.id}
            id={section.id}
            kicker={section.kicker}
            title={section.title}
          >
            <div className="proc-prose">
              {section.prose.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>

            {section.pipeline && <PipelineDiagram />}
            {section.stats && <StatGrid stats={section.stats} />}
            {section.table && <MiniTable {...section.table} />}
            {section.tables?.map((t, i) => <MiniTable key={i} {...t} />)}

            {section.figures && section.figures.length > 0 && (
              <div className="proc-figures">
                {section.figures.map((f) => (
                  <FigureSlot key={f.src} {...f} />
                ))}
              </div>
            )}
          </ProcessSection>
        ))}
      </div>

      <footer className="proc-cta">
        <h2>See it run on your own code</h2>
        <div className="hero-actions">
          <Link href="/auth" className="btn btn-gradient">
            Start Analysing →
          </Link>
          <Link href="/analyze?demo=1" className="btn btn-outline">
            View Demo
          </Link>
        </div>
      </footer>
    </main>
  );
}
