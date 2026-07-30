"use client";

import { Icon } from "@/components/ui/Icon";
import { RoleToggle } from "@/components/ui/RoleToggle";
import type { AnalyzeResponse } from "@/lib/types";
import { BugCard } from "./BugCard";
import { FeedbackCard } from "./FeedbackCard";
import { RecommendedLessons } from "./RecommendedLessons";
import { SkeletonCards } from "./SkeletonCards";
import { SkillCard } from "./SkillCard";

type AIPanelProps = {
  result: AnalyzeResponse | null;
  isAnalyzing: boolean;
  error: string | null;
  onReset: () => void;
};

export function AIPanel({ result, isAnalyzing, error, onReset }: AIPanelProps) {
  return (
    <aside className="ai-panel">
      <div className="ai-header">
        <span className="glow-dot" aria-hidden />
        <h3>AI Analysis</h3>
        <span className="cb-badge">CodeBERT</span>
      </div>

      <div className="view-as">
        <p>View feedback as:</p>
        <RoleToggle />
      </div>

      <div className="ai-scroll">
        {isAnalyzing ? (
          <SkeletonCards />
        ) : error ? (
          <div className="ai-empty">
            <Icon name="error" size={30} className="ic" style={{ color: "var(--danger)" }} />
            <h4>Analysis failed</h4>
            <p>{error}</p>
            <button className="reset-link" onClick={onReset}>
              <Icon name="restart_alt" size={16} /> Try again
            </button>
          </div>
        ) : result ? (
          <>
            <BugCard bug={result.bug} />
            <SkillCard skill={result.skill} />
            <FeedbackCard feedback={result.feedback} />
            <RecommendedLessons lessons={result.lessons} />
            <button className="reset-link" onClick={onReset}>
              <Icon name="restart_alt" size={16} /> Reset Analysis
            </button>
          </>
        ) : (
          <div className="ai-empty">
            <Icon name="search" size={30} className="ic" />
            <h4>Ready to Analyse</h4>
            <p>Load a bug example or paste your own Python code, then click Analyse.</p>
          </div>
        )}
      </div>
    </aside>
  );
}
