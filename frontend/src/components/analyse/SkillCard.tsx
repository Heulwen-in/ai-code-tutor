import { Icon } from "@/components/ui/Icon";
import type { AnalyzeResponse } from "@/lib/types";

export function SkillCard({ skill }: { skill: AnalyzeResponse["skill"] }) {
  const isNovice = skill.skill_level === "novice";
  const pct = Math.round(skill.confidence * 100);
  // Position the divider between the two segments by confidence.
  const noviceWidth = isNovice ? pct : 100 - pct;

  return (
    <div className="result-card">
      <p className="section-label">Skill Level Inferred</p>
      <div className="result-row">
        <span className={isNovice ? "skill-badge novice" : "skill-badge pro"}>
          <Icon name={isNovice ? "eco" : "bolt"} size={15} />
          {isNovice ? "Novice" : "Professional"}
        </span>
        <span className="line-ref">conf: {pct}%</span>
      </div>

      <div className="two-seg">
        <div className="fill" style={{ width: `${noviceWidth}%` }} />
      </div>
      <div className="seg-labels">
        <span>Novice</span>
        <span>Professional</span>
      </div>

      <p className="card-footer">LR Classifier · Macro F1: 0.8542</p>
    </div>
  );
}
