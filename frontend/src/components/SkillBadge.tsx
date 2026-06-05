import type { AnalyzeResponse } from "@/lib/types";

type SkillBadgeProps = {
  result: AnalyzeResponse | null;
};

export function SkillBadge({ result }: SkillBadgeProps) {
  const skill = result?.skill.skill_level ?? "novice";
  const confidence = result ? Math.round(result.skill.confidence * 100) : 0;

  return (
    <section className="panel skill-card">
      <p className="section-label">Skill Signal</p>
      <div className="skill-medal">{skill === "professional" ? "PRO" : "NOV"}</div>
      <h3>{skill === "professional" ? "Professional style" : "Novice style"}</h3>
      <p className="muted">
        {result
          ? `${confidence}% confidence from code behaviour features.`
          : "Skill prediction will appear after analysis."}
      </p>
    </section>
  );
}
