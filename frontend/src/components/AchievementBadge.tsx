import type { Achievement } from "@/lib/types";

type AchievementBadgeProps = {
  achievement: Achievement;
};

export function AchievementBadge({ achievement }: AchievementBadgeProps) {
  return (
    <article className={`achievement ${achievement.status}`}>
      <div className="achievement-icon">{achievement.status === "earned" ? "OK" : achievement.status === "active" ? "..." : "--"}</div>
      <div>
        <h3>{achievement.title}</h3>
        <p>{achievement.description}</p>
        <span>{achievement.value}</span>
      </div>
    </article>
  );
}
