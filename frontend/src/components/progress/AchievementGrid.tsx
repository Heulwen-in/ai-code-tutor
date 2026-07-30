import { Icon } from "@/components/ui/Icon";
import type { Achievement } from "@/lib/api";

const GROUPS = [
  { key: "novice", label: "Novice", icon: "school" },
  { key: "professional", label: "Professional", icon: "workspace_premium" },
] as const;

export function AchievementGrid({ achievements }: { achievements: Achievement[] }) {
  return (
    <>
      {GROUPS.map((group) => {
        const items = achievements.filter((a) => a.category === group.key);
        if (items.length === 0) return null;
        const earned = items.filter((a) => a.status === "earned").length;

        return (
          <div key={group.key} className="achieve-section">
            <div className="achieve-section-head">
              <Icon name={group.icon} size={18} />
              {group.label}
              <span className="count">
                {earned}/{items.length}
              </span>
            </div>
            <div className="achieve-grid">
              {items.map((a) => (
                <div key={a.id} className={`card achieve-card ${a.status}`}>
                  <span className="ic" aria-hidden>
                    <Icon name={a.icon} size={22} />
                  </span>
                  <div className="body">
                    <strong>{a.title}</strong>
                    <p>{a.description}</p>
                  </div>
                  <span className="val">{a.value}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </>
  );
}
