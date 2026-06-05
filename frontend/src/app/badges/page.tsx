import { AchievementBadge } from "@/components/AchievementBadge";
import { AppShell } from "@/components/AppShell";
import { achievements } from "@/lib/mock-data";

export default function BadgesPage() {
  return (
    <AppShell
      title="Badges And Achievements"
      description="Motivate learners with visible milestones connected to code analysis and lesson completion."
    >
      <div className="achievement-grid">
        {achievements.map((achievement) => (
          <AchievementBadge key={achievement.id} achievement={achievement} />
        ))}
      </div>
    </AppShell>
  );
}
