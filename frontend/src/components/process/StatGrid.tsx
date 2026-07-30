import type { Stat } from "@/lib/processContent";

export function StatGrid({ stats }: { stats: Stat[] }) {
  return (
    <div className="proc-stat-grid">
      {stats.map((s) => (
        <div className="proc-stat" key={s.label}>
          <strong>{s.value}</strong>
          <span className="proc-stat-label">{s.label}</span>
          {s.caption && <span className="proc-stat-cap">{s.caption}</span>}
        </div>
      ))}
    </div>
  );
}
