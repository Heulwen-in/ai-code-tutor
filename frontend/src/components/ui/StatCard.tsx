import type { ReactNode } from "react";

type StatCardProps = {
  value: ReactNode;
  label: string;
  trend?: string;
  accent?: string;
};

export function StatCard({ value, label, trend, accent }: StatCardProps) {
  return (
    <div className="card stat-card">
      <div className="val" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      <div className="label">{label}</div>
      {trend ? <div className="trend">{trend}</div> : null}
    </div>
  );
}
