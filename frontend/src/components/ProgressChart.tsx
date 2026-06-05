import { progressPoints } from "@/lib/mock-data";

export function ProgressChart() {
  const max = 100;
  const points = progressPoints.map((point, index) => {
    const x = (index / (progressPoints.length - 1)) * 100;
    const y = 100 - (point.score / max) * 100;
    return `${x},${y}`;
  });

  return (
    <section className="panel progress-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Progress</p>
          <h3>Weekly improvement</h3>
        </div>
        <strong>+44%</strong>
      </div>
      <svg viewBox="0 0 100 100" className="line-chart" role="img" aria-label="Weekly progress chart">
        <polyline points={points.join(" ")} fill="none" stroke="currentColor" strokeWidth="3" />
        {progressPoints.map((point, index) => (
          <circle
            key={point.label}
            cx={(index / (progressPoints.length - 1)) * 100}
            cy={100 - (point.score / max) * 100}
            r="2.5"
          />
        ))}
      </svg>
      <div className="chart-labels">
        {progressPoints.map((point) => (
          <span key={point.label}>{point.label}</span>
        ))}
      </div>
    </section>
  );
}
