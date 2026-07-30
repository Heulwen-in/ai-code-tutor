type ProgressBarProps = {
  value: number; // 0–100
  gradient?: string; // optional custom fill (e.g. bug-type gradient)
};

export function ProgressBar({ value, gradient }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className="progress-bar">
      <span style={{ width: `${clamped}%`, ...(gradient ? { background: gradient } : {}) }} />
    </div>
  );
}
