// Three shimmer skeletons that mirror the Bug · Skill · Feedback result cards.
const LABELS = ["Bug Detected", "Skill Level", "Personalised Feedback"];

export function SkeletonCards() {
  return (
    <>
      {LABELS.map((label) => (
        <div className="skeleton-card" key={label}>
          <p className="section-label">{label}</p>
          <div className="sk pill" style={{ marginBottom: 12 }} />
          <div className="sk line w90" />
          <div className="sk line w70" />
          <div className="sk line w40" style={{ marginBottom: 0 }} />
        </div>
      ))}
      <div className="loading-note">
        <span className="spinner" />
        Running CodeBERT · F1: 0.9556
      </div>
    </>
  );
}
