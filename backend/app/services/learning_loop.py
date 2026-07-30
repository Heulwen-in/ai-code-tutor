from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.models.database import AnalysisHistory, LessonProgress, ReviewSchedule

if TYPE_CHECKING:
    from app.services.bug_classifier import ClassificationResult
    from app.services.skill_detector import SkillResult

# Every analysis submission is worth this many learning points.
POINTS_PER_ANALYSIS = 10

# The four coarse bug classes (excludes no_bug) used for achievement coverage.
BUG_CLASSES = ("syntax_error", "indentation_error", "logic_error", "variable_misuse")

# Leitner spaced-repetition intervals (days). A detected bug resets its item to
# the first interval; each successful review advances to the next; success
# beyond the last interval marks the item mastered.
REVIEW_INTERVALS = (1, 3, 7, 14)

# lesson_id prefix → the bug class that lesson trains (good_* has no review item).
_LESSON_BUG_PREFIX = {
    "syntax": "syntax_error",
    "indent": "indentation_error",
    "logic": "logic_error",
    "var": "variable_misuse",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(code: str, max_chars: int = 2000) -> str:
    """Store only the first 2 000 chars to keep the DB lean."""
    return code[:max_chars] if len(code) > max_chars else code


def _current_streak(rows: list["AnalysisHistory"]) -> int:
    """
    Consecutive-day streak of analysis activity.

    Counts back from today (UTC): a streak stays alive as long as the user was
    active today OR yesterday (today may simply not be done yet). Returns 0 when
    the most recent activity is older than yesterday, or when there is none.
    """
    active_days = {r.created_at.date() for r in rows}
    if not active_days:
        return 0

    today = datetime.utcnow().date()
    # Anchor to today if active today, else yesterday (grace for an unfinished day)
    if today in active_days:
        cursor = today
    elif (today - timedelta(days=1)) in active_days:
        cursor = today - timedelta(days=1)
    else:
        return 0

    streak = 0
    while cursor in active_days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _score_over_time(rows: list["AnalysisHistory"]) -> list[dict]:
    """Points earned on each day of the current week (Monday→Sunday).

    A day's score is POINTS_PER_ANALYSIS multiplied by the number of analyses
    submitted that day, so activity in the current calendar week is shown as
    bars that rise and fall per day.
    """
    today = datetime.utcnow().date()
    monday = today - timedelta(days=today.weekday())   # weekday(): Mon=0
    week = [monday + timedelta(days=i) for i in range(7)]

    per_day = Counter(r.created_at.date() for r in rows)

    series: list[dict] = []
    for d in week:
        series.append({
            "label": d.strftime("%a"),          # Mon, Tue, ...
            "date": d.strftime("%d/%m/%Y"),     # 22/07/2026
            "day": d.day,
            "score": per_day.get(d, 0) * POINTS_PER_ANALYSIS,
        })
    return series


# ─────────────────────────────────────────────────────────────────────────────
#  Spaced repetition (Leitner)
# ─────────────────────────────────────────────────────────────────────────────

def _lesson_bug_type(lesson_id: str) -> str | None:
    """Map a lesson id (e.g. 'logic_pro_02') to the bug class it trains."""
    return _LESSON_BUG_PREFIX.get(lesson_id.split("_", 1)[0])


def _next_interval(current: int) -> int | None:
    """The next larger Leitner interval, or None when the last one is passed."""
    for interval in REVIEW_INTERVALS:
        if interval > current:
            return interval
    return None


def _advance_review(item: ReviewSchedule) -> None:
    """Move an item one Leitner box up; beyond the last box it is mastered."""
    nxt = _next_interval(item.interval_days)
    if nxt is None:
        item.status = "mastered"
    else:
        item.interval_days = nxt
        item.next_due = datetime.utcnow() + timedelta(days=nxt)


def update_review_schedule(db: Session, user_id: int, bug_type: str) -> None:
    """Apply one analysis result to the user's review schedule.

    A detected bug (re)creates its review item at the first interval — the
    concept must be revisited tomorrow. A clean submission counts as a
    successful review for every item currently due.
    """
    now = datetime.utcnow()

    if bug_type in BUG_CLASSES:
        item = (
            db.query(ReviewSchedule)
            .filter_by(user_id=user_id, bug_type=bug_type)
            .first()
        )
        if item is None:
            item = ReviewSchedule(user_id=user_id, bug_type=bug_type)
            db.add(item)
        item.interval_days = REVIEW_INTERVALS[0]
        item.next_due = now + timedelta(days=REVIEW_INTERVALS[0])
        item.status = "active"
        db.commit()

    elif bug_type == "no_bug":
        due_items = (
            db.query(ReviewSchedule)
            .filter(
                ReviewSchedule.user_id == user_id,
                ReviewSchedule.status == "active",
                ReviewSchedule.next_due <= now,
            )
            .all()
        )
        for item in due_items:
            _advance_review(item)
        if due_items:
            db.commit()


def get_review_queue(db: Session, user_id: int, role: str = "student") -> list[dict]:
    """The user's spaced-repetition queue, due items first.

    Each entry carries the recommended lesson for its bug type and role so the
    UI can link a due review directly to the matching exercise.
    """
    from app.services import lesson_recommender

    now = datetime.utcnow()
    items = (
        db.query(ReviewSchedule)
        .filter_by(user_id=user_id)
        .order_by(ReviewSchedule.next_due.asc())
        .all()
    )

    queue: list[dict] = []
    for item in items:
        lessons = lesson_recommender.recommend(item.bug_type, role)
        queue.append({
            "bug_type": item.bug_type,
            "interval_days": item.interval_days,
            "next_due": item.next_due.strftime("%d/%m/%Y"),
            "due": item.status == "active" and item.next_due <= now,
            "status": item.status,
            "lesson": lessons[0].model_dump() if lessons else None,
        })
    # Due items first, then soonest upcoming, mastered last
    queue.sort(key=lambda q: (q["status"] == "mastered", not q["due"]))
    return queue


# ─────────────────────────────────────────────────────────────────────────────
#  Write operations
# ─────────────────────────────────────────────────────────────────────────────

def save_analysis(
    db: Session,
    user_id: int | None,
    code: str,
    bug_result: "ClassificationResult",
    skill_result: "SkillResult",
    language: str = "python",
) -> AnalysisHistory:
    """Persist one analysis result.  Returns the new DB row."""
    row = AnalysisHistory(
        user_id=user_id,
        code_snippet=_truncate(code),
        language=language,
        bug_type=bug_result.bug_type,
        bug_confidence=bug_result.confidence,
        line_number=bug_result.line_number,
        skill_level=skill_result.skill_level,
        skill_source=skill_result.source,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # Feed the spaced-repetition loop (signed-in users only)
    if user_id is not None:
        update_review_schedule(db, user_id, bug_result.bug_type)

    return row


def record_lesson(
    db: Session,
    user_id: int,
    lesson_id: str,
    status: str = "started",
) -> LessonProgress:
    """Upsert a lesson progress record."""
    row = (
        db.query(LessonProgress)
        .filter_by(user_id=user_id, lesson_id=lesson_id)
        .first()
    )
    if row is None:
        row = LessonProgress(user_id=user_id, lesson_id=lesson_id, status=status)
        db.add(row)
    else:
        row.status = status
        row.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(row)

    # Completing a lesson counts as a successful review of its bug class
    if status == "completed":
        bug_type = _lesson_bug_type(lesson_id)
        if bug_type:
            item = (
                db.query(ReviewSchedule)
                .filter_by(user_id=user_id, bug_type=bug_type, status="active")
                .first()
            )
            if item is not None:
                _advance_review(item)
                db.commit()

    return row


# ─────────────────────────────────────────────────────────────────────────────
#  Read operations
# ─────────────────────────────────────────────────────────────────────────────

def get_user_progress(db: Session, user_id: int) -> dict:
    """
    Returns a progress summary for a user:
      - total submissions
      - bug type breakdown (last 30 days)
      - skill level trend (last 7 days)
      - lessons started / completed
    """
    since_30d = datetime.utcnow() - timedelta(days=30)
    since_7d  = datetime.utcnow() - timedelta(days=7)

    all_rows = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.user_id == user_id)
        .order_by(AnalysisHistory.created_at.desc())
        .all()
    )
    recent_30 = [r for r in all_rows if r.created_at >= since_30d]
    recent_7  = [r for r in all_rows if r.created_at >= since_7d]

    bug_counts = Counter(r.bug_type for r in recent_30)
    skill_trend = [
        {"date": r.created_at.strftime("%Y-%m-%d"), "skill_level": r.skill_level}
        for r in reversed(recent_7)
    ]

    lesson_rows = db.query(LessonProgress).filter_by(user_id=user_id).all()
    lessons_started   = sum(1 for l in lesson_rows if l.status in ("started", "completed"))
    lessons_completed = sum(1 for l in lesson_rows if l.status == "completed")

    return {
        "total_submissions": len(all_rows),
        "submissions_last_30d": len(recent_30),
        "bug_breakdown": dict(bug_counts),
        "skill_trend_last_7d": skill_trend,
        "score_over_time": _score_over_time(all_rows),
        "lessons_started": lessons_started,
        "lessons_completed": lessons_completed,
    }


def get_stats(db: Session, user_id: int) -> dict:
    """Lightweight stats for the dashboard header."""
    rows = db.query(AnalysisHistory).filter_by(user_id=user_id).all()
    if not rows:
        return {
            "total": 0, "no_bug_rate": 0.0, "most_common_bug": None,
            "day_streak": 0, "xp": 0,
        }

    total    = len(rows)
    no_bug   = sum(1 for r in rows if r.bug_type == "no_bug")
    bugs     = Counter(r.bug_type for r in rows if r.bug_type != "no_bug")
    top_bug  = bugs.most_common(1)[0][0] if bugs else None

    return {
        "total": total,
        "no_bug_rate": round(no_bug / total, 3),
        "most_common_bug": top_bug,
        "day_streak": _current_streak(rows),
        "xp": total * POINTS_PER_ANALYSIS,   # lifetime learning points
    }


def compute_achievements(db: Session, user_id: int) -> list[dict]:
    """Derive every achievement from real analysis and lesson history.

    Returns a flat list of badges, each tagged with a `category`
    ("novice" | "professional") and progress toward its target. Status is
    "earned" at/above target, "active" once started, else "locked".
    """
    rows = db.query(AnalysisHistory).filter_by(user_id=user_id).all()
    lesson_rows = db.query(LessonProgress).filter_by(user_id=user_id).all()

    total       = len(rows)
    no_bug      = sum(1 for r in rows if r.bug_type == "no_bug")
    bug_rows    = [r for r in rows if r.bug_type in BUG_CLASSES]
    bugs_found  = len(bug_rows)
    distinct_bug_types = len({r.bug_type for r in bug_rows})
    active_days = len({r.created_at.date() for r in rows})
    streak      = _current_streak(rows)

    completed_ids = {l.lesson_id for l in lesson_rows if l.status == "completed"}
    started_ids   = {l.lesson_id for l in lesson_rows if l.status in ("started", "completed")}
    lessons_completed = len(completed_ids)
    pro_completed = sum(1 for lid in completed_ids if "_pro_" in lid)
    pro_started   = sum(1 for lid in started_ids if "_pro_" in lid)

    def badge(bid, category, title, description, icon, current, target):
        current = min(current, target)
        status = "earned" if current >= target else "active" if current > 0 else "locked"
        return {
            "id": bid, "category": category, "title": title,
            "description": description, "icon": icon,
            "current": current, "target": target,
            "value": f"{current}/{target}", "status": status,
        }

    return [
        # ── Novice track ────────────────────────────────────────────────────
        badge("first-analysis", "novice", "First Steps",
              "Run your first code analysis.", "rocket_launch", total, 1),
        badge("syntax-starter", "novice", "Syntax Starter",
              "Detect 5 syntax or indentation issues.", "spellcheck",
              sum(1 for r in bug_rows if r.bug_type in ("syntax_error", "indentation_error")), 5),
        badge("loop-detective", "novice", "Loop Detective",
              "Detect 3 logic errors.", "search",
              sum(1 for r in bug_rows if r.bug_type == "logic_error"), 3),
        badge("steady-learner", "novice", "Steady Learner",
              "Analyse code on 7 different days.", "calendar_month", active_days, 7),
        badge("bug-hunter", "novice", "Bug Hunter",
              "Find 10 bugs across your submissions.", "pest_control", bugs_found, 10),
        badge("first-lesson", "novice", "Bookworm",
              "Complete your first lesson.", "menu_book", lessons_completed, 1),
        badge("streak-3", "novice", "On a Roll",
              "Reach a 3-day analysis streak.", "local_fire_department", streak, 3),
        badge("clean-sweep", "novice", "Clean Sweep",
              "Submit your first no-bug solution.", "task_alt", no_bug, 1),
        # ── Professional track ──────────────────────────────────────────────
        badge("clean-code", "professional", "Clean Code Builder",
              "Submit 10 no-bug solutions.", "verified", no_bug, 10),
        badge("polyglot-debugger", "professional", "Full-Spectrum Debugger",
              "Encounter all 4 bug classes.", "category", distinct_bug_types, 4),
        badge("marathon", "professional", "Marathoner",
              "Reach 50 total analyses.", "trending_up", total, 50),
        badge("lesson-master", "professional", "Lesson Master",
              "Complete 5 professional lessons.", "workspace_premium", pro_completed, 5),
        badge("consistency-pro", "professional", "Consistency Pro",
              "Reach a 14-day analysis streak.", "bolt", streak, 14),
        badge("refactor-ready", "professional", "Refactor Ready",
              "Start 10 professional lessons.", "construction", pro_started, 10),
    ]
