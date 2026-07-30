import { getStoredToken } from "./auth";
import { mockAnalysis } from "./mock-data";
import type { AnalyzeRequest, AnalyzeResponse, User } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API !== "false";

export async function analyzeCode(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  if (USE_MOCK_API || !API_BASE_URL) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    return mockAnalysis;
  }

  const token = getStoredToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Unable to analyze code.");
  }

  return response.json();
}

function authHeaders(): Record<string, string> {
  const token = getStoredToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export type UserStats = {
  total: number;
  no_bug_rate: number;
  most_common_bug: string | null;
  day_streak: number;
  xp: number;
};

export type ScorePoint = { label: string; date: string; day?: number; score: number };

export type UserProgress = {
  total_submissions: number;
  submissions_last_30d: number;
  bug_breakdown: Record<string, number>;
  skill_trend_last_7d: { date: string; skill_level: string }[];
  score_over_time: ScorePoint[];
  lessons_started: number;
  lessons_completed: number;
};

export type Achievement = {
  id: string;
  category: "novice" | "professional";
  title: string;
  description: string;
  icon: string;
  current: number;
  target: number;
  value: string;
  status: "earned" | "active" | "locked";
};

export type LessonStatusMap = Record<string, "started" | "completed">;

export type ReviewItem = {
  bug_type: string;
  interval_days: number;
  next_due: string;
  due: boolean;
  status: "active" | "mastered";
  lesson: {
    lesson_id: string;
    title: string;
    description: string;
    difficulty: string;
    url: string;
  } | null;
};

export const getMe         = ()  => getJson<User>("/users/me");
export const getUserStats  = ()  => getJson<UserStats>("/users/me/stats");
export const getUserProgress = () => getJson<UserProgress>("/users/me/progress");
export const getAchievements = () => getJson<Achievement[]>("/users/me/achievements");
export const getReviews      = () => getJson<ReviewItem[]>("/users/me/reviews");
export const getLessonProgress = (userId: number) =>
  getJson<LessonStatusMap>(`/lessons/progress?user_id=${userId}`);

export async function updateProfile(displayName: string): Promise<User> {
  const res = await fetch(`${API_BASE_URL}/users/me`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ display_name: displayName }),
  });
  if (!res.ok) throw new Error(`profile update failed: ${res.status}`);
  return res.json() as Promise<User>;
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/users/me/password`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Password change failed.");
  }
}

export async function updateRole(role: "student" | "worker"): Promise<User> {
  const res = await fetch(`${API_BASE_URL}/users/me/role`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ role }),
  });
  if (!res.ok) throw new Error(`role update failed: ${res.status}`);
  return res.json() as Promise<User>;
}

// Lesson progress tracking (demo mode: identified by user_id query param)
async function postLessonEvent(lessonId: string, event: "start" | "complete", userId: number): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/lessons/${lessonId}/${event}?user_id=${userId}`,
    { method: "POST", headers: authHeaders() },
  );
  if (!res.ok) throw new Error(`lesson ${event} failed: ${res.status}`);
}

export const startLesson    = (lessonId: string, userId: number) => postLessonEvent(lessonId, "start", userId);
export const completeLesson = (lessonId: string, userId: number) => postLessonEvent(lessonId, "complete", userId);
