import type { AuthToken, User, UserRole } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Token storage ─────────────────────────────────────────────────────────────

const TOKEN_KEY = "ct_token";
const USER_KEY  = "ct_user";

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): User | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw) as User; } catch { return null; }
}

export function setStoredUser(user: User): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function saveSession(token: string, user: User): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function register(
  email: string,
  password: string,
  role: UserRole,
): Promise<User> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, role }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Registration failed.");
  }
  return res.json() as Promise<User>;
}

// Login uses OAuth2 form encoding (username/password fields, not JSON)
export async function login(email: string, password: string): Promise<User> {
  const body = new URLSearchParams({ username: email, password });
  const tokenRes = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });
  if (!tokenRes.ok) {
    const err = await tokenRes.json().catch(() => ({}));
    throw new Error(err.detail ?? "Incorrect email or password.");
  }
  const { access_token }: AuthToken = await tokenRes.json();

  // Fetch the full user profile with the new token
  const meRes = await fetch(`${API_BASE}/users/me`, {
    headers: { Authorization: `Bearer ${access_token}` },
  });
  if (!meRes.ok) throw new Error("Could not load user profile.");
  const user: User = await meRes.json();

  saveSession(access_token, user);
  return user;
}

export function logout(): void {
  clearSession();
}
