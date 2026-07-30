"use client";

import { useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { Icon } from "@/components/ui/Icon";
import { Toggle } from "@/components/ui/Toggle";
import {
  changePassword,
  getUserStats,
  updateProfile,
  type UserStats,
} from "@/lib/api";
import { getStoredUser, setStoredUser } from "@/lib/auth";
import { useRole } from "@/lib/roleStore";
import type { UserRole } from "@/lib/types";

const ROLE_CARDS: { role: UserRole; icon: string; title: string; sub: string }[] = [
  { role: "student", icon: "school", title: "Student", sub: "Beginner-friendly explanations" },
  { role: "worker", icon: "work", title: "Professional", sub: "Concise technical notes" },
];

const NOTIFY_KEY = "pt_notify";
type Notify = { streak: boolean; lessons: boolean; badges: boolean };
const DEFAULT_NOTIFY: Notify = { streak: true, lessons: true, badges: false };

export default function SettingsPage() {
  const { role, setRole } = useRole();
  const [stats, setStats] = useState<UserStats | null>(null);
  const [notify, setNotify] = useState<Notify>(DEFAULT_NOTIFY);

  const [displayName, setDisplayName] = useState("");
  const [nameStatus, setNameStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  const [pw, setPw] = useState({ current: "", next: "" });
  const [pwStatus, setPwStatus] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [pwError, setPwError] = useState("");

  const user = getStoredUser();
  const email = user?.email ?? "han@university.edu";

  useEffect(() => {
    getUserStats().then(setStats).catch(console.error);
    const stored = getStoredUser();
    setDisplayName(stored?.display_name ?? stored?.email?.split("@")[0] ?? "");
    if (typeof window !== "undefined") {
      const raw = localStorage.getItem(NOTIFY_KEY);
      if (raw) {
        try {
          setNotify({ ...DEFAULT_NOTIFY, ...JSON.parse(raw) });
        } catch {
          /* ignore malformed prefs */
        }
      }
    }
  }, []);

  const name = displayName || email.split("@")[0];
  const initial = name[0]?.toUpperCase() ?? "H";
  const roleLabel = role === "worker" ? "Professional" : "Student";

  function persistNotify(next: Notify) {
    setNotify(next);
    if (typeof window !== "undefined") localStorage.setItem(NOTIFY_KEY, JSON.stringify(next));
  }

  async function saveName() {
    setNameStatus("saving");
    try {
      const updated = await updateProfile(displayName.trim());
      if (user) setStoredUser({ ...user, display_name: updated.display_name });
      setNameStatus("saved");
      setTimeout(() => setNameStatus("idle"), 2000);
    } catch {
      setNameStatus("error");
    }
  }

  async function savePassword() {
    setPwError("");
    if (pw.next.length < 6) {
      setPwError("New password must be at least 6 characters.");
      setPwStatus("error");
      return;
    }
    setPwStatus("saving");
    try {
      await changePassword(pw.current, pw.next);
      setPw({ current: "", next: "" });
      setPwStatus("done");
      setTimeout(() => setPwStatus("idle"), 2500);
    } catch (e) {
      setPwError(e instanceof Error ? e.message : "Password change failed.");
      setPwStatus("error");
    }
  }

  return (
    <AppShell>
      <div className="settings-page">
        <header className="settings-head">
          <h1>Account Settings</h1>
          <p>Manage your profile, security, and learning preferences — all in one place.</p>
        </header>

        <div className="card settings-card">
          {/* Profile header */}
          <section className="settings-section profile-section">
            <div className="profile-row">
              <span className="avatar lg">{initial}</span>
              <div className="profile-id">
                <h2>{name}</h2>
                <div className="email">{email}</div>
                <span className="role-tag">
                  <Icon name={role === "worker" ? "work" : "school"} size={14} /> {roleLabel}
                </span>
              </div>
              <div className="profile-stats">
                <div className="box">
                  <strong>{stats?.total ?? 0}</strong>
                  <span>Analyses</span>
                </div>
                <div className="box">
                  <strong>{stats?.xp ?? 0}</strong>
                  <span>XP Total</span>
                </div>
              </div>
            </div>
          </section>

          {/* Account details — editable */}
          <section className="settings-section">
            <div className="section-head">
              <h3>Account</h3>
              <p>Update how your name appears across PyTutor.</p>
            </div>
            <div className="field-grid">
              <div className="field">
                <label className="field-label" htmlFor="displayName">Display name</label>
                <input
                  id="displayName"
                  className="text-input"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="Your name"
                  maxLength={120}
                />
              </div>
              <div className="field">
                <label className="field-label" htmlFor="email">Email</label>
                <input id="email" className="text-input" value={email} disabled />
              </div>
            </div>
            <div className="section-actions">
              {nameStatus === "saved" && (
                <p className="field-note ok">
                  <Icon name="check_circle" size={14} fill /> Name updated.
                </p>
              )}
              {nameStatus === "error" && (
                <p className="field-note err">Could not save — is the backend running?</p>
              )}
              <button
                type="button"
                className="btn btn-gradient"
                onClick={saveName}
                disabled={nameStatus === "saving"}
              >
                {nameStatus === "saving" ? "Saving…" : "Save"}
              </button>
            </div>
          </section>

          {/* Password */}
          <section className="settings-section">
            <div className="section-head">
              <h3>Password</h3>
              <p>Choose a strong password you don't use elsewhere.</p>
            </div>
            <div className="field-grid">
              <div className="field">
                <label className="field-label" htmlFor="curpw">Current password</label>
                <input
                  id="curpw"
                  type="password"
                  className="text-input"
                  value={pw.current}
                  onChange={(e) => setPw((p) => ({ ...p, current: e.target.value }))}
                  autoComplete="current-password"
                />
              </div>
              <div className="field">
                <label className="field-label" htmlFor="newpw">New password</label>
                <input
                  id="newpw"
                  type="password"
                  className="text-input"
                  value={pw.next}
                  onChange={(e) => setPw((p) => ({ ...p, next: e.target.value }))}
                  autoComplete="new-password"
                />
              </div>
            </div>
            <div className="section-actions">
              {pwStatus === "done" && (
                <p className="field-note ok">
                  <Icon name="check_circle" size={14} fill /> Password changed.
                </p>
              )}
              {pwStatus === "error" && <p className="field-note err">{pwError}</p>}
              <button
                type="button"
                className="btn btn-gradient"
                onClick={savePassword}
                disabled={pwStatus === "saving" || !pw.current || !pw.next}
              >
                {pwStatus === "saving" ? "Saving…" : "Update"}
              </button>
            </div>
          </section>

          {/* Learning role */}
          <section className="settings-section">
            <div className="section-head">
              <h3>Learning Role</h3>
              <p>Changes how AI feedback is phrased across the entire app.</p>
            </div>
            <div className="role-cards">
              {ROLE_CARDS.map((c) => (
                <button
                  type="button"
                  key={c.role}
                  className={role === c.role ? "role-card active" : "role-card"}
                  aria-pressed={role === c.role}
                  onClick={() => setRole(c.role)}
                >
                  <Icon name={c.icon} size={22} className="emoji" />
                  <strong>{c.title}</strong>
                  <span>{c.sub}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Notifications */}
          <section className="settings-section">
            <div className="section-head">
              <h3>Notifications</h3>
              <p>Choose what PyTutor reminds you about.</p>
            </div>
            <div className="toggle-row">
              <div>
                <strong>Daily streak reminder</strong>
                <span>Reminded to analyse code daily</span>
              </div>
              <Toggle
                label="Daily streak reminder"
                checked={notify.streak}
                onChange={(v) => persistNotify({ ...notify, streak: v })}
              />
            </div>
            <div className="toggle-row">
              <div>
                <strong>New lesson alerts</strong>
                <span>Notify when lessons match your bugs</span>
              </div>
              <Toggle
                label="New lesson alerts"
                checked={notify.lessons}
                onChange={(v) => persistNotify({ ...notify, lessons: v })}
              />
            </div>
            <div className="toggle-row">
              <div>
                <strong>Badge achievements</strong>
                <span>Celebrate when you earn a badge</span>
              </div>
              <Toggle
                label="Badge achievements"
                checked={notify.badges}
                onChange={(v) => persistNotify({ ...notify, badges: v })}
              />
            </div>
          </section>

          {/* About */}
          <section className="settings-section">
            <div className="section-head">
              <h3>About This System</h3>
              <p>
                BSc Computer Science Final Year Project · Nguyen Ngoc Gia Han. An AI-based
                personalised Python programming tutor.
              </p>
            </div>
            <div className="info-badges">
              <div className="info-badge">
                <strong>0.9556</strong>
                <span>CodeBERT Macro F1</span>
              </div>
              <div className="info-badge">
                <strong>0.8542</strong>
                <span>Skill LR Macro F1</span>
              </div>
              <div className="info-badge">
                <strong>4 + 1</strong>
                <span>Bug classes + no-bug</span>
              </div>
              <div className="info-badge">
                <strong>FastAPI</strong>
                <span>Next.js · SQLAlchemy</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
