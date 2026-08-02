"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Icon } from "@/components/ui/Icon";
import { getStoredToken, login, register } from "@/lib/auth";
import { useRole } from "@/lib/roleStore";
import type { UserRole } from "@/lib/types";

type Mode = "login" | "register";

const ROLE_CARDS: { role: UserRole; icon: string; title: string; sub: string }[] = [
  { role: "student", icon: "school", title: "Student", sub: "Learning Python" },
  { role: "worker", icon: "work", title: "Professional", sub: "Dev / Engineer" },
];

export default function AuthPage() {
  const router = useRouter();
  const { role, setRole } = useRole();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // If already logged in, go straight to dashboard
  if (typeof window !== "undefined" && getStoredToken()) {
    router.replace("/dashboard");
    return null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === "register") {
        await register(email, password, role);
      }
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth">
      <section className="auth-left">
        <Link href="/" className="brand">
          <span className="brand-mark"><Icon name="smart_toy" size={18} /></span>
          <span>PyTutor</span>
        </Link>
        <div>
          <h1>Your AI Python learning companion</h1>
          <p className="lead">
            Paste code → AI detects the bug → personalised feedback → targeted lessons. Repeat until
            you level up.
          </p>
        </div>
        <div className="auth-bullets">
          <div className="auth-bullet">
            <Icon name="search" className="ic" />
            CodeBERT bug classifier · F1: 0.9556
          </div>
          <div className="auth-bullet">
            <Icon name="tune" className="ic" />
            Student or Professional role — feedback adapts
          </div>
          <div className="auth-bullet">
            <Icon name="eco" className="ic" />
            Skill level: Novice or Professional — LR F1: 0.8542
          </div>
        </div>
        <Link href="/" className="auth-back">
          ← Back to home
        </Link>
      </section>

      <section className="auth-right">
        <form className="auth-form" onSubmit={handleSubmit}>
          <h2>{mode === "register" ? "Create your account" : "Welcome back"}</h2>
          <p className="sub">
            {mode === "register"
              ? "Start your personalised learning journey"
              : "Sign in to continue your learning journey"}
          </p>

          <div className="field-label" style={{ marginBottom: 8 }}>
            I am a…
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

          <label className="field">
            <span className="field-label">Email</span>
            <input
              className="input"
              type="email"
              placeholder="username@university.edu"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </label>

          <label className="field">
            <span className="field-label">
              Password
              {mode === "login" && <a href="#">Forgot?</a>}
            </span>
            <input
              className="input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              autoComplete={mode === "register" ? "new-password" : "current-password"}
            />
          </label>

          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}

          <button
            type="submit"
            className="btn btn-gradient"
            disabled={loading}
            style={{ width: "100%" }}
          >
            {loading
              ? mode === "register"
                ? "Creating account…"
                : "Signing in…"
              : mode === "register"
                ? "Create Account →"
                : "Sign In to PyTutor →"}
          </button>

          <p className="auth-alt">
            {mode === "login" ? "No account? " : "Already have an account? "}
            <button
              type="button"
              onClick={() => {
                setMode(mode === "login" ? "register" : "login");
                setError(null);
              }}
            >
              {mode === "login" ? "Create one free" : "Sign in"}
            </button>
          </p>
        </form>
      </section>
    </main>
  );
}
