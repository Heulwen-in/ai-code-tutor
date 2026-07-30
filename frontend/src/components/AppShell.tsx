"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { getUserStats } from "@/lib/api";
import { useAuth } from "@/lib/authStore";
import { Icon } from "@/components/ui/Icon";
import { RoleToggle } from "@/components/ui/RoleToggle";

// Pages that require a logged-in user
const PROTECTED = ["/dashboard", "/analyze", "/progress", "/settings"];

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: "grid_view" },
  { href: "/analyze", label: "Analyse", icon: "bolt" },
  { href: "/lessons", label: "Lessons", icon: "menu_book" },
  { href: "/progress", label: "Progress", icon: "insights" },
  { href: "/process", label: "How It's Built", icon: "science" },
  { href: "/settings", label: "Settings", icon: "settings" },
];

type AppShellProps = {
  children: ReactNode;
  /** When true, children fill the main grid area with no padding (Analyse view). */
  bare?: boolean;
  /** When true, the page is a public demo — guests may view it without auth. */
  publicDemo?: boolean;
};

export function AppShell({ children, bare = false, publicDemo = false }: AppShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, ready, signOut } = useAuth();
  const [streak, setStreak] = useState<number | null>(null);

  const gated = !publicDemo && PROTECTED.some((p) => pathname.startsWith(p));

  // Redirect to /auth if the user visits a protected page while logged out
  useEffect(() => {
    if (ready && !user && gated) {
      router.replace("/auth");
    }
  }, [ready, user, gated, router]);

  // Real activity streak for the sidebar badge (fail-soft if backend is down)
  useEffect(() => {
    if (!user) return;
    getUserStats()
      .then((s) => setStreak(s.day_streak))
      .catch(() => setStreak(null));
  }, [user]);

  // Avoid a flash of protected content while auth state resolves
  if (!ready || (!user && gated)) {
    return null;
  }

  function handleLogout() {
    signOut();
    router.push("/auth");
  }

  const initial = user?.email?.[0]?.toUpperCase() ?? "H";
  const name = user?.email?.split("@")[0] ?? "Guest";
  const roleLabel = user?.role === "worker" ? "Professional" : "Student";

  return (
    <div className={bare ? "shell bare" : "shell"}>
      <aside className="shell-sidebar">
        <Link href="/dashboard" className="brand">
          <span className="brand-mark"><Icon name="smart_toy" size={18} /></span>
          <span>PyTutor</span>
        </Link>

        <nav className="shell-nav">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={pathname.startsWith(item.href) ? "nav-item active" : "nav-item"}
            >
              <Icon name={item.icon} size={20} className="ic" />
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="streak-badge">
          <Icon name="local_fire_department" size={22} fill style={{ color: "var(--warning)" }} />
          <div className="streak-text">
            {streak && streak > 0 ? (
              <>
                <strong>{streak}-day streak!</strong>
                <span>Keep analysing daily</span>
              </>
            ) : (
              <>
                <strong>Start your streak</strong>
                <span>Analyse code to begin</span>
              </>
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <span className="avatar">{initial}</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="name">{name}</div>
            <div className="role">{roleLabel}</div>
          </div>
          <button
            onClick={handleLogout}
            aria-label="Sign out"
            title="Sign out"
            style={{
              border: 0,
              background: "none",
              color: "var(--text-faint)",
              cursor: "pointer",
              display: "grid",
              placeItems: "center",
              padding: 4,
            }}
          >
            <Icon name="logout" size={18} />
          </button>
        </div>
      </aside>

      <header className="shell-header">
        <div className="spacer" />
        <RoleToggle />
        <span className="avatar" title={user?.email}>
          {initial}
        </span>
      </header>

      {bare ? children : <main className="shell-main">{children}</main>}
    </div>
  );
}
