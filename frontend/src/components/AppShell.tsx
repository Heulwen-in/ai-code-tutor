"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navItems = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/analyze", label: "AI Tutor" },
  { href: "/lessons", label: "Lessons" },
  { href: "/progress", label: "Progress" },
  { href: "/badges", label: "Badges" },
  { href: "/profile", label: "Profile" },
  { href: "/settings", label: "Settings" }
];

type AppShellProps = {
  children: ReactNode;
  title: string;
  description: string;
};

export function AppShell({ children, title, description }: AppShellProps) {
  const pathname = usePathname();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link href="/" className="brand">
          <span className="brand-mark">CT</span>
          <span>CodeTutor AI</span>
        </Link>
        <nav>
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={pathname === item.href ? "nav-link active" : "nav-link"}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      <main className="app-main">
        <header className="topbar">
          <div>
            <h1>{title}</h1>
            <p>{description}</p>
          </div>
          <div className="topbar-actions">
            <input aria-label="Search" placeholder="Search lessons or sessions" />
            <span className="role-chip">Student</span>
            <span className="avatar">H</span>
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}
