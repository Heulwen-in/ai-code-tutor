"use client";

import { Icon } from "@/components/ui/Icon";
import { useRole } from "@/lib/roleStore";
import type { UserRole } from "@/lib/types";

const OPTIONS: { role: UserRole; label: string; icon: string }[] = [
  { role: "student", label: "Student", icon: "school" },
  { role: "worker", label: "Professional", icon: "work" },
];

/**
 * Global role pill toggle. Reads/writes the shared role context, so the header,
 * the Analyse panel, and Settings all stay in sync.
 */
export function RoleToggle() {
  const { role, setRole } = useRole();

  return (
    <div className="role-pill" role="group" aria-label="Feedback role">
      {OPTIONS.map((opt) => (
        <button
          key={opt.role}
          type="button"
          className={role === opt.role ? "active" : ""}
          aria-pressed={role === opt.role}
          onClick={() => setRole(opt.role)}
        >
          <Icon name={opt.icon} size={16} />
          {opt.label}
        </button>
      ))}
    </div>
  );
}
