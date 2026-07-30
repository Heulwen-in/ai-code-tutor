"use client";

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import { updateRole } from "./api";
import { getStoredToken, getStoredUser, setStoredUser } from "./auth";
import type { UserRole } from "./types";

const ROLE_KEY = "pt_role";

type RoleContextValue = {
  role: UserRole;
  setRole: (role: UserRole) => void;
};

const RoleContext = createContext<RoleContextValue | null>(null);

/**
 * Global learning-role state. Single source of truth mirrored by the header
 * toggle, the Analyse panel toggle, and Settings. Initialised from the logged-in
 * user's role (or a persisted override) and saved to localStorage.
 */
export function RoleProvider({ children }: { children: ReactNode }) {
  const [role, setRoleState] = useState<UserRole>("student");

  // Hydrate on mount: prefer an explicit override, else the signed-in user's role.
  useEffect(() => {
    const stored = typeof window !== "undefined" ? localStorage.getItem(ROLE_KEY) : null;
    if (stored === "student" || stored === "worker") {
      setRoleState(stored);
    } else {
      const user = getStoredUser();
      if (user?.role) setRoleState(user.role);
    }
  }, []);

  const setRole = useCallback((next: UserRole) => {
    setRoleState(next);
    if (typeof window !== "undefined") localStorage.setItem(ROLE_KEY, next);
    // Persist to the account so the choice survives re-login (best-effort).
    if (getStoredToken()) {
      const user = getStoredUser();
      if (user) setStoredUser({ ...user, role: next });
      updateRole(next).catch(() => {});
    }
  }, []);

  return <RoleContext.Provider value={{ role, setRole }}>{children}</RoleContext.Provider>;
}

export function useRole(): RoleContextValue {
  const ctx = useContext(RoleContext);
  if (!ctx) throw new Error("useRole must be used within a RoleProvider");
  return ctx;
}
