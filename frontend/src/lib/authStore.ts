"use client";

import { useCallback, useEffect, useState } from "react";
import { clearSession, getStoredToken, getStoredUser, logout as authLogout } from "./auth";
import type { User } from "./types";

export function useAuth() {
  const [user, setUser]   = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  // Hydrate from localStorage on mount (client-only)
  useEffect(() => {
    setToken(getStoredToken());
    setUser(getStoredUser());
    setReady(true);
  }, []);

  const signIn = useCallback((newUser: User, newToken: string) => {
    setUser(newUser);
    setToken(newToken);
  }, []);

  const signOut = useCallback(() => {
    authLogout();
    setUser(null);
    setToken(null);
  }, []);

  return { user, token, ready, signIn, signOut };
}
