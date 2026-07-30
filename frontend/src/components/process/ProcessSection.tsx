"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

type ProcessSectionProps = {
  id: string;
  kicker: string;
  title: string;
  children: ReactNode;
};

// Anchored section that fades/slides in the first time it scrolls into view.
export function ProcessSection({ id, kicker, title, children }: ProcessSectionProps) {
  const ref = useRef<HTMLElement | null>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Reveal immediately if already in view on mount (e.g. deep-linked anchor).
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      setShown(true);
      return;
    }

    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -10% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <section ref={ref} id={id} className={`proc-section${shown ? " in" : ""}`}>
      <p className="proc-kicker">{kicker}</p>
      <h2 className="proc-title">{title}</h2>
      {children}
    </section>
  );
}
