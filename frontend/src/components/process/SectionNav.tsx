"use client";

import { useEffect, useState } from "react";

import { SECTION_NAV } from "@/lib/processContent";

// Sticky anchor bar with IntersectionObserver scroll-spy.
export function SectionNav() {
  const [active, setActive] = useState(SECTION_NAV[0].id);

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-45% 0px -45% 0px", threshold: [0, 0.25, 0.5, 1] },
    );
    SECTION_NAV.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, []);

  function go(e: React.MouseEvent<HTMLAnchorElement>, id: string) {
    e.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActive(id);
  }

  return (
    <nav className="proc-nav" aria-label="Section navigation">
      <div className="proc-nav-inner">
        {SECTION_NAV.map(({ id, label }) => (
          <a
            key={id}
            href={`#${id}`}
            className={active === id ? "active" : ""}
            aria-current={active === id ? "true" : undefined}
            onClick={(e) => go(e, id)}
          >
            {label}
          </a>
        ))}
      </div>
    </nav>
  );
}
