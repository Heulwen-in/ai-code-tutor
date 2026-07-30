"use client";

import { useState } from "react";

import type { Figure } from "@/lib/processContent";

// Renders the image; if the file is missing (onError) it swaps to a labelled
// dashed placeholder so the page never shows a broken-image icon.
export function FigureSlot({ src, alt, caption, wide }: Figure) {
  const [failed, setFailed] = useState(false);
  const filename = src.split("/").pop() ?? src;

  return (
    <figure className={wide ? "proc-figure wide" : "proc-figure"}>
      {failed ? (
        <div className="proc-figure-missing" role="img" aria-label={alt}>
          <span className="proc-figure-missing-icon material-symbols-rounded" aria-hidden>
            image
          </span>
          <span className="proc-figure-missing-name">{filename}</span>
          <span className="proc-figure-missing-hint">Drop this image into /public/process/</span>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />
      )}
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
