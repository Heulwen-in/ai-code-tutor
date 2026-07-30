"use client";

import { useState, type CSSProperties, type KeyboardEvent, type MouseEvent } from "react";

import { Icon } from "@/components/ui/Icon";
import { BUG_CONFIG, EXAMPLE_SNIPPETS } from "@/lib/constants";
import type { BugType } from "@/lib/types";

type CodeEditorProps = {
  code: string;
  onChange: (code: string) => void;
  onAnalyse: () => void;
  isAnalyzing: boolean;
  /** Line reported by the classifier (1-indexed); null when there is no bug. */
  bugLine?: number | null;
  bugType?: BugType | null;
  /** Demo mode — the textarea is locked; examples only. */
  readOnly?: boolean;
  /** Called when a locked (demo) user tries to edit; routes to /auth. */
  onRequestEdit?: () => void;
};

export function CodeEditor({
  code,
  onChange,
  onAnalyse,
  isAnalyzing,
  bugLine = null,
  bugType = null,
  readOnly = false,
  onRequestEdit,
}: CodeEditorProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const lines = code.split("\n");
  const lineCount = Math.max(lines.length, 12);

  // Only highlight while the reported line still exists in the current code
  const bugCfg = bugType ? BUG_CONFIG[bugType] : null;
  const showBand =
    bugCfg != null && bugLine != null && bugLine >= 1 && bugLine <= lines.length;
  const bandTint = bugCfg ? `${bugCfg.color}26` : undefined;

  // In demo mode any attempt to interact with the textarea routes to sign-in.
  function guard(): boolean {
    if (readOnly) {
      onRequestEdit?.();
      return true;
    }
    return false;
  }

  // Tab inserts 4 spaces instead of moving focus
  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (readOnly) {
      e.preventDefault();
      guard();
      return;
    }
    if (e.key === "Tab") {
      e.preventDefault();
      const target = e.currentTarget;
      const { selectionStart, selectionEnd } = target;
      const next = code.slice(0, selectionStart) + "    " + code.slice(selectionEnd);
      onChange(next);
      requestAnimationFrame(() => {
        target.selectionStart = target.selectionEnd = selectionStart + 4;
      });
    }
  }

  function handleMouseDown(e: MouseEvent<HTMLTextAreaElement>) {
    if (readOnly) {
      e.preventDefault();
      guard();
    }
  }

  return (
    <div className="editor">
      <div className="editor-toolbar">
        <span className="traffic" aria-hidden>
          <i />
          <i />
          <i />
        </span>
        <div className="spacer" />

        {readOnly && (
          <button type="button" className="demo-write" onClick={() => onRequestEdit?.()}>
            <Icon name="edit" size={15} /> Write your own code
          </button>
        )}

        <div className="load-example">
          <button type="button" onClick={() => setMenuOpen((v) => !v)} aria-expanded={menuOpen}>
            Load Example <Icon name="expand_more" size={16} />
          </button>
          {menuOpen && (
            <div className="load-menu" role="menu">
              {EXAMPLE_SNIPPETS.map((snip) => (
                <button
                  key={snip.label}
                  role="menuitem"
                  onClick={() => {
                    onChange(snip.code);
                    setMenuOpen(false);
                  }}
                >
                  {snip.label}
                </button>
              ))}
            </div>
          )}
        </div>

        <span className="py-badge">Python 3.11</span>
        <button
          className="btn btn-gradient analyse-run"
          onClick={onAnalyse}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? "Analysing…" : <><Icon name="play_arrow" size={17} /> Analyse</>}
        </button>
      </div>

      <div className="editor-body">
        {showBand && bugCfg && (
          <div
            className="bug-line-band"
            aria-hidden
            style={
              {
                "--bug-line": bugLine,
                color: bugCfg.color,
                background: bandTint,
              } as CSSProperties
            }
          >
            <span className="tag"><Icon name="warning" size={13} fill /> {bugType}</span>
          </div>
        )}

        <div className="editor-gutter" aria-hidden>
          {Array.from({ length: lineCount }, (_, i) => {
            const isBug = showBand && i + 1 === bugLine;
            return (
              <span
                key={i}
                className={isBug ? "bug" : undefined}
                style={isBug && bugCfg ? { color: bugCfg.color, background: bandTint } : undefined}
              >
                {i + 1}
              </span>
            );
          })}
        </div>
        <textarea
          className="code-area"
          value={code}
          spellCheck={false}
          readOnly={readOnly}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onMouseDown={handleMouseDown}
          placeholder={readOnly ? "Load an example below, or sign in to write your own code." : undefined}
          aria-label="Python code editor"
        />
      </div>

      <div className="editor-status">
        <span className="dot" />
        CodeBERT ready · FastAPI
        <span className="spacer" />
        Python 3.11 · UTF-8
      </div>
    </div>
  );
}
