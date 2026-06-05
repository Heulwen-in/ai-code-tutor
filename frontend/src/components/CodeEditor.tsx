"use client";

type CodeEditorProps = {
  code: string;
  onChange: (code: string) => void;
};

export function CodeEditor({ code, onChange }: CodeEditorProps) {
  return (
    <div className="editor-panel">
      <div className="editor-toolbar">
        <div>
          <span className="status-dot" />
          Python
        </div>
        <span>main.py</span>
      </div>
      <textarea
        className="code-editor"
        value={code}
        spellCheck={false}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Python code editor"
      />
    </div>
  );
}
