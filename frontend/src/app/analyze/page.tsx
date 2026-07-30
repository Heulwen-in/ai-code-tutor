"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { AppShell } from "@/components/AppShell";
import { AIPanel } from "@/components/analyse/AIPanel";
import { CodeEditor } from "@/components/analyse/CodeEditor";
import { getStoredUser } from "@/lib/auth";
import { useAnalysis } from "@/lib/useAnalysis";

function AnalyzeView() {
  const { code, result, isAnalyzing, error, setCode, runAnalysis, reset } = useAnalysis();
  const router = useRouter();
  const params = useSearchParams();

  // Demo mode = ?demo=1 AND not signed in. Guests may run the bundled examples
  // but must authenticate to type their own code.
  const isDemo = params.get("demo") === "1" && !getStoredUser();

  return (
    <AppShell bare publicDemo={isDemo}>
      <div className="analyse">
        <CodeEditor
          code={code}
          onChange={setCode}
          onAnalyse={runAnalysis}
          isAnalyzing={isAnalyzing}
          bugLine={result?.bug.line_number ?? null}
          bugType={result?.bug.bug_type ?? null}
          readOnly={isDemo}
          onRequestEdit={isDemo ? () => router.push("/auth") : undefined}
        />
        <AIPanel result={result} isAnalyzing={isAnalyzing} error={error} onReset={reset} />
      </div>
    </AppShell>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={null}>
      <AnalyzeView />
    </Suspense>
  );
}
