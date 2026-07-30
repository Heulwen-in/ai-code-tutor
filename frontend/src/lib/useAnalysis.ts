"use client";

import { useState } from "react";

import { analyzeCode } from "./api";
import { sampleCode } from "./mock-data";
import { useRole } from "./roleStore";
import type { AnalyzeResponse } from "./types";

export function useAnalysis() {
  const { role, setRole } = useRole();
  const [code, setCode] = useState(sampleCode);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runAnalysis() {
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeCode({
        code,
        role,
        language: "python",
      });
      setResult(response);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function reset() {
    setResult(null);
    setError(null);
  }

  return {
    code,
    role,
    result,
    isAnalyzing,
    error,
    setCode,
    setRole,
    runAnalysis,
    reset,
  };
}
