"use client";

import { useState } from "react";

import { analyzeCode } from "./api";
import { sampleCode } from "./mock-data";
import type { AnalyzeResponse, UserRole } from "./types";

export function useAnalysis() {
  const [code, setCode] = useState(sampleCode);
  const [role, setRole] = useState<UserRole>("student");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runAnalysis() {
    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await analyzeCode({
        code,
        role,
        language: "python"
      });
      setResult(response);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return {
    code,
    role,
    result,
    isAnalyzing,
    error,
    setCode,
    setRole,
    runAnalysis
  };
}
