import { mockAnalysis } from "./mock-data";
import type { AnalyzeRequest, AnalyzeResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API !== "false";

export async function analyzeCode(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  if (USE_MOCK_API || !API_BASE_URL) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    return mockAnalysis;
  }

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Unable to analyze code.");
  }

  return response.json();
}
