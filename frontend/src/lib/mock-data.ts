import type { AnalyzeResponse } from "./types";

export const sampleCode = `def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []`;

export const mockAnalysis: AnalyzeResponse = {
  bug: {
    bug_type: "logic_error",
    confidence: 0.91,
    line_number: 5,
    description: "The fallback return is valid, but the model highlights edge-case handling for repeated values.",
    bug_subtype: "off_by_one_index",
    subtype_confidence: 0.88
  },
  skill: {
    skill_level: "novice",
    confidence: 0.84,
    source: "mock",
    description: "Skill level inferred from code structure and naming patterns."
  },
  feedback: {
    summary: "Possible logic issue detected around the search loop.",
    explanation:
      "Your nested-loop approach is readable and correct for small inputs. For a stronger solution, consider using a dictionary to store previously seen values and reduce the time complexity.",
    next_steps: [
      "Test the code with duplicate numbers and negative values.",
      "Compare the O(n^2) loop with a dictionary-based O(n) version.",
      "Write one small comment explaining why the fallback return is safe."
    ],
    tone: "beginner",
    source: "mock"
  },
  lessons: [
    {
      lesson_id: "logic_01",
      title: "Debugging with Print Statements",
      description: "Learn how to step through loops and catch edge-case mistakes.",
      difficulty: "beginner",
      url: "/lessons/logic_01"
    },
    {
      lesson_id: "logic_02",
      title: "Understanding Loops & Conditions",
      description: "Common loop mistakes: off-by-one, infinite loops, wrong conditions.",
      difficulty: "beginner",
      url: "/lessons/logic_02"
    }
  ]
};
