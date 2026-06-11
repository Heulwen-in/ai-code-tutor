import type { Achievement, AnalyzeResponse, LessonModule, ProgressPoint } from "./types";

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
    description: "The fallback return is valid, but the model highlights edge-case handling for repeated values."
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
    tone: "beginner"
  },
  lessons: [
    {
      lesson_id: "logic_01",
      title: "Trace Loops With Test Cases",
      description: "Learn how to step through loops and catch edge-case mistakes.",
      difficulty: "beginner",
      url: "/lessons#logic_01"
    },
    {
      lesson_id: "dict_01",
      title: "Hash Maps For Faster Search",
      description: "Use dictionaries to replace repeated nested searches.",
      difficulty: "intermediate",
      url: "/lessons#dict_01"
    }
  ]
};

export const lessonModules: LessonModule[] = [
  {
    id: "syntax_01",
    title: "Python Syntax Basics",
    summary: "Colons, brackets, quotes, and the tiny rules that make Python readable.",
    level: "Beginner",
    duration: "12 min",
    bugType: "syntax_error",
    progress: 72,
    example: "if score >= 50:\n    print('Passed')"
  },
  {
    id: "indent_01",
    title: "Indentation Rules",
    summary: "Understand blocks, nested statements, and how to avoid mixed indentation.",
    level: "Beginner",
    duration: "10 min",
    bugType: "indentation_error",
    progress: 64,
    example: "for item in items:\n    print(item)"
  },
  {
    id: "logic_01",
    title: "Debug Logic With Tracing",
    summary: "Use dry runs and targeted examples to reveal off-by-one and branch errors.",
    level: "Intermediate",
    duration: "18 min",
    bugType: "logic_error",
    progress: 38,
    example: "for index in range(len(values)):\n    print(index, values[index])"
  },
  {
    id: "var_01",
    title: "Variable Scope And Naming",
    summary: "Avoid wrong-variable usage with clearer names and smaller functions.",
    level: "Intermediate",
    duration: "15 min",
    bugType: "variable_misuse",
    progress: 45,
    example: "total_price = item_price * quantity"
  }
];

export const progressPoints: ProgressPoint[] = [
  { label: "Mon", score: 34 },
  { label: "Tue", score: 42 },
  { label: "Wed", score: 48 },
  { label: "Thu", score: 57 },
  { label: "Fri", score: 63 },
  { label: "Sat", score: 71 },
  { label: "Sun", score: 78 }
];

export const achievements: Achievement[] = [
  {
    id: "syntax-starter",
    title: "Syntax Starter",
    description: "Fixed 5 syntax or indentation issues.",
    status: "earned",
    value: "5/5"
  },
  {
    id: "loop-detective",
    title: "Loop Detective",
    description: "Completed 3 logic tracing exercises.",
    status: "active",
    value: "2/3"
  },
  {
    id: "steady-learner",
    title: "Steady Learner",
    description: "Analyzed code on 7 different days.",
    status: "active",
    value: "4/7"
  },
  {
    id: "clean-code",
    title: "Clean Code Builder",
    description: "Submit 10 no-bug solutions after the clean-code gate is added.",
    status: "locked",
    value: "0/10"
  }
];
