export type UserRole = "student" | "worker";
export type BugType =
  | "syntax_error"
  | "indentation_error"
  | "logic_error"
  | "variable_misuse"
  | "no_bug";
export type SkillLevel = "novice" | "professional";

export type AnalyzeRequest = {
  code: string;
  role: UserRole;
  language: "python";
};

export type FeedbackResponse = {
  summary: string;
  explanation: string;
  next_steps: string[];
  tone: string;
};

export type LessonItem = {
  lesson_id: string;
  title: string;
  description: string;
  difficulty: string;
  url: string;
};

export type AnalyzeResponse = {
  bug: {
    bug_type: BugType;
    confidence: number;
    line_number: number | null;
    description: string;
  };
  skill: {
    skill_level: SkillLevel;
    confidence: number;
    source: string;
    description: string;
  };
  feedback: FeedbackResponse;
  lessons: LessonItem[];
};

export type LessonModule = {
  id: string;
  title: string;
  summary: string;
  level: "Beginner" | "Intermediate" | "Advanced";
  duration: string;
  bugType: BugType;
  progress: number;
  example: string;
};

export type ProgressPoint = {
  label: string;
  score: number;
};

export type Achievement = {
  id: string;
  title: string;
  description: string;
  status: "earned" | "active" | "locked";
  value: string;
};
