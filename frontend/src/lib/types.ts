export type UserRole = "student" | "worker";

export type User = {
  id: number;
  email: string;
  role: UserRole;
  display_name?: string | null;
};

export type AuthToken = {
  access_token: string;
  token_type: string;
};
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
  source: string;
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
    bug_subtype: string | null;
    subtype_confidence: number | null;
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

