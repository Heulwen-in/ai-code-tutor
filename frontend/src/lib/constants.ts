import type { BugType } from "./types";

// Per bug type: label, colors, and the Material Symbols icon shared by the
// results panel, cards, and the Progress legend.
export const BUG_CONFIG: Record<
  BugType,
  { label: string; color: string; bg: string; grad: string; icon: string }
> = {
  logic_error: {
    label: "Logic Error",
    color: "#B45309",
    bg: "#FEF9C3",
    grad: "linear-gradient(90deg,#F59E0B,#B45309)",
    icon: "psychology",
  },
  syntax_error: {
    label: "Syntax Error",
    color: "#DC2626",
    bg: "#FEE2E2",
    grad: "linear-gradient(90deg,#F87171,#DC2626)",
    icon: "code_off",
  },
  variable_misuse: {
    label: "Variable Misuse",
    color: "#7C3AED",
    bg: "#EDE9FE",
    grad: "linear-gradient(90deg,#A855F7,#7C3AED)",
    icon: "data_object",
  },
  indentation_error: {
    label: "Indentation Error",
    color: "#0369A1",
    bg: "#DBEAFE",
    grad: "linear-gradient(90deg,#60A5FA,#0369A1)",
    icon: "format_indent_increase",
  },
  no_bug: {
    label: "No Bug",
    color: "#059669",
    bg: "#DCFCE7",
    grad: "linear-gradient(90deg,#34D399,#059669)",
    icon: "check_circle",
  },
};

// Preset snippets loaded from the "Load Example" menu in the Analyse toolbar.
export const EXAMPLE_SNIPPETS: { label: string; code: string }[] = [
  {
    label: "Logic Error",
    code: `def get_row(rowIndex):
    row = [1] * (rowIndex + 1)
    for i in range(1, rowIndex + 1):
        for j in range(i - 1, 0, -1):
            row[j] += row[j - 1]
    return`,
  },
  {
    label: "Syntax Error",
    code: `def greet(name)
    msg = "Hello, " + name
    print(msg)`,
  },
  {
    label: "Variable Misuse",
    code: `def calculate_total(items):
    totl = 0
    for item in items:
        total += item
    return total`,
  },
  {
    label: "Indentation Error",
    code: `def process(data):
    for item in data:
        if item > 0:
        print(item)
    return data`,
  },
  {
    label: "No Bug",
    code: `def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []`,
  },
];
