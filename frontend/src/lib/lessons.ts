import type { BugType } from "./types";

export type LessonRole = "student" | "worker";
export type LessonDifficulty = "beginner" | "intermediate" | "advanced";

export type Lesson = {
  id: string;
  title: string;
  summary: string;
  difficulty: LessonDifficulty;
  role: LessonRole;      // student = easy track, worker = professional track
  bugType: BugType;
  duration: string;
  explanation: string;
  buggyCode: string;
  fixedCode: string;
  takeaways: string[];
  // A short practice task whose bug matches the lesson's bugType, so a learner
  // routed here by a detected bug type immediately trains on that same class.
  exercise: { prompt: string; starter: string };
};

// Single source of truth for lesson content. Consumed by the lessons library,
// the lesson detail page, the dashboard, and the recommendation panel.
export const LESSONS: Record<string, Lesson> = {
  // ── Syntax ──────────────────────────────────────────────────────────────────
  syntax_01: {
    id: "syntax_01",
    title: "Python Syntax Basics",
    summary: "Colons, brackets, quotes, and the tiny rules that make Python run.",
    difficulty: "beginner",
    role: "student",
    bugType: "syntax_error",
    duration: "12 min",
    explanation:
      "Python requires strict punctuation. Every function definition needs a colon, every string needs matching quotes, and every open bracket must be closed. A single missing character causes a SyntaxError before your code even runs.",
    buggyCode: `# Missing colon after if, unclosed string
if score >= 50
    print("Passed)`,
    fixedCode: `if score >= 50:
    print("Passed")`,
    takeaways: [
      "Always end def, if, for, while, and else lines with a colon.",
      "Open and close every bracket (), [], {} on the same logical line unless continuing.",
      "Strings must start and end with the same quote character.",
      "Read the error line number — Python usually points directly at or just after the problem.",
    ],
    exercise: {
      prompt: "This function has two syntax errors — a missing colon and a missing quote. Fix both so it runs.",
      starter: `def greet(name)
    print("Hello, + name)`,
    },
  },

  syntax_02: {
    id: "syntax_02",
    title: "Reading Python Error Messages",
    summary: "Decode SyntaxError tracebacks and jump straight to the real problem.",
    difficulty: "beginner",
    role: "student",
    bugType: "syntax_error",
    duration: "10 min",
    explanation:
      "Python SyntaxError messages tell you the file, line number, and a short description. The caret (^) in the traceback points to where the parser gave up — usually one token past the actual mistake.",
    buggyCode: `def greet(name
    print("Hello", name)
# SyntaxError: '(' was never closed`,
    fixedCode: `def greet(name):
    print("Hello", name)`,
    takeaways: [
      "Read the line number in the error, not just the message.",
      "The problem is often on the line ABOVE the one Python flags.",
      "Fix the first error before looking at others — one fix often clears many.",
      "Use a linter (Ruff, Pylint) to see all issues at once before running.",
    ],
    exercise: {
      prompt: "Python reports 'unexpected EOF while parsing'. Find the unclosed bracket and close it.",
      starter: `nums = [1, 2, 3, 4
print(sum(nums))`,
    },
  },

  syntax_pro_01: {
    id: "syntax_pro_01",
    title: "Linting & Static Analysis",
    summary: "Catch syntax and style errors in CI before they ever merge.",
    difficulty: "intermediate",
    role: "worker",
    bugType: "syntax_error",
    duration: "15 min",
    explanation:
      "Professional Python projects run a linter in CI to reject code with syntax or style errors before merging. Ruff is the fastest modern option — it catches syntax errors, unused imports, and PEP 8 violations in milliseconds.",
    buggyCode: `# Ruff would flag all of these
import os, sys
x=1+2
def foo( a,b ):
    return a+b`,
    fixedCode: `import os
import sys

x = 1 + 2


def foo(a, b):
    return a + b`,
    takeaways: [
      "Run `ruff check .` locally before every commit.",
      "Add Ruff to pre-commit hooks so it runs automatically on git commit.",
      "Configure `ruff.toml` or `pyproject.toml` to match your team's style.",
      "Ruff replaces Flake8, isort, and most Pylint rules in a single fast tool.",
    ],
    exercise: {
      prompt: "Rewrite this snippet so Ruff reports zero issues: one import per line, spaces around operators, PEP 8 spacing.",
      starter: `import os,sys
def add( x,y ):return x+y`,
    },
  },

  // ── Indentation ─────────────────────────────────────────────────────────────
  indent_01: {
    id: "indent_01",
    title: "Python Indentation Rules",
    summary: "Blocks, nesting, and how to avoid mixed tabs and spaces.",
    difficulty: "beginner",
    role: "student",
    bugType: "indentation_error",
    duration: "10 min",
    explanation:
      "Python uses indentation (whitespace) to define code blocks instead of curly braces. Every statement inside a def, if, for, or while must be indented by the same amount. Mixing tabs and spaces causes an IndentationError.",
    buggyCode: `def greet():
print("Hello")   # not indented — IndentationError

for i in range(3):
    print(i)
  print("done")  # wrong level — IndentationError`,
    fixedCode: `def greet():
    print("Hello")

for i in range(3):
    print(i)
print("done")`,
    takeaways: [
      "Use 4 spaces per indent level — never tabs.",
      "Every line inside a block must align exactly.",
      "Configure your editor to insert spaces when you press Tab.",
      "The IndentationError line number is where Python lost track — check the line above it too.",
    ],
    exercise: {
      prompt: "The print inside the loop is not indented. Indent it by 4 spaces so it belongs to the loop.",
      starter: `for i in range(3):
print(i)`,
    },
  },

  indent_pro_01: {
    id: "indent_pro_01",
    title: "Editor Config & PEP 8 Compliance",
    summary: "Enforce 4-space indentation across a whole team automatically.",
    difficulty: "intermediate",
    role: "worker",
    bugType: "indentation_error",
    duration: "12 min",
    explanation:
      "Indentation issues in team projects usually come from editors mixing tabs and spaces. An `.editorconfig` file enforces consistent settings across every editor and contributor automatically.",
    buggyCode: `# .editorconfig missing — every dev uses different defaults
# Result: tabs from one dev, spaces from another
# Python raises TabError at runtime`,
    fixedCode: `# .editorconfig at project root
[*.py]
indent_style = space
indent_size = 4
trim_trailing_whitespace = true
insert_final_newline = true`,
    takeaways: [
      "Commit a `.editorconfig` file to enforce indentation project-wide.",
      "Add `ruff format` or `black` to CI to auto-format on every push.",
      "Never use `expand_tabs` manually — let the formatter own it.",
      "PEP 8 requires 4 spaces; no exceptions for professional Python code.",
    ],
    exercise: {
      prompt: "This block mixes a tab and spaces. Re-indent every nested line to a consistent 4 spaces.",
      starter: `def totals(rows):
\ttotal = 0
    for r in rows:
        total += r
    return total`,
    },
  },

  // ── Logic ───────────────────────────────────────────────────────────────────
  logic_01: {
    id: "logic_01",
    title: "Debugging with Print Statements",
    summary: "Trace values step by step to find where logic goes wrong.",
    difficulty: "beginner",
    role: "student",
    bugType: "logic_error",
    duration: "14 min",
    explanation:
      "Logic errors don't crash Python — the code runs but produces the wrong answer. The simplest debugging technique is adding print() calls to trace values at each step so you can see exactly where the result diverges from your expectation.",
    buggyCode: `def total_price(price, qty):
    discount = price * 0.1
    return price * qty - discount   # discount only applied once, not per item`,
    fixedCode: `def total_price(price, qty):
    discount = price * 0.1          # per-item discount
    return (price - discount) * qty`,
    takeaways: [
      "Add print() before and after the suspect line to compare expected vs actual values.",
      "Test with the simplest possible input first (e.g., qty=1) to isolate the problem.",
      "Remove or comment out debug prints before committing.",
      "If the output is always wrong, check the formula. If it's wrong only sometimes, check the conditions.",
    ],
    exercise: {
      prompt: "average() returns the wrong value because it divides before summing correctly. Fix the logic so it returns the true mean.",
      starter: `def average(nums):
    total = 0
    for n in nums:
        total += n
        avg = total / len(nums)
    return avg`,
    },
  },

  logic_02: {
    id: "logic_02",
    title: "Understanding Loops & Conditions",
    summary: "Beat off-by-one errors and wrong boundary conditions.",
    difficulty: "beginner",
    role: "student",
    bugType: "logic_error",
    duration: "16 min",
    explanation:
      "The most common logic bugs in loops are off-by-one errors (iterating one step too many or too few) and wrong boundary conditions in comparisons. Drawing a small table of values for each iteration is the fastest way to catch these.",
    buggyCode: `# Off-by-one: should sum indices 0 to n-1
def sum_first_n(nums, n):
    total = 0
    for i in range(1, n + 1):   # starts at 1, misses index 0
        total += nums[i]
    return total`,
    fixedCode: `def sum_first_n(nums, n):
    total = 0
    for i in range(0, n):       # range(n) is cleaner
        total += nums[i]
    return total`,
    takeaways: [
      "range(n) gives 0 to n-1. range(1, n+1) gives 1 to n — know which you need.",
      "Trace the first and last iteration by hand before running.",
      "Use len(list) - 1 as the last valid index, not len(list).",
      "For boundary checks: draw a number line and mark the edge values.",
    ],
    exercise: {
      prompt: "count_to(n) should print 1 through n, but it stops one short. Fix the range boundary.",
      starter: `def count_to(n):
    for i in range(1, n):
        print(i)`,
    },
  },

  logic_pro_01: {
    id: "logic_pro_01",
    title: "Unit Testing with pytest",
    summary: "Catch logic regressions automatically on every change.",
    difficulty: "advanced",
    role: "worker",
    bugType: "logic_error",
    duration: "20 min",
    explanation:
      "Logic errors that slip past manual inspection are caught by automated tests. pytest lets you write small focused tests that run on every code change, so regressions are caught immediately rather than in production.",
    buggyCode: `def divide(a, b):
    return a / b   # no guard for b=0

# No tests — bug only found when a user triggers it`,
    fixedCode: `def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# test_divide.py
def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    import pytest
    with pytest.raises(ValueError):
        divide(10, 0)`,
    takeaways: [
      "Write one test per behaviour, not one test per function.",
      "Always test the happy path AND the edge cases (zero, empty, negative).",
      "Run `pytest` before every commit — add it to your pre-commit hook.",
      "Aim for tests that fail for exactly one reason so failures are easy to diagnose.",
    ],
    exercise: {
      prompt: "Write a failing-then-passing test: add a pytest test that asserts is_even(4) is True and is_even(3) is False.",
      starter: `def is_even(n):
    return n % 2 == 0

# test_is_even.py — write your assertions below
def test_is_even():
    ...`,
    },
  },

  logic_pro_02: {
    id: "logic_pro_02",
    title: "Using the Python Debugger (pdb)",
    summary: "Pause execution and inspect state to isolate complex bugs.",
    difficulty: "intermediate",
    role: "worker",
    bugType: "logic_error",
    duration: "15 min",
    explanation:
      "pdb lets you pause execution, inspect variables, step line-by-line, and evaluate expressions interactively. It is faster than adding and removing print statements for complex logic bugs.",
    buggyCode: `def find_max(nums):
    max_val = 0            # bug: fails for all-negative lists
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val`,
    fixedCode: `def find_max(nums):
    max_val = nums[0]             # correct: start from first element
    for n in nums[1:]:
        if n > max_val:
            max_val = n
    return max_val`,
    takeaways: [
      "Use `breakpoint()` (Python 3.7+) instead of `import pdb; pdb.set_trace()`.",
      "Key commands: n=next line, s=step into, p=print var, c=continue, q=quit.",
      "VS Code and PyCharm have graphical debuggers — same concepts, easier interface.",
      "Remove all breakpoint() calls before committing.",
    ],
    exercise: {
      prompt: "find_max returns 0 for all-negative input. Fix the initial value so it works for any list.",
      starter: `def find_max(nums):
    max_val = 0
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val`,
    },
  },

  // ── Variable misuse ──────────────────────────────────────────────────────────
  var_01: {
    id: "var_01",
    title: "Variables & Scope in Python",
    summary: "Avoid NameError with clear scope and correct initialisation.",
    difficulty: "beginner",
    role: "student",
    bugType: "variable_misuse",
    duration: "13 min",
    explanation:
      "A NameError means Python cannot find the variable you referenced. This happens when a variable is only defined inside an if block that didn't run, defined inside a function but used outside, or simply misspelled.",
    buggyCode: `def calculate(x):
    if x > 0:
        result = x * 2
    return result   # NameError if x <= 0 — result was never defined`,
    fixedCode: `def calculate(x):
    result = 0          # define before the branch
    if x > 0:
        result = x * 2
    return result`,
    takeaways: [
      "Always initialise variables before conditional blocks that might set them.",
      "Variables defined inside a function do not exist outside it.",
      "Check spelling — Python is case-sensitive: total is not Total.",
      "If you get NameError, search for where the variable is first assigned.",
    ],
    exercise: {
      prompt: "This function accumulates into the wrong variable name so the total is always 0. Use the correct name.",
      starter: `def add_all(items):
    total = 0
    for item in items:
        totl += item
    return total`,
    },
  },

  var_pro_01: {
    id: "var_pro_01",
    title: "Type Hints & mypy",
    summary: "Catch variable and type misuse statically, before runtime.",
    difficulty: "advanced",
    role: "worker",
    bugType: "variable_misuse",
    duration: "18 min",
    explanation:
      "Type hints let you declare what type a variable or parameter should be. mypy reads these annotations statically — before you run the code — and flags type mismatches that would cause AttributeError or incorrect behaviour at runtime.",
    buggyCode: `# No type hints — misuse found only at runtime
def process(items):
    return items.upper()   # crashes if items is a list, not a str`,
    fixedCode: `# With type hints — mypy catches the bug before runtime
def process(items: str) -> str:
    return items.upper()`,
    takeaways: [
      "Add type hints to all function signatures: `def f(x: int) -> str:`.",
      "Run `mypy .` in CI to block merges with type errors.",
      "Use `list[str]`, `dict[str, int]`, `Optional[str]` from built-ins (Python 3.10+).",
      "Type hints are documentation that the computer can verify — use them.",
    ],
    exercise: {
      prompt: "Add type hints so mypy would flag a caller passing an int. The function should take and return a str.",
      starter: `def shout(text):
    return text.upper() + "!"`,
    },
  },

  // ── No bug ──────────────────────────────────────────────────────────────────
  good_01: {
    id: "good_01",
    title: "Keep it Up — Next Challenges",
    summary: "Your code is clean — now make it simpler and more Pythonic.",
    difficulty: "beginner",
    role: "student",
    bugType: "no_bug",
    duration: "10 min",
    explanation:
      "Your code passed the classifier with no major bug detected. That is a great sign. The next step is to challenge yourself with harder problems that stress-test your understanding of edge cases, efficiency, and code clarity.",
    buggyCode: `# Clean code — but can you improve it?
def contains(nums, target):
    for n in nums:
        if n == target:
            return True
    return False`,
    fixedCode: `# Pythonic version using 'in'
def contains(nums, target):
    return target in nums`,
    takeaways: [
      "Clean code is a starting point — always ask if there is a simpler or faster way.",
      "Use Python built-ins (in, any, all, sum) before writing manual loops.",
      "Try solving LeetCode Easy problems to practise writing correct code under constraints.",
      "Read your own code after 10 minutes — if it needs a comment to be understood, rename the variable instead.",
    ],
    exercise: {
      prompt: "Rewrite this loop as a single Pythonic line using a built-in.",
      starter: `def all_positive(nums):
    for n in nums:
        if n <= 0:
            return False
    return True`,
    },
  },

  good_pro_01: {
    id: "good_pro_01",
    title: "Code Review Best Practices",
    summary: "Move past 'passes the linter' to genuinely reviewable code.",
    difficulty: "advanced",
    role: "worker",
    bugType: "no_bug",
    duration: "16 min",
    explanation:
      "Passing the linter is necessary but not sufficient. Good code review checks for correctness at the boundary, clarity for the next reader, and whether the change can be tested in isolation.",
    buggyCode: `# Passes linter but has review issues:
def get_user(id):
    return db.query(f"SELECT * FROM users WHERE id={id}")  # SQL injection
    # no error handling, no type hints, SELECT * is lazy`,
    fixedCode: `from sqlalchemy.orm import Session
from app.models import User

def get_user(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()`,
    takeaways: [
      "Never interpolate user input into SQL strings — always use parameterised queries.",
      "SELECT * couples your code to the schema — select only the columns you need.",
      "Every public function should have a type signature and handle None/empty cases.",
      "A good review comment explains *why* something is wrong, not just that it is.",
    ],
    exercise: {
      prompt: "Rewrite this query to avoid SQL injection by using a parameterised ORM query instead of string interpolation.",
      starter: `def find_user(db, name):
    return db.execute(f"SELECT * FROM users WHERE name = '{name}'")`,
    },
  },
};

export const LESSON_LIST: Lesson[] = Object.values(LESSONS);
