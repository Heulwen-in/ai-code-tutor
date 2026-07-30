# System Test Snippets (from test4_v3 — unseen problems)

Paste each into the analyzer. 'Expected' = ground-truth label.

## syntax_error_1

**Expected:** `syntax_error`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `syntax_error` (conf 1.000, subtype incorrect_type, line 2)

```python
def reverse_string(s):
    return s[::-"1"]
```

## syntax_error_2

**Expected:** `syntax_error`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `syntax_error` (conf 1.000, subtype incorrect_type, line 2)

```python
def min_operations(n):
    return n * n // "4"
```

## syntax_error_3

**Expected:** `syntax_error`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `syntax_error` (conf 1.000, subtype incorrect_type, line 2)

```python
def addDigits(num: int) -> int:
    return "1" + (num - "1") % "9"
```

## logic_error_1

**Expected:** `logic_error`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `logic_error` (conf 1.000, subtype returning_early, line 2)

```python
def sortByBits(arr):
    return
```

## logic_error_2

**Expected:** `logic_error`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `logic_error` (conf 1.000, subtype returning_early, line 2)

```python
def min_operations(n):
    return
```

## logic_error_3

**Expected:** `logic_error`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `logic_error` (conf 1.000, subtype returning_early, line 2)

```python
def reverse_string(s):
    return
```

## variable_misuse_1

**Expected:** `variable_misuse`  |  **Buggy line(s):** [np.int64(1), np.int64(2)]

**System returned:** `variable_misuse` (conf 1.000, subtype mutable_default_argument, line 1)

```python
def min_operations(n = []):
    n.append(1)
    return n * n // 4
```

## variable_misuse_2

**Expected:** `variable_misuse`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `variable_misuse` (conf 1.000, subtype forgotten_variable_update, line 2)

```python
def truncate_sentence(s, k):
    words = words
    return ' '.join(words[:k])
```

## variable_misuse_3

**Expected:** `variable_misuse`  |  **Buggy line(s):** [np.int64(2)]

**System returned:** `variable_misuse` (conf 1.000, subtype use_before_definition, line 2)

```python
def count_odds(low, high):
    lowhigh = low + 1
    return (high + 1) // 2 - low // 2
```

## no_bug_1

**Expected:** `no_bug`

**System returned:** `no_bug` (conf 0.999, line 1)

```python
(A). negative
```

## no_bug_2

**Expected:** `no_bug`

**System returned:** `no_bug` (conf 0.999, line 1)

```python
import pandas as pd
```

## no_bug_3

**Expected:** `no_bug`

**System returned:** `no_bug` (conf 0.999, line 1)

```python
message = "Hello World!"
```

