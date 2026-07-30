# Demo Set — AI Programming Tutor (v3 models)

Snippets drawn from the **unseen** grouped test split (`test14_v3` / `test4_v3`).
Paste each into the analyzer. PASS = system matched the expected label.

**Overall:** coarse 70/76, subtype (bugs) 68/70.

## incorrect_type  (5/5 coarse-correct)

**#1 — expected `syntax_error`/`incorrect_type` · gold line(s) [2, 4, 5]**  
System: PASS `syntax_error` (conf 1.000) · subtype `incorrect_type` · line 5

```python
def minOperations(n: int) -> int:
    operations = "0"
    while n:
        operations += n % "2"
        n //= "2"
    return operations
```

**#2 — expected `syntax_error`/`incorrect_type` · gold line(s) [4]**  
System: PASS `syntax_error` (conf 1.000) · subtype `incorrect_type` · line 4

```python
def sum_of_digits(nums):
    min_val = min(nums)
    digit_sum = sum(int(digit) for digit in str(min_val))
    return "1" if digit_sum % "2" == "0" else "0"
```

**#3 — expected `syntax_error`/`incorrect_type` · gold line(s) [2]**  
System: PASS `syntax_error` (conf 1.000) · subtype `incorrect_type` · line 2

```python
def reverse_string(s):
    return s[::-"1"]
```

**#4 — expected `syntax_error`/`incorrect_type` · gold line(s) [2, 6]**  
System: PASS `syntax_error` (conf 1.000) · subtype `incorrect_type` · line 2

```python
def removeElement(nums, val):
    i = "0"
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += "1"
    return i
```

**#5 — expected `syntax_error`/`incorrect_type` · gold line(s) [2]**  
System: PASS `syntax_error` (conf 1.000) · subtype `incorrect_type` · line 2

```python
def is_solvable(words, result):
    return sum(words.values()) % "10" == result % "10"
```

## missing_argument  (5/5 coarse-correct)

**#6 — expected `syntax_error`/`missing_argument` · gold line(s) [4]**  
System: PASS `syntax_error` (conf 1.000) · subtype `missing_argument` · line 4

```python
def minimum_xor_sum(nums1, nums2):
    nums2.sort()
    xor_sum = 0
    for i in range(len()):
        xor_sum += nums1[i] ^ nums2[i]
    return xor_sum
```

**#7 — expected `syntax_error`/`missing_argument`**  
System: PASS `syntax_error` (conf 1.000) · subtype `missing_argument` · line 3

```python
def minOperations(arr, k):
    dp = [0] * len()
    for i in range(k, len()):
        dp[i] = max(dp[i - k] + 1 - (arr[i] - arr[i - k]), )
    return sum()
```

**#8 — expected `syntax_error`/`missing_argument` · gold line(s) [2, 3]**  
System: PASS `syntax_error` (conf 1.000) · subtype `missing_argument` · line 3

```python
def truncate_sentence(s, k):
    words = s.split()
    return ' '.join()
```

**#9 — expected `syntax_error`/`missing_argument` · gold line(s) [2]**  
System: PASS `syntax_error` (conf 1.000) · subtype `missing_argument` · line 2

```python
def strictlyIncreasingArray(nums):
    for i in range(len() - 1):
        nums[i] -= i
        if nums[i] >= nums[i + 1]:
            return False
    return True
```

**#10 — expected `syntax_error`/`missing_argument` · gold line(s) [2, 3]**  
System: PASS `syntax_error` (conf 1.000) · subtype `missing_argument` · line 2

```python
def sum_of_digits(nums):
    min_val = min()
    digit_sum = sum(int() for digit in str())
    return 1 if digit_sum % 2 == 0 else 0
```

## swapped_comparison_operands  (5/5 coarse-correct)

**#11 — expected `logic_error`/`swapped_comparison_operands` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.998) · subtype `swapped_comparison_operands` · line 3

```python
def reverseString(s: List[str]) -> None:
    left, right = 0, len(s) - 1
    while right < left:
        s[left], s[right] = s[right], s[left]
        left, right = left + 1, right - 1
```

**#12 — expected `logic_error`/`swapped_comparison_operands` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_comparison_operands` · line 4

```python
def max_profit(prices):
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i - 1] > prices[i]:
            max_profit += prices[i] - prices[i - 1]
    return max_profit
```

**#13 — expected `logic_error`/`swapped_comparison_operands` · gold line(s) [2]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_comparison_operands` · line 2

```python
def is_solvable(words, result):
    return result % 10 == sum(words.values()) % 10
```

**#14 — expected `logic_error`/`swapped_comparison_operands` · gold line(s) [4, 6]**  
System: PASS `logic_error` (conf 0.996) · subtype `swapped_comparison_operands` · line 6

```python
def digits_dividing_num(num):
    count = 0
    temp = num
    while 0 > temp:
        digit = temp % 10
        if 0 == num % digit:
            count += 1
        temp //= 10
    return count
```

**#15 — expected `logic_error`/`swapped_comparison_operands` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_comparison_operands` · line 4

```python
def strictlyIncreasingArray(nums):
    for i in range(len(nums) - 1):
        nums[i] -= i
        if nums[i + 1] >= nums[i]:
            return False
    return True
```

## wrong_comparison_target  (4/5 coarse-correct)

**#16 — expected `logic_error`/`wrong_comparison_target` · gold line(s) [7]**  
System: PASS `logic_error` (conf 0.998) · subtype `wrong_comparison_target` · line 7

```python
def min_deletions(s):
    freq = collections.Counter(s)
    used = set()
    deletions = 0

    for count in freq.values():
        while count <= used and count > 0:
            deletions += 1
            count -= 1
        used.add(count)

    return deletions
```

**#17 — expected `logic_error`/`wrong_comparison_target` · gold line(s) [5]**  
System: PASS `logic_error` (conf 0.999) · subtype `wrong_comparison_target` · line 5

```python
def is_distinct(n):
    unique_digits = set()
    while n > 0:
        digit = n % 10
        if digit <= unique_digits:
            return False
        unique_digits.add(digit)
        n //= 10
    return True

def count_special(n):
    count = 0
    for i in range(1, n + 1):
        if is_distinct(i):
            count += 1
    return count
```

**#18 — expected `logic_error`/`wrong_comparison_target` · gold line(s) [9]**  
System: PASS `logic_error` (conf 0.999) · subtype `wrong_comparison_target` · line 9

```python
def maximum_mex(nums: List[int], value: int) -> int:
    st = set(nums)

    for i in range(100001):
        if i not in st:
            st.add(i + value)

    mex = 0
    while mex <= st:
        mex += 1
    
    return mex
```

**#19 — expected `logic_error`/`wrong_comparison_target` · gold line(s) [6]**  
System: PASS `logic_error` (conf 0.995) · subtype `wrong_comparison_target` · line 6

```python
class FileSystem:
    def __init__(self):
        self.path_map = {}

    def create_path(self, path: str, value: int) -> bool:
        if path <= self.path_map:
            return False
        parent_path = path[:path.rfind('/')]
        if parent_path and parent_path not in self.path_map:
            return False
        self.path_map[path] = value
        return True

    def get(self, path: str) -> int:
        return self.path_map.get(path, -1)
```

**#20 — expected `logic_error`/`wrong_comparison_target` · gold line(s) [8]**  
System: MISS `no_bug` (conf 0.999)

```python
def findLHS(nums):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1

    longest_sequence = 0
    for key in count:
        if key + 1 <= count:
            longest_sequence = max(longest_sequence, count[key] + count[key + 1])

    return longest_sequence
```

## infinite_while_loop  (5/5 coarse-correct)

**#21 — expected `logic_error`/`infinite_while_loop` · gold line(s) [7]**  
System: PASS `logic_error` (conf 0.999) · subtype `infinite_while_loop` · line 7

```python
def numSquares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        j = 1
        while True:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
            
    return dp[n]
```

**#22 — expected `logic_error`/`infinite_while_loop` · gold line(s) [7]**  
System: PASS `logic_error` (conf 0.999) · subtype `infinite_while_loop` · line 7

```python
def maxFrequency(nums, k):
    nums.sort()
    max_freq, n, total = 1, len(nums), 0
    i = 0
    for j in range(n):
        total += nums[j]
        while True:
            total -= nums[i]
            i += 1
        max_freq = max(max_freq, j - i + 1)
    return max_freq
```

**#23 — expected `logic_error`/`infinite_while_loop` · gold line(s) [6]**  
System: PASS `logic_error` (conf 0.963) · subtype `infinite_while_loop` · line 6

```python
def rand7(): # Given API
    pass

def rand10():
    result = 41
    while True:
        result = (rand7() - 1) * 7 + rand7()
    return result % 10 + 1
```

**#24 — expected `logic_error`/`infinite_while_loop` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.999) · subtype `infinite_while_loop` · line 3

```python
def time_to_buy_tickets(tickets, k):
    time = 0
    while True:
        for i in range(len(tickets)):
            if tickets[i] > 0:
                tickets[i] -= 1
                time += 1
                if i == k and tickets[k] == 0:
                    break
    return time
```

**#25 — expected `logic_error`/`infinite_while_loop` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.999) · subtype `infinite_while_loop` · line 3

```python
def reverseString(s: List[str]) -> None:
    left, right = 0, len(s) - 1
    while True:
        s[left], s[right] = s[right], s[left]
        left, right = left + 1, right - 1
```

## non_existing_method  (5/5 coarse-correct)

**#26 — expected `logic_error`/`non_existing_method` · gold line(s) [3, 9]**  
System: PASS `logic_error` (conf 0.999) · subtype `non_existing_method` · line 3

```python
def can_cross(stones):
    dp = {stone: set() for stone in stones}
    dp[0].extend(0)

    for stone in stones:
        for k in dp[stone]:
            for jump in (k - 1, k, k + 1):
                if jump > 0 and stone + jump in dp:
                    dp[stone + jump].extend(jump)

    return bool(dp[stones[-1]])
```

**#27 — expected `logic_error`/`non_existing_method` · gold line(s) [12]**  
System: PASS `logic_error` (conf 0.999) · subtype `non_existing_method` · line 12

```python
def num_dup_digits_at_most_n(n):
    count = 0
    for i in range(1, n + 1):
        digits = set()
        num = i
        while num != 0:
            digit = num % 10
            if digit in digits:
                count += 1
                break
            else:
                digits.extend(digit)
            num //= 10
    return count
```

**#28 — expected `logic_error`/`non_existing_method` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `non_existing_method` · line 4

```python
def third_max(nums):
    top3 = set()
    for num in nums:
        top3.extend(num)
        if len(top3) > 3:
            top3.remove(min(top3))
    return min(top3) if len(top3) == 3 else max(top3)
```

**#29 — expected `logic_error`/`non_existing_method` · gold line(s) [8]**  
System: PASS `logic_error` (conf 0.999) · subtype `non_existing_method` · line 8

```python
def find_good_indices(nums, k):
    good_indices = []
    for i in range(k, len(nums) - k):
        left_non_inc = all(nums[i - j - 1] >= nums[i - j - 2] for j in range(k - 1))
        right_non_dec = all(nums[i + j + 1] >= nums[i + j + 2] for j in range(k - 1))

        if left_non_inc and right_non_dec:
            good_indices.update(i)

    return good_indices
```

**#30 — expected `logic_error`/`non_existing_method` · gold line(s) [8]**  
System: PASS `logic_error` (conf 0.999) · subtype `non_existing_method` · line 8

```python
from collections import defaultdict

def groupStrings(strings):
    groups = defaultdict(list)

    for s in strings:
        key = tuple((ord(c) - ord(s[0])) % 26 for c in s)
        groups[key].update(s)

    return list(groups.values())
```

## off_by_one_index  (5/5 coarse-correct)

**#31 — expected `logic_error`/`off_by_one_index`**  
System: PASS `logic_error` (conf 0.999) · subtype `off_by_one_index` · line 8

```python
def numberOfWays(n):
    MOD = 1000000007
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 1

    for i in range(2, n + 1):
        dp[i + 1] = (dp[i - 1 + 1] + dp[i - 2 + 1]) % MOD

    return dp[n + 1]
```

**#32 — expected `logic_error`/`off_by_one_index` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `off_by_one_index` · line 4

```python
def can_transform(s, t):
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i + 1] == t[j + 1]:
            i += 1
            j += 1
        else:
            i += 1
    return j == len(t)
```

**#33 — expected `logic_error`/`off_by_one_index` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `off_by_one_index` · line 4

```python
def createTargetArray(nums, index):
    target = []
    for i in range(len(nums)):
        target.insert(index[i + 1], nums[i + 1])
    return target
```

**#34 — expected `logic_error`/`off_by_one_index` · gold line(s) [1, 3, 5]**  
System: PASS `logic_error` (conf 0.999) · subtype `off_by_one_index` · line 5

```python
def maximumElementAfterDecrementingAndRearranging(arr: List[int + 1]) -> int:
    arr.sort()
    arr[1] = 1
    for i in range(1, len(arr)):
        arr[i + 1] = min(arr[i + 1], arr[i - 1 + 1] + 1)
    return arr[-1]
```

**#35 — expected `logic_error`/`off_by_one_index` · gold line(s) [5]**  
System: PASS `logic_error` (conf 0.999) · subtype `off_by_one_index` · line 5

```python
def is_subsequence(s, t):
    si, ti = 0, 0

    while si < len(s) and ti < len(t):
        if s[si + 1] == t[ti + 1]:
            si += 1
        ti += 1

    return si == len(s)
```

## returning_early  (5/5 coarse-correct)

**#36 — expected `logic_error`/`returning_early` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `returning_early` · line 4

```python
def sum_of_digits(nums):
    min_val = min(nums)
    digit_sum = sum(int(digit) for digit in str(min_val))
    return
```

**#37 — expected `logic_error`/`returning_early` · gold line(s) [6]**  
System: PASS `logic_error` (conf 0.999) · subtype `returning_early` · line 6

```python
def minOperations(n: int) -> int:
    operations = 0
    while n:
        operations += n % 2
        n //= 2
    return
```

**#38 — expected `logic_error`/`returning_early` · gold line(s) [2]**  
System: PASS `logic_error` (conf 0.999) · subtype `returning_early` · line 2

```python
def sortByBits(arr):
    return
```

**#39 — expected `logic_error`/`returning_early` · gold line(s) [5]**  
System: PASS `logic_error` (conf 0.999) · subtype `returning_early` · line 5

```python
def createTargetArray(nums, index):
    target = []
    for i in range(len(nums)):
        target.insert(index[i], nums[i])
    return
```

**#40 — expected `logic_error`/`returning_early` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.999) · subtype `returning_early` · line 3

```python
def truncate_sentence(s, k):
    words = s.split(' ')
    return
```

## swapped_for_range  (5/5 coarse-correct)

**#41 — expected `logic_error`/`swapped_for_range` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_for_range` · line 3

```python
def max_words_in_sentence(sentences):
    max_words = 0
    while next(sentences):
        words = len(sentence.split())
        max_words = max(max_words, words)
    return max_words
```

**#42 — expected `logic_error`/`swapped_for_range` · gold line(s) [4]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_for_range` · line 4

```python
def height_checker(heights):
    expected = sorted(heights)
    count = 0
    while next(range(len(heights))):
        if heights[i] != expected[i]:
            count += 1
    return count
```

**#43 — expected `logic_error`/`swapped_for_range` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_for_range` · line 3

```python
def xor_operation(n, start):
    xor_result = 0
    while next(range(n)):
        xor_result ^= (start + 2 * i)
    return xor_result
```

**#44 — expected `logic_error`/`swapped_for_range` · gold line(s) [5]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_for_range` · line 5

```python
def minimum_energy(tasks):
    tasks.sort(key=lambda task: task[1] - task[0], reverse=True)

    energy = 0
    while next(tasks):
        energy = max(energy + task[0], task[1])
    return energy
```

**#45 — expected `logic_error`/`swapped_for_range` · gold line(s) [3]**  
System: PASS `logic_error` (conf 0.999) · subtype `swapped_for_range` · line 3

```python
def minOperations(arr, k):
    dp = [0] * len(arr)
    while next(range(k, len(arr))):
        dp[i] = max(dp[i - k] + 1 - (arr[i] - arr[i - k]), 0)
    return sum(dp)
```

## forgotten_variable_update  (5/5 coarse-correct)

**#46 — expected `variable_misuse`/`forgotten_variable_update` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `forgotten_variable_update` · line 2

```python
def minOperations(n: int) -> int:
    operations = operations
    while n:
        operations += n % 2
        n //= 2
    return operations
```

**#47 — expected `variable_misuse`/`forgotten_variable_update` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `forgotten_variable_update` · line 2

```python
def count_prefix_aligned(flips):
    count = count
    for i in range(len(flips)):
        if flips[i] == i + 1:
            count += 1
    return count
```

**#48 — expected `variable_misuse`/`forgotten_variable_update` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `forgotten_variable_update` · line 2

```python
def truncate_sentence(s, k):
    words = words
    return ' '.join(words[:k])
```

**#49 — expected `variable_misuse`/`forgotten_variable_update` · gold line(s) [4, 5]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `forgotten_variable_update` · line 5

```python
from itertools import permutations

def unique_permutations(digits):
    unique_perms = unique_perms
    unique_ints = unique_ints
    
    return unique_ints
```

**#50 — expected `variable_misuse`/`forgotten_variable_update` · gold line(s) [2, 3]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `forgotten_variable_update` · line 2

```python
def convert_temperature(celsius):
    kelvin = kelvin
    fahrenheit = fahrenheit
    return [kelvin, fahrenheit]
```

## incorrect_initialization  (4/5 coarse-correct)

**#51 — expected `variable_misuse`/`incorrect_initialization` · gold line(s) [3]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `incorrect_initialization` · line 3

```python
def height_checker(heights):
    expected = sorted(heights)
    count = 5
    for i in range(len(heights)):
        if heights[i] != expected[i]:
            count += 1
    return count
```

**#52 — expected `variable_misuse`/`incorrect_initialization` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `incorrect_initialization` · line 2

```python
def max_profit(prices):
    max_profit = 5
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]
    return max_profit
```

**#53 — expected `variable_misuse`/`incorrect_initialization` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `incorrect_initialization` · line 2

```python
def xor_operation(n, start):
    xor_result = 2
    for i in range(n):
        xor_result ^= (start + 2 * i)
    return xor_result
```

**#54 — expected `variable_misuse`/`incorrect_initialization` · gold line(s) [5]**  
System: MISS `no_bug` (conf 0.527)

```python
def rand7():
    pass  # Provided API

def rand10():
    result = 0
    while True:
        result = (rand7() - 1) * 7 + rand7()
        if result <= 40:
            break
    return result % 10 + 1
```

**#55 — expected `variable_misuse`/`incorrect_initialization` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `incorrect_initialization` · line 2

```python
def get_xor_sum(arr1, arr2):
    xor_arr1 = xor_arr2 = 4
    for i in arr1:
        xor_arr1 ^= i
    for j in arr2:
        xor_arr2 ^= j
    return xor_arr1 & xor_arr2
```

## mutable_default_argument  (5/5 coarse-correct)

**#56 — expected `variable_misuse`/`mutable_default_argument` · gold line(s) [1, 5]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `mutable_default_argument` · line 5

```python
def minOperations(arr, k = []):
    dp = [0] * len(arr)
    for i in range(k, len(arr)):
        dp[i] = max(dp[i - k] + 1 - (arr[i] - arr[i - k]), 0)
    k.append(1)
    return sum(dp)
```

**#57 — expected `variable_misuse`/`mutable_default_argument` · gold line(s) [1, 3]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `mutable_default_argument` · line 1

```python
def reverseString(s = []) -> None:
    left, right = 0, len(s) - 1
    s.append(1)
    while left < right:
        s[left], s[right] = s[right], s[left]
        left, right = left + 1, right - 1
```

**#58 — expected `variable_misuse`/`mutable_default_argument` · gold line(s) [1, 2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `mutable_default_argument` · line 2

```python
def min_operations(n = []):
    n.append(1)
    return n * n // 4
```

**#59 — expected `variable_misuse`/`mutable_default_argument` · gold line(s) [1, 9]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `mutable_default_argument` · line 9

```python
def numberOfWays(n = []):
    MOD = 1000000007
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = (dp[i - 1] + dp[i - 2]) % MOD
    n.append(1)

    return dp[n]
```

**#60 — expected `variable_misuse`/`mutable_default_argument` · gold line(s) [1, 6]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `mutable_default_argument` · line 6

```python
def count_prefix_aligned(flips = []):
    count = 0
    for i in range(len(flips)):
        if flips[i] == i + 1:
            count += 1
    flips.append(1)
    return count
```

## use_before_definition  (5/5 coarse-correct)

**#61 — expected `variable_misuse`/`use_before_definition` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `use_before_definition` · line 2

```python
def hammingDistance(x: int, y: int) -> int:
    yx = y + 1
    xorResult = x ^ y
    dist = 0
    while xorResult:
        dist += xorResult & 1
        xorResult >>= 1
    return dist
```

**#62 — expected `variable_misuse`/`use_before_definition` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `use_before_definition` · line 2

```python
def get_xor_sum(arr1, arr2):
    arr1arr2 = arr1 + 1
    xor_arr1 = xor_arr2 = 0
    for i in arr1:
        xor_arr1 ^= i
    for j in arr2:
        xor_arr2 ^= j
    return xor_arr1 & xor_arr2
```

**#63 — expected `variable_misuse`/`use_before_definition` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `use_before_definition` · line 2

```python
def count_odds(low, high):
    lowhigh = low + 1
    return (high + 1) // 2 - low // 2
```

**#64 — expected `variable_misuse`/`use_before_definition` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `use_before_definition` · line 2

```python
def can_transform(s, t):
    ts = t + 1
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            i += 1
    return j == len(t)
```

**#65 — expected `variable_misuse`/`use_before_definition` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `use_before_definition` · line 2

```python
def add(a, b):
    ba = b + 1
    MAX = 0x7FFFFFFF
    MASK = 0xFFFFFFFF
    while b:
        a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK
    return a if a <= MAX else ~(a ^ MASK)
```

## variable_name_typo  (5/5 coarse-correct)

**#66 — expected `variable_misuse`/`variable_name_typo` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `variable_name_typo` · line 2

```python
def removeElement(nums, val):
    ii = 0
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += 1
    return i
```

**#67 — expected `variable_misuse`/`variable_name_typo` · gold line(s) [3]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `variable_name_typo` · line 3

```python
def minimum_xor_sum(nums1, nums2):
    nums2.sort()
    xor_sumxor_sum = 0
    for i in range(len(nums1)):
        xor_sum += nums1[i] ^ nums2[i]
    return xor_sum
```

**#68 — expected `variable_misuse`/`variable_name_typo` · gold line(s) [2]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `variable_name_typo` · line 2

```python
def truncate_sentence(s, k):
    wordswords = s.split(' ')
    return ' '.join(words[:k])
```

**#69 — expected `variable_misuse`/`variable_name_typo`**  
System: PASS `variable_misuse` (conf 1.000) · subtype `variable_name_typo` · line 2

```python
def split_and_minimize(num):
    aa = num // 1000
    ba = (num % 1000) // 100
    cb = (num % 100) // 10
    dd = num % 10
    return ((a * 10 + c) * 10 + b) * 10 + d
```

**#70 — expected `variable_misuse`/`variable_name_typo` · gold line(s) [2, 3]**  
System: PASS `variable_misuse` (conf 1.000) · subtype `variable_name_typo` · line 3

```python
def convert_temperature(celsius):
    kelvinkelvin = celsius + 273.15
    fahrenheitkelvin = celsius * 1.8 + 32
    return [kelvin, fahrenheit]
```

## no_bug  (2/6 coarse-correct)

**#71 — expected `no_bug`**  
System: PASS `no_bug` (conf 0.979)

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def construct_tree(descriptions):
    nodes = {}

    for d in descriptions:
        nodes[d[0]] = TreeNode(d[0])
        nodes[d[1]] = TreeNode(d[1])

    for d in descriptions:
        if d[2]:
            nodes[d[0]].left = nodes[d[1]]
        else:
            nodes[d[0]].right = nodes[d[1]]

    return nodes[descriptions[0][0]]
```

**#72 — expected `no_bug`**  
System: MISS `logic_error` (conf 0.998) · subtype `wrong_comparison_target`

```python
def compute_avg(numbers):
    total = 0
    for num in numbers:
        total += num
    return total/len(numbers)
```

**#73 — expected `no_bug`**  
System: MISS `logic_error` (conf 0.999) · subtype `swapped_comparison_operands` · line 7

```python
def max_invites(favorite):
    n = len(favorite)
    dp = [0] * n
    max_invites = 0

    for i in range(n):
        dp[i] = 2 if i == favorite[favorite[i]] else 1
        max_invites = max(max_invites, dp[i])

    return max_invites
```

**#74 — expected `no_bug`**  
System: MISS `logic_error` (conf 0.999) · subtype `wrong_comparison_target` · line 7

```python
def hireWorkers(costs, k, candidates):
    n = len(costs)
    workers = sorted([(cost, i) for i, cost in enumerate(costs)])

    cost = 0
    for i in range(k):
        if min(workers[i][1], n - workers[i][1] - 1) < candidates:
            cost += workers[i][0]

    return cost
```

**#75 — expected `no_bug`**  
System: MISS `logic_error` (conf 0.960) · subtype `wrong_comparison_target`

```python
def print_sum(a, b):
    """Print the sum of two numbers."""
    print(a + b)
```

**#76 — expected `no_bug`**  
System: PASS `no_bug` (conf 0.591)

```python
def rand7():
    pass  # Provided API

def rand10():
    result = None
    while True:
        result = (rand7() - 1) * 7 + rand7()
        if result <= 40:
            break
    return result % 10 + 1
```
