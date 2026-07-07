# Core Concepts & Verification Contracts

Lirien relies on refinement types and path-sensitive solver queries to guarantee the absence of runtime errors. This page explains Lirien's logic and contract model.

---

## Refinement Types (Liquid Types)

A refinement type (or liquid type) is a base type paired with a logical predicate that constrains its value space. 

$$T_{\text{refined}} = \{ v: T \mid P(v) \}$$

Z3 checks that every assignment, function argument, and return value satisfies its declared predicate along all reachable control-flow paths. Lirien also supports postcondition inference using `...`, where it statically calculates the tightest bounds.

Lirien accepts both `Refined[T, pred]` and Python's standard `Annotated[T, pred]` (PEP 593) interchangeably.

```python
from typing import Annotated
from lirien import verify, i64, Refined

# These declarations are equivalent:
Positive = Refined[i64, lambda x: x > 0]
Positive = Annotated[i64, lambda x: x > 0]

@verify
def divide_verified(n: i64, d: Positive) -> i64:
    # Z3 proves 'd > 0' holds. ZeroDivisionError is statically impossible.
    return n // d

@verify
def clamp(x: i64) -> Refined[i64, ...]:
    # Lirien infers the postcondition: (and (>= {v} 1) (<= {v} 10))
    if x > 10: return 10
    if x < 1: return 1
    return x
```

---

## Symbolic Refinement DSL (`V`)

Writing raw `lambda` functions for every refinement type can become verbose. Lirien provides a symbolic placeholder object, `V`, to write point-free predicate expressions. You can chain comparisons and boolean operators to form complex predicates.

```python
from lirien import verify, i64, Refined, V

# Equivalent to lambda x: x > 0
Positive = Refined[i64, V > 0]

# Complex predicate: in [1, 100] and odd
BoundedOdd = Refined[i64, (V >= 1) & (V <= 100) & (V % 2 != 0)]

@verify
def next_odd(n: BoundedOdd) -> Refined[i64, V > 0]:
    return n + 2
```

Supported operators on `V`:
*   **Arithmetic:** `+`, `-`, `*`, `/`, `%`
*   **Comparison:** `>`, `>=`, `<`, `<=`, `==`, `!=`
*   **Logical:** `&` (AND), `|` (OR), `~` (NOT)

---

## Design by Contract (DbC)

Lirien promotes standard Python `assert` statements to static theorem proofs. 

*   **Static Verification:** The compiler validates preconditions at call sites, loop invariants on block entries and back-edges, and postconditions at function boundaries. 
*   **Zero Execution Overhead:** Once proven by Z3, the assertions are completely pruned from the compiled machine code.
*   **Runtime Fallbacks:** If a verified function is invoked from standard Python code (outside `@verify`), runtime type checks automatically assert the contracts at boundaries.

```python
from lirien import verify, i64

@verify
def add_one(x: i64) -> i64:
    # 1. Precondition
    assert 0 < x < 100, "x must be between 0 and 100"
    
    res = x + 1
    
    # 2. Postcondition
    assert res > x, "result must be greater than input"
    return res

@verify
def sum_to_n(n: i64) -> i64:
    assert 0 <= n < 100, "n must be non-negative"
    total = 0
    i = 0
    while i < n:
        # 3. Loop invariant (verified inductively)
        assert i >= 0, "loop index cannot be negative"
        assert total >= 0, "running total cannot be negative"
        total = total + i
        i = i + 1
    return total
```

---

## Inductive Proofs & Recursion

For recursive functions, Lirien verifies the inductive step by assuming the function holds for recursive calls under the declared contract, and then proves that both the base case and recursive step satisfy the return type postcondition.

```python
from lirien import verify, i64, Refined

SmallPos = Refined[i64, lambda x: (0 <= x) & (x <= 20)]
StrictPositive = Refined[i64, lambda x: x >= 1]

@verify
def factorial(n: SmallPos) -> StrictPositive:
    if n <= 1:
        return 1
    # Z3 proves: n * factorial(n - 1) >= 1
    return n * factorial(n - 1)
```

---

## Higher-Order Functions

Lirien supports closures and lambdas with compile-time capture analysis. The captured environments are automatically heap-allocated and tracked using refinement-aware function signatures.

```python
from lirien import verify, i64, Closure

@verify
def make_adder(x: i64) -> Closure[[i64], i64]:
    # x is captured by the closure environment and managed at the FFI boundary
    return lambda y: x + y
```
