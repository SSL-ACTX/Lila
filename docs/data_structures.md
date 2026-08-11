# Data Structures & Memory Layouts

Lirien maps high-level Python types to flat, contiguous, C-compatible memory layouts, maximizing cache locality and eliminating object header overhead.

---

## C-Compatible Structs (`@struct`)

Classes decorated with `@struct` compile to flat memory. Nested fields are inlined directly at absolute byte offsets inside the parent allocation rather than referencing pointer chains.

Lirien automatically derives `__repr__` and `__eq__` behaviors for all `@struct` definitions at load time, ensuring they interact naturally with standard Python code.

```python
from lirien import struct, f64, i32, Refined


@struct
class Point:
    x: f64
    y: f64


@struct
class Trace:
    p: Point  # Inlined directly starting at offset 0 (16 bytes)
    id: i32  # Placed at offset 16


# Refinement predicates can traverse nested struct fields
SafeTrace = Refined[Trace, lambda t: t.p.x > 0]
```

---

## Stack-Allocated Value Types (`@value`)

Classes decorated with `@value` act as pass-by-value aggregates. They are allocated directly on the stack of the execution thread. When placed in containers like `Buffer[T]`, they are stored contiguously in memory with zero indirection.

```python
from lirien import value, i64, Buffer, verify


@value
class Point3D:
    x: i64
    y: i64
    z: i64


@verify
def process_points(data: Buffer[Point3D]) -> None:
    # Contiguous memory iteration, no pointer dereferencing needed
    for i in range(len(data)):
        val = data[i].x
```

---

## Tensors & Kernel Fusion

Lirien models `Tensor[T, *Shape]` at the type level. It uses `TypeVarTuple` and `Unpack` to support rank-polymorphism.

```python
from typing import TypeVarTuple, Unpack
from lirien import verify, Tensor, f32, i64

Shape = TypeVarTuple("Shape")


@verify
def get_rank(a: Tensor[f32, Unpack[Shape]]) -> i64:
    return len(Shape)
```

### Kernel Fusion
When you chain tensor operations (e.g. `a * b + d`), Lirien's optimizer fuses them into a single loop execution pass at the IR level. The resulting machine code computes the value element-by-element without allocating any intermediate temporary tensors on the heap.

---

## Algebraic Data Types (`@adt`)

An `@adt` represents a tagged union of variants. Lirien compiles dispatch match blocks to native `switch` jump tables executing in $O(1)$ time. Z3 verifies that pattern matches are exhaustive and that variant fields are only accessed if their corresponding tag is active.

```python
from lirien import verify, i64, adt, Box


@adt
class Node:
    Cons: (i64, Box["Node"])
    Nil: None


@verify
def sum_list(n: Node) -> i64:
    match n:
        case Node.Cons(val, next):
            return val + sum_list(next)
        case Node.Nil:
            return 0
```

---

## Native String Type (`str`)

Lirien provides a native UTF-8 `str` type. A string is represented as a pointer to a struct containing the character buffer pointer and its 64-bit length.
All operations (concatenation, slicing, indexing) are supported, and Z3 proves index bounds checking statically.

```python
from lirien import verify, i64


@verify
def get_char(s: str, idx: i64) -> str:
    # Z3 proves the access is safe if the guards are met
    if idx >= 0 and idx < len(s):
        return s[idx]
    return ""
```

---

## Tuples & NamedTuples: Register Flattening

Standard tuples and `NamedTuple` classes are recursively flattened into primitive variables during compilation. Small aggregates (2 registers or less, $\le 16$ bytes) are passed directly inside CPU registers. Larger ones utilize return-by-pointer (SRet) layouts but are still flattened in argument lists.

```python
from typing import NamedTuple
from lirien import verify, i64


class Point(NamedTuple):
    x: i64
    y: i64


@verify
def scale_tuple(data: tuple[Point, i64]) -> Point:
    [p, factor] = data
    return Point(p.x * factor, p.y * factor)


# Lowered ABI representation:
# Parameters: x: i64, y: i64, factor: i64
# Returns:    new_x: i64, new_y: i64
```

---

## `TypedDict`: Zero-Cost Dictionaries

Lirien compiles `TypedDict` variables down to the same flat memory layout as `@struct`. Key lookups (`config["timeout"]`) are resolved to constant byte offsets at compile time. At runtime, the dictionary and its string keys are completely compiled away—leaving only single instruction memory offsets.

```python
from typing import TypedDict
from lirien import verify, i64


class Config(TypedDict):
    timeout: i64
    enabled: bool


@verify
def check(cfg: Config) -> i64:
    # Compiles to a direct load at (base_ptr + 8 bytes)
    if cfg["enabled"]:
        return cfg["timeout"]
    return 0
```

---

## Growable Lists (`List[T]`)

`List[T]` provides a heap-allocated, growable array. Z3 models the list size and validates every index operation dynamically against the list's current capacity to ensure memory safety.

```python
from lirien import verify, List, i64


@verify
def build_and_index() -> i64:
    l = List[i64]()
    l.append(42)
    return l[0]  # Z3 verifies that index 0 is valid because size is 1
```

---

## Buffers & Zero-Copy Slicing

`Buffer[T]` wraps any Python object implementing the buffer protocol (e.g. NumPy arrays, `memoryview`). 

*   **Slicing**: Creating slices (including strided, negative step, and reverse slicing) returns zero-copy views. 
*   **Safety**: Z3 computes the slice limits and verifies that the slicing bounds are within the parent buffer capacity.

```python
from lirien import verify, SizedArray, i64


@verify
def strided_slice(arr: SizedArray[i64, 10]) -> i64:
    # Zero-copy view containing indices 0, 2, 4, 6, 8
    view = arr[0:10:2]
    return view[0] + view[1]
```
