# Performance & Monomorphization

Lirien achieves native C-like execution speed by bypassing the CPython VM interpreter, eliminating runtime object boxing, and resolving polymorphism statically.

---

## GIL-Free Parallelism

Because Lirien functions execute on native primitive types and flat byte buffers rather than CPython PyObject structures, they do not need to hold the Global Interpreter Lock (GIL). You can distribute work across all available OS threads using the `parallel_for` construct.

```python
from lirien import verify, parallel_for, Buffer, f64, i64

@verify
def parallel_scale(vec: Buffer[f64], factor: f64) -> None:
    def body(i: i64):
        vec[i] *= factor
        
    # Spawns OS threads without acquiring the Python GIL
    parallel_for(range(len(vec)), body)
```

---

## Static Dispatch via `typing.Protocol`

Polymorphism and structural interfaces are declared using `typing.Protocol`. 

Instead of generating virtual method tables (vtables) or relying on slow dynamic method resolution (duck-typing), Lirien uses **monomorphization**:
1. The compiler maps implementations statically.
2. For each unique concrete struct or ADT type passed to a protocol argument, Lirien creates and JIT-compiles a specialized copy of the function.
3. Method calls inside the monomorphized function are lowered to direct jumps.

```python
from typing import Protocol
from lirien import verify, f32, struct, adt

class Renderable(Protocol):
    def render(self) -> f32: ...

@struct
class Circle:
    radius: f32
    def render(self) -> f32:
        return self.radius * 3.14

@adt
class Shape:
    Rect: f32
    Dot: f32
    def render(self) -> f32:
        match self:
            case Shape.Rect(w): return w * w
            case Shape.Dot(_):  return 0.0

@verify
def draw(obj: Renderable) -> f32:
    # Monomorphized to a direct call (e.g. `Circle_render` or `Shape_render`)
    return obj.render()
```

---

## Null-Pointer Optimization & Smart Casts

In Python, optionals are usually wrapped objects. In Lirien:
* `Optional[Box[T]]` (or `Box[T] | None`) is represented as a raw 64-bit pointer.
* `None` is compiled to the null pointer (`0x0`). There is zero memory or allocation overhead.
* Z3 enforces that the pointer is non-null before any dereference (`.val` or field access).

To make checks ergonomic, the compiler performs **flow-sensitive smart casts**:
Checking `x is not None` inserts a type narrowing cast in the SSA graph. Inside the guarded branch, the variable's type is narrowed automatically from `Box[T] | None` to `Box[T]`.

```python
@struct
class Node:
    val: i64
    next: Optional[Box["Node"]]

@verify
def sum_list(n: Optional[Box[Node]]) -> i64:
    if n is None: 
        return 0
    # From here on, `n` is statically narrowed to `Box[Node]`.
    # Accessing .val is guaranteed safe by Z3.
    return n.val + sum_list(n.next)
```

---

## Const Generics & Type-Level Arithmetic

`TypeVar` is used to capture types and constant integers (e.g., buffer dimensions) statically. Lirien evaluates symbolic type-level expressions (such as `N + 1`) during monomorphization.

```python
from typing import TypeVar
from lirien import verify, i64, f64, SizedArray

T = TypeVar("T", i64, f64)
N = TypeVar("N")  # Const generic integer

@verify
def pad_one(x: SizedArray[T, N], out: SizedArray[T, N + 1]) -> i64:
    for i in range(N):
        out[i] = x[i]
    return N + 1
```

---

## Multiple Dispatch via `@overload`

Ad-hoc polymorphism can be declared via Python's standard `typing.overload` decorator. The JIT JIT-compiles distinct machine code implementations for each signature and dispatches calls statically at the call site.

```python
from typing import overload
from lirien import verify, i64, f64

@overload
def compute(x: i64) -> i64: ...

@overload
def compute(x: f64) -> f64: ...

@verify
def compute(x):
    return x * 2
```

---

## Loop Unrolling via `typing.Literal`

When a function parameter is annotated as `typing.Literal[N]`, the compiler treats that value as a compile-time constant. This allows the JIT to unroll loops bounded by `N` into linear instruction sequences, enabling Z3 to perform precise loop-invariant analysis without induction approximation.

```python
from typing import Literal
from lirien import verify, i64

@verify
def unrolled_sum(limit: Literal[5]) -> i64:
    total = 0
    # Fully unrolled to exactly 5 linear additions
    for i in range(limit):
        total += i
    return total
```

---

## SIMD Vectorization

Lirien supports vector-register types directly. Operations on these types translate directly to native CPU vector instructions (e.g., SSE, AVX, NEON). Scalar literals are automatically splatted to fill all vector lanes.

```python
from lirien import verify, i8x16

@verify
def process_pixels(a: i8x16, b: i8x16) -> i8x16:
    # Compiles to vector SIMD instructions (e.g. vpadd + vpsub on x86)
    return (a + b) - 10
```

Available SIMD types:
*   **128-bit float vectors:** `f32x4`, `f64x2`
*   **128-bit integer vectors:** `i8x16`, `u8x16`, `i16x8`, `u16x8`, `i32x4`, `i64x2`
