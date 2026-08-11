# Getting Started

This guide walks you through setting up Lirien, building it from source, running tests, and creating your first verified JIT-compiled Python function.

---

## Prerequisites

To compile and run Lirien, your system must have:

*   **Rust Toolchain** (Stable, version `1.80` or later)
*   **Python** (version `3.10` or later, along with `python3-dev` / headers)
*   **Z3 Theorem Prover** (Shared library, version `4.12` or later)
*   **maturin** (Python tool to build/install Rust extension modules)

### Installing Prerequisites

#### Debian/Ubuntu
```bash
sudo apt-get update
sudo apt-get install -y clang libz3-dev python3-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### macOS (via Homebrew)
```bash
brew install rust z3
```

---

## Build and Installation

Lirien is built as a hybrid Python/Rust package. The Rust package acts as a Python extension module built via PyO3 and Maturin.

### 1. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip maturin ruff
```

### 2. Build and Install the JIT Module
Run `maturin develop` inside the repository root to compile the Rust core and link it dynamically to your virtual environment:
```bash
maturin develop --release
```

---

## Running the Test Suites

To verify your installation, compile and run the full test suite.

### Rust Unit Tests
```bash
cargo test
```

### Python Integration Tests
```bash
PYTHONPATH=./python python3 -m unittest discover tests/python
```

---

## Your First Program

Save the following code as `example.py` and run it:

```python
from lirien import verify, i64, Refined, V

# Define a refinement type: an integer strictly greater than zero
Positive = Refined[i64, V > 0]


@verify
def divide(n: i64, d: Positive) -> i64:
    # Z3 formally proves at compile-time that 'd > 0' holds.
    # A ZeroDivisionError is mathematically impossible in native code.
    return n // d


if __name__ == "__main__":
    # Successful execution (compiled to native JIT code bypasses interpreter)
    print("100 / 5 =", divide(100, 5))

    try:
        # This will fail compilation at import/load time because 0 is not positive!
        print("100 / 0 =", divide(100, 0))
    except Exception as e:
        print("Verification error successfully caught:", e)
```
