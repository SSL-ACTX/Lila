def _raise_python_exception(exc_id: int):
    if exc_id == 1:
        raise ValueError("Lirien JIT: ValueError")
    elif exc_id == 2:
        raise TypeError("Lirien JIT: TypeError")
    elif exc_id == 3:
        raise IndexError("Lirien JIT: IndexError")
    elif exc_id == 4:
        raise RuntimeError("Lirien JIT: RuntimeError")
    elif exc_id == 5:
        raise ZeroDivisionError("Lirien JIT: ZeroDivisionError")
    else:
        raise RuntimeError(f"Lirien JIT: Unknown exception ID {exc_id}")
