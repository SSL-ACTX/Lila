import unittest
import subprocess
import sys
from lirien import jit, i64


class TestJitAssert(unittest.TestCase):
    def test_jit_assert_success(self):
        @jit
        def assert_positive(x: i64) -> i64:
            assert x > 0
            return x

        self.assertEqual(assert_positive(5), 5)

    def test_jit_assert_trap(self):
        # Since a Cranelift trap aborts the process, we run the failing case
        # in a subprocess and assert that it exits with a non-zero/error code.
        code = """
import sys
from lirien import jit, i64

@jit
def assert_positive(x: i64) -> i64:
    assert x > 0
    return x

assert_positive(-5)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertNotEqual(res.returncode, 0)

    def test_jit_assert_middle_success(self):
        @jit
        def assert_middle(x: i64) -> i64:
            y = x + 1
            assert y > 0
            return y

        self.assertEqual(assert_middle(5), 6)

    def test_jit_assert_middle_trap(self):
        code = """
import sys
from lirien import jit, i64

@jit
def assert_middle(x: i64) -> i64:
    y = x + 1
    assert y > 0
    return y

assert_middle(-5)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertNotEqual(res.returncode, 0)

    def test_jit_assert_body_success(self):
        @jit
        def assert_body(x: i64) -> i64:
            y = x + 1
            assert y > 0
            z = y * 2
            return z

        self.assertEqual(assert_body(5), 12)

    def test_jit_assert_body_trap(self):
        code = """
import sys
from lirien import jit, i64

@jit
def assert_body(x: i64) -> i64:
    y = x + 1
    assert y > 0
    z = y * 2
    return z

assert_body(-5)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertNotEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
