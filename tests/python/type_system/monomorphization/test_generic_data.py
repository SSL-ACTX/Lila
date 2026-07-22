import unittest
import sys
from lirien import verify, struct, adt, i64, f64
from typing import TypeVar, Generic

T = TypeVar("T")


@struct
class BoxedVal(Generic[T]):
    value: T


@adt
class Opt(Generic[T]):
    Some: T
    None_: None


@verify
def get_boxed_i64(b: BoxedVal[i64]) -> i64:
    return b.value


@verify
def get_boxed_f64(b: BoxedVal[f64]) -> f64:
    return b.value


@verify
def unwrap_opt_i64(o: Opt[i64]) -> i64:
    match o:
        case Opt_i64.Some(val):
            return val
        case Opt_i64.None_:
            return -1


@verify
def unwrap_opt_f64(o: Opt[f64]) -> f64:
    match o:
        case Opt_f64.Some(val):
            return val
        case Opt_f64.None_:
            return -1.0


# 1. Single Generic Function using standard TypeVar
@verify
def get_boxed(b: BoxedVal[T]) -> T:
    return b.value


@verify
def unwrap_opt(o: Opt[T], default_val: T) -> T:
    match o:
        case Opt_T.Some(val):
            return val
        case Opt_T.None_:
            return default_val


class TestGenericData(unittest.TestCase):
    def test_struct_specialization(self):
        b1 = BoxedVal[i64](10)
        b2 = BoxedVal[f64](20.5)

        self.assertEqual(get_boxed_i64(b1), 10)
        self.assertEqual(get_boxed_f64(b2), 20.5)

    def test_adt_specialization(self):
        o1 = Opt[i64].Some(42)
        o2 = Opt[f64].Some(3.14)
        o3 = Opt[i64].None_()

        self.assertEqual(unwrap_opt_i64(o1), 42)
        self.assertAlmostEqual(unwrap_opt_f64(o2), 3.14, places=2)
        self.assertEqual(unwrap_opt_i64(o3), -1)

    def test_auto_monomorphization_struct(self):
        b1 = BoxedVal[i64](42)
        b2 = BoxedVal[f64](3.14)

        # Standard TypeVar
        self.assertEqual(get_boxed(b1), 42)
        self.assertAlmostEqual(get_boxed(b2), 3.14, places=2)

    def test_auto_monomorphization_adt(self):
        o1 = Opt[i64].Some(100)
        o2 = Opt[i64].None_()
        o3 = Opt[f64].Some(2.718)
        o4 = Opt[f64].None_()

        # Standard TypeVar
        self.assertEqual(unwrap_opt(o1, -1), 100)
        self.assertEqual(unwrap_opt(o2, -1), -1)
        self.assertAlmostEqual(unwrap_opt(o3, -1.0), 2.718, places=3)
        self.assertAlmostEqual(unwrap_opt(o4, -1.0), -1.0, places=1)

    def test_auto_monomorphization_pep695(self):
        if sys.version_info >= (3, 12):
            import os

            proj_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            if proj_root not in sys.path:
                sys.path.insert(0, proj_root)
            from tests.python.type_system.monomorphization.pep695_test_cases import (
                get_boxed_pep695,
                unwrap_opt_pep695,
            )

            b1 = BoxedVal[i64](42)
            b2 = BoxedVal[f64](3.14)
            self.assertEqual(get_boxed_pep695(b1), 42)
            self.assertAlmostEqual(get_boxed_pep695(b2), 3.14, places=2)

            o1 = Opt[i64].Some(100)
            o2 = Opt[i64].None_()
            o3 = Opt[f64].Some(2.718)
            o4 = Opt[f64].None_()
            self.assertEqual(unwrap_opt_pep695(o1, -1), 100)
            self.assertEqual(unwrap_opt_pep695(o2, -1), -1)
            self.assertAlmostEqual(unwrap_opt_pep695(o3, -1.0), 2.718, places=3)
            self.assertAlmostEqual(unwrap_opt_pep695(o4, -1.0), -1.0, places=1)


if __name__ == "__main__":
    unittest.main()
