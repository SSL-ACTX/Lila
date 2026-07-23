import unittest
from lirien import verify, i64, Refined, Box, SizedArray
from lirien.clif import clif, v0, v1, v2, v3

Positive = Refined[i64, lambda x: x > 0]


@verify
def clif_kernel(a: i64, b: Positive) -> i64:
    offset = 42
    with clif(inputs={a: v0, b: v1, offset: v2}, outputs={v3: "res"}):  # noqa: F823
        v4 = v0 + v2
        v3 = v4 // v1
    return res


@verify
def clif_store(ptr: Box[i64], val: i64) -> None:
    with clif(inputs={ptr: v0, val: v1}):
        v0[0] = val


@verify
def clif_load(ptr: Box[i64]) -> i64:
    with clif(inputs={ptr: v0}, outputs={v1: "res"}):  # noqa: F823
        v1 = v0[0]
    return res


@verify
def clif_store_offset(ptr: Box[i64], val: i64) -> None:
    with clif(inputs={ptr: v0, val: v1}):
        v0[8] = val


@verify
def clif_load_offset(ptr: Box[i64]) -> i64:
    with clif(inputs={ptr: v0}, outputs={v1: "res"}):  # noqa: F823
        v1 = v0[8]
    return res


@verify
def clif_dot_product_4d(a: Box[i64], b: Box[i64]) -> i64:
    with clif(inputs={a: v0, b: v1}, outputs={v2: "res"}):  # noqa: F823
        # Load elements from vector a
        v3 = v0[0]
        v4 = v0[8]
        v5 = v0[16]
        v6 = v0[24]

        # Load elements from vector b
        v7 = v1[0]
        v8 = v1[8]
        v9 = v1[16]
        v10 = v1[24]

        # Perform element-wise multiplication
        v11 = v3 * v7
        v12 = v4 * v8
        v13 = v5 * v9
        v14 = v6 * v10

        # Sum the products
        v15 = v11 + v12
        v16 = v13 + v14
        v2 = v15 + v16
    return res


@verify
def clif_string_syntax(a: i64, b: i64) -> i64:
    with clif(inputs={"v0": a, "v1": b}, outputs={"v2": "res"}):
        v2 = v0 * v1
    return res


class TestClifBlocks(unittest.TestCase):
    def test_clif_basic(self):
        res = clif_kernel(10, 2)
        # (10 + 42) // 2 = 52 // 2 = 26
        self.assertEqual(res, 26)

    def test_clif_memory(self):
        buf = Box(99)
        clif_store(buf, 123)
        res = clif_load(buf)
        self.assertEqual(res, 123)

    def test_clif_memory_offset(self):
        arr = SizedArray[i64, 2]([0, 0])
        buf = Box(arr)

        clif_store_offset(buf, 456)
        self.assertEqual(arr[1], 456)

        res = clif_load_offset(buf)
        self.assertEqual(res, 456)

    def test_clif_dot_product(self):
        arr_a = SizedArray[i64, 4]([1, 2, 3, 4])
        arr_b = SizedArray[i64, 4]([5, 6, 7, 8])
        a = Box(arr_a)
        b = Box(arr_b)

        res = clif_dot_product_4d(a, b)
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        self.assertEqual(res, 70)

    def test_clif_string_syntax(self):
        res = clif_string_syntax(7, 8)
        self.assertEqual(res, 56)


if __name__ == "__main__":
    unittest.main()
