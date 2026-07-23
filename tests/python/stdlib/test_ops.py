import unittest
from lirien import Tensor, f32, num


class TestStdlibOps(unittest.TestCase):
    def test_transpose(self):
        a = Tensor.alloc((2, 3), f32)
        out = Tensor.alloc((3, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[0, 2] = 3.0
        a[1, 0] = 4.0
        a[1, 1] = 5.0
        a[1, 2] = 6.0

        num.transpose(a, out)

        self.assertEqual(out[0, 0], 1.0)
        self.assertEqual(out[0, 1], 4.0)
        self.assertEqual(out[1, 0], 2.0)
        self.assertEqual(out[1, 1], 5.0)
        self.assertEqual(out[2, 0], 3.0)
        self.assertEqual(out[2, 1], 6.0)

    def test_matmul(self):
        a = Tensor.alloc((2, 3), f32)
        b = Tensor.alloc((3, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[0, 2] = 3.0
        a[1, 0] = 4.0
        a[1, 1] = 5.0
        a[1, 2] = 6.0

        b[0, 0] = 7.0
        b[0, 1] = 8.0
        b[1, 0] = 9.0
        b[1, 1] = 10.0
        b[2, 0] = 11.0
        b[2, 1] = 12.0

        num.matmul(a, b, out)

        self.assertEqual(out[0, 0], 58.0)
        self.assertEqual(out[0, 1], 64.0)
        self.assertEqual(out[1, 0], 139.0)
        self.assertEqual(out[1, 1], 154.0)

    def test_add(self):
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        b[0, 0] = 5.0
        b[0, 1] = 6.0
        b[1, 0] = 7.0
        b[1, 1] = 8.0

        num.add(a, b, out)

        self.assertEqual(out[0, 0], 6.0)
        self.assertEqual(out[0, 1], 8.0)
        self.assertEqual(out[1, 0], 10.0)
        self.assertEqual(out[1, 1], 12.0)

    def test_sub(self):
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 5.0
        a[0, 1] = 6.0
        a[1, 0] = 7.0
        a[1, 1] = 8.0

        b[0, 0] = 1.0
        b[0, 1] = 2.0
        b[1, 0] = 3.0
        b[1, 1] = 4.0

        num.sub(a, b, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 4.0)
        self.assertEqual(out[1, 0], 4.0)
        self.assertEqual(out[1, 1], 4.0)

    def test_mul(self):
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        b[0, 0] = 5.0
        b[0, 1] = 6.0
        b[1, 0] = 7.0
        b[1, 1] = 8.0

        num.mul(a, b, out)

        self.assertEqual(out[0, 0], 5.0)
        self.assertEqual(out[0, 1], 12.0)
        self.assertEqual(out[1, 0], 21.0)
        self.assertEqual(out[1, 1], 32.0)

    def test_scale(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        num.scale(a, out, 2.5)

        self.assertEqual(out[0, 0], 2.5)
        self.assertEqual(out[0, 1], 5.0)
        self.assertEqual(out[1, 0], 7.5)
        self.assertEqual(out[1, 1], 10.0)

    def test_bias_add(self):
        a = Tensor.alloc((2, 3), f32)
        bias = Tensor.alloc((3,), f32)
        out = Tensor.alloc((2, 3), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[0, 2] = 3.0
        a[1, 0] = 4.0
        a[1, 1] = 5.0
        a[1, 2] = 6.0

        bias[0] = 0.5
        bias[1] = 1.0
        bias[2] = 1.5

        num.bias_add(a, bias, out)

        self.assertEqual(out[0, 0], 1.5)
        self.assertEqual(out[0, 1], 3.0)
        self.assertEqual(out[0, 2], 4.5)
        self.assertEqual(out[1, 0], 4.5)
        self.assertEqual(out[1, 1], 6.0)
        self.assertEqual(out[1, 2], 7.5)

    def test_outer(self):
        a = Tensor.alloc((3,), f32)
        b = Tensor.alloc((2,), f32)
        out = Tensor.alloc((3, 2), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        b[0] = 4.0
        b[1] = 5.0

        num.outer(a, b, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 5.0)
        self.assertEqual(out[1, 0], 8.0)
        self.assertEqual(out[1, 1], 10.0)
        self.assertEqual(out[2, 0], 12.0)
        self.assertEqual(out[2, 1], 15.0)

    def test_dot(self):
        a = Tensor.alloc((3,), f32)
        b = Tensor.alloc((3,), f32)
        out = Tensor.alloc((1,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        b[0] = 4.0
        b[1] = 5.0
        b[2] = 6.0

        num.dot(a, b, out)

        self.assertEqual(out[0], 32.0)

    def test_bmm(self):
        a = Tensor.alloc((2, 2, 3), f32)
        b = Tensor.alloc((2, 3, 2), f32)
        out = Tensor.alloc((2, 2, 2), f32)

        # Batch 0
        a[0, 0, 0] = 1.0
        a[0, 0, 1] = 2.0
        a[0, 0, 2] = 3.0
        a[0, 1, 0] = 4.0
        a[0, 1, 1] = 5.0
        a[0, 1, 2] = 6.0

        b[0, 0, 0] = 7.0
        b[0, 0, 1] = 8.0
        b[0, 1, 0] = 9.0
        b[0, 1, 1] = 10.0
        b[0, 2, 0] = 11.0
        b[0, 2, 1] = 12.0

        # Batch 1
        a[1, 0, 0] = 0.5
        a[1, 0, 1] = 1.5
        a[1, 0, 2] = -1.0
        a[1, 1, 0] = 2.0
        a[1, 1, 1] = 0.0
        a[1, 1, 2] = 1.0

        b[1, 0, 0] = 1.0
        b[1, 0, 1] = 0.0
        b[1, 1, 0] = 2.0
        b[1, 1, 1] = 3.0
        b[1, 2, 0] = 0.0
        b[1, 2, 1] = 4.0

        num.bmm(a, b, out)

        self.assertAlmostEqual(out[0, 0, 0], 58.0, places=5)
        self.assertAlmostEqual(out[0, 0, 1], 64.0, places=5)
        self.assertAlmostEqual(out[0, 1, 0], 139.0, places=5)
        self.assertAlmostEqual(out[0, 1, 1], 154.0, places=5)

        self.assertAlmostEqual(out[1, 0, 0], 3.5, places=5)
        self.assertAlmostEqual(out[1, 0, 1], 0.5, places=5)
        self.assertAlmostEqual(out[1, 1, 0], 2.0, places=5)
        self.assertAlmostEqual(out[1, 1, 1], 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
