import unittest
import math
from lirien import Tensor, f32, num, f32x4


class TestStdlibSIMD(unittest.TestCase):
    def test_dot_simd(self):
        a = Tensor.alloc((2,), f32x4)
        b = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((1,), f32)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        b[0] = f32x4(2.0, 1.0, 0.5, 0.25)
        b[1] = f32x4(0.0, 1.0, 2.0, 3.0)

        num.dot_simd(a, b, out)

        self.assertAlmostEqual(out[0], 50.5, places=5)

    def test_matvec_simd(self):
        matrix = Tensor.alloc((2, 2), f32x4)
        vector = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((2,), f32)

        matrix[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        matrix[0, 1] = f32x4(5.0, 6.0, 7.0, 8.0)
        matrix[1, 0] = f32x4(0.0, 1.0, 2.0, 3.0)
        matrix[1, 1] = f32x4(1.0, 1.0, 1.0, 1.0)

        vector[0] = f32x4(2.0, 1.0, 0.5, 0.25)
        vector[1] = f32x4(0.0, 1.0, 2.0, 3.0)

        num.matvec_simd(matrix, vector, out)

        self.assertAlmostEqual(out[0], 50.5, places=5)
        self.assertAlmostEqual(out[1], 8.75, places=5)

    def test_mse_simd(self):
        a = Tensor.alloc((2,), f32x4)
        b = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((1,), f32)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        b[0] = f32x4(2.0, 1.0, 4.0, 2.0)
        b[1] = f32x4(4.0, 7.0, 5.0, 9.0)

        num.mse_simd(a, b, out)

        self.assertAlmostEqual(out[0], 14.0, places=5)

    def test_mae_simd(self):
        a = Tensor.alloc((2,), f32x4)
        b = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((1,), f32)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        b[0] = f32x4(2.0, 1.0, 4.0, 2.0)
        b[1] = f32x4(4.0, 7.0, 5.0, 9.0)

        num.mae_simd(a, b, out)

        self.assertAlmostEqual(out[0], 10.0, places=5)

    def test_add_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        b[0, 0] = f32x4(10.0, 20.0, 30.0, 40.0)

        num.add_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 11.0)
        self.assertEqual(res[1], 22.0)
        self.assertEqual(res[2], 33.0)
        self.assertEqual(res[3], 44.0)

    def test_sub_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(10.0, 20.0, 30.0, 40.0)
        b[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)

        num.sub_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 9.0)
        self.assertEqual(res[1], 18.0)
        self.assertEqual(res[2], 27.0)
        self.assertEqual(res[3], 36.0)

    def test_mul_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        b[0, 0] = f32x4(5.0, 6.0, 7.0, 8.0)

        num.mul_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 5.0)
        self.assertEqual(res[1], 12.0)
        self.assertEqual(res[2], 21.0)
        self.assertEqual(res[3], 32.0)

    def test_scale_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)

        num.scale_simd(a, out, 5.0)

        res = out[0, 0]
        self.assertEqual(res[0], 5.0)
        self.assertEqual(res[1], 10.0)
        self.assertEqual(res[2], 15.0)
        self.assertEqual(res[3], 20.0)

    def test_relu_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(-1.5, 0.0, 2.5, -0.5)

        num.relu_simd(a, out)

        res = out[0, 0]
        self.assertEqual(res[0], 0.0)
        self.assertEqual(res[1], 0.0)
        self.assertEqual(res[2], 2.5)
        self.assertEqual(res[3], 0.0)

    def test_div_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(10.0, 20.0, 30.0, 40.0)
        b[0, 0] = f32x4(2.0, 5.0, 10.0, 4.0)

        num.div_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 5.0)
        self.assertEqual(res[1], 4.0)
        self.assertEqual(res[2], 3.0)
        self.assertEqual(res[3], 10.0)

    def test_matmul_simd(self):
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[0, 1] = f32x4(5.0, 6.0, 7.0, 8.0)
        a[1, 0] = f32x4(0.0, 1.0, 2.0, 3.0)
        a[1, 1] = f32x4(1.0, 1.0, 1.0, 1.0)

        b[0, 0] = f32x4(2.0, 1.0, 0.5, 0.25)
        b[0, 1] = f32x4(0.0, 2.0, 4.0, 6.0)
        b[1, 0] = f32x4(0.0, 1.0, 2.0, 3.0)
        b[1, 1] = f32x4(1.0, 0.0, 1.0, 0.0)

        num.matmul_simd(a, b, out)

        self.assertAlmostEqual(out[0, 0], 50.5, places=5)
        self.assertAlmostEqual(out[0, 1], 52.0, places=5)
        self.assertAlmostEqual(out[1, 0], 8.75, places=5)
        self.assertAlmostEqual(out[1, 1], 30.0, places=5)

    def test_bmm_simd(self):
        a = Tensor.alloc((2, 2, 2), f32x4)
        b = Tensor.alloc((2, 2, 2), f32x4)
        out = Tensor.alloc((2, 2, 2), f32)

        # Batch 0
        a[0, 0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[0, 0, 1] = f32x4(5.0, 6.0, 7.0, 8.0)
        a[0, 1, 0] = f32x4(0.0, 1.0, 2.0, 3.0)
        a[0, 1, 1] = f32x4(1.0, 1.0, 1.0, 1.0)

        b[0, 0, 0] = f32x4(2.0, 1.0, 0.5, 0.25)
        b[0, 0, 1] = f32x4(0.0, 2.0, 4.0, 6.0)
        b[0, 1, 0] = f32x4(0.0, 1.0, 2.0, 3.0)
        b[0, 1, 1] = f32x4(1.0, 0.0, 1.0, 0.0)

        # Batch 1
        a[1, 0, 0] = f32x4(1.0, 1.0, 1.0, 1.0)
        a[1, 0, 1] = f32x4(2.0, 2.0, 2.0, 2.0)
        a[1, 1, 0] = f32x4(0.5, 0.5, 0.5, 0.5)
        a[1, 1, 1] = f32x4(0.25, 0.25, 0.25, 0.25)

        b[1, 0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        b[1, 0, 1] = f32x4(0.0, 1.0, 2.0, 3.0)
        b[1, 1, 0] = f32x4(1.0, 1.0, 1.0, 1.0)
        b[1, 1, 1] = f32x4(2.0, 2.0, 2.0, 2.0)

        num.bmm_simd(a, b, out)

        self.assertAlmostEqual(out[0, 0, 0], 50.5, places=5)
        self.assertAlmostEqual(out[0, 0, 1], 52.0, places=5)
        self.assertAlmostEqual(out[0, 1, 0], 8.75, places=5)
        self.assertAlmostEqual(out[0, 1, 1], 30.0, places=5)

        self.assertAlmostEqual(out[1, 0, 0], 18.0, places=5)
        self.assertAlmostEqual(out[1, 0, 1], 22.0, places=5)

    def test_rms_norm_simd(self):
        a = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((2,), f32x4)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        num.rms_norm_simd(a, out, 1e-5, 8.0)

        rms = math.sqrt(204.0 / 8.0 + 1e-5)
        inv_rms = 1.0 / rms

        res0 = out[0]
        res1 = out[1]
        self.assertAlmostEqual(res0[0], 1.0 * inv_rms, places=5)
        self.assertAlmostEqual(res1[3], 8.0 * inv_rms, places=5)

    def test_layer_norm_simd(self):
        a = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((2,), f32x4)
        gamma = Tensor.alloc((2,), f32x4)
        beta = Tensor.alloc((2,), f32x4)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        gamma[0] = f32x4(1.0, 1.0, 1.0, 1.0)
        gamma[1] = f32x4(1.0, 1.0, 1.0, 1.0)
        beta[0] = f32x4(0.0, 0.0, 0.0, 0.0)
        beta[1] = f32x4(0.0, 0.0, 0.0, 0.0)

        num.layer_norm_simd(a, out, gamma, beta, 1e-5, 8.0)

        mean = 4.5
        std = math.sqrt(5.25 + 1e-5)
        inv_std = 1.0 / std

        res0 = out[0]
        res1 = out[1]
        self.assertAlmostEqual(res0[0], (1.0 - mean) * inv_std, places=5)
        self.assertAlmostEqual(res1[3], (8.0 - mean) * inv_std, places=5)


if __name__ == "__main__":
    unittest.main()
