import unittest
import math
from lirien import Tensor, f32, num


class TestStdlibNN(unittest.TestCase):
    def test_convolve1d(self):
        signal = Tensor.alloc((5,), f32)
        kernel = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        for i in range(5):
            signal[i] = float(i + 1)
        for i in range(3):
            kernel[i] = 1.0

        num.convolve1d(signal, kernel, out)

        self.assertEqual(out[0], 6.0)
        self.assertEqual(out[1], 9.0)
        self.assertEqual(out[2], 12.0)

    def test_convolve2d(self):
        image = Tensor.alloc((3, 3), f32)
        kernel = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        for i in range(3):
            for j in range(3):
                image[i, j] = float(i * 3 + j + 1)
        for i in range(2):
            for j in range(2):
                kernel[i, j] = 1.0

        num.convolve2d(image, kernel, out)

        self.assertEqual(out[0, 0], 12.0)
        self.assertEqual(out[0, 1], 16.0)
        self.assertEqual(out[1, 0], 24.0)
        self.assertEqual(out[1, 1], 28.0)

    def test_max_pool2d_2x2(self):
        image = Tensor.alloc((4, 4), f32)
        out = Tensor.alloc((2, 2), f32)

        val_list = [
            1.0,
            2.0,
            5.0,
            6.0,
            3.0,
            4.0,
            7.0,
            8.0,
            9.0,
            10.0,
            13.0,
            14.0,
            11.0,
            12.0,
            15.0,
            16.0,
        ]
        for i in range(4):
            for j in range(4):
                image[i, j] = val_list[i * 4 + j]

        num.max_pool2d_2x2(image, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 8.0)
        self.assertEqual(out[1, 0], 12.0)
        self.assertEqual(out[1, 1], 16.0)

    def test_avg_pool2d_2x2(self):
        image = Tensor.alloc((4, 4), f32)
        out = Tensor.alloc((2, 2), f32)

        val_list = [
            1.0,
            2.0,
            5.0,
            6.0,
            3.0,
            4.0,
            7.0,
            8.0,
            9.0,
            10.0,
            13.0,
            14.0,
            11.0,
            12.0,
            15.0,
            16.0,
        ]
        for i in range(4):
            for j in range(4):
                image[i, j] = val_list[i * 4 + j]

        num.avg_pool2d_2x2(image, out)

        self.assertAlmostEqual(out[0, 0], 2.5)
        self.assertAlmostEqual(out[0, 1], 6.5)
        self.assertAlmostEqual(out[1, 0], 10.5)
        self.assertAlmostEqual(out[1, 1], 14.5)

    def test_mean(self):
        a = Tensor.alloc((4,), f32)
        out = Tensor.alloc((1,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        a[3] = 4.0

        num.mean(a, out, 4.0)

        self.assertAlmostEqual(out[0], 2.5)

    def test_standardize(self):
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0

        num.standardize(a, out, 2.0, 1.0)

        self.assertAlmostEqual(out[0], -1.0)
        self.assertAlmostEqual(out[1], 0.0)
        self.assertAlmostEqual(out[2], 1.0)

    def test_matvec(self):
        matrix = Tensor.alloc((2, 3), f32)
        vector = Tensor.alloc((3,), f32)
        out = Tensor.alloc((2,), f32)

        matrix[0, 0] = 1.0
        matrix[0, 1] = 2.0
        matrix[0, 2] = 3.0
        matrix[1, 0] = 4.0
        matrix[1, 1] = 5.0
        matrix[1, 2] = 6.0

        vector[0] = 2.0
        vector[1] = 1.0
        vector[2] = 3.0

        num.matvec(matrix, vector, out)

        self.assertEqual(out[0], 13.0)
        self.assertEqual(out[1], 31.0)

    def test_l2_normalize(self):
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 3.0
        a[1] = 4.0
        a[2] = 0.0

        num.l2_normalize(a, out, 1e-9)

        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[1], 0.8)
        self.assertAlmostEqual(out[2], 0.0)

    def test_l1_normalize(self):
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = -2.0
        a[2] = 1.0

        num.l1_normalize(a, out, 1e-9)

        self.assertAlmostEqual(out[0], 0.25)
        self.assertAlmostEqual(out[1], -0.5)
        self.assertAlmostEqual(out[2], 0.25)

    def test_cosine_similarity(self):
        a = Tensor.alloc((3,), f32)
        b = Tensor.alloc((3,), f32)
        out = Tensor.alloc((1,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        b[0] = 2.0
        b[1] = 4.0
        b[2] = 6.0

        num.cosine_similarity(a, b, out, 1e-9)

        self.assertAlmostEqual(out[0], 1.0, places=5)

    def test_rms_norm(self):
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0

        num.rms_norm(a, out, 1e-9, 3.0)

        rms = math.sqrt(14.0 / 3.0)
        self.assertAlmostEqual(out[0], 1.0 / rms, places=5)
        self.assertAlmostEqual(out[1], 2.0 / rms, places=5)
        self.assertAlmostEqual(out[2], 3.0 / rms, places=5)

    def test_layer_norm(self):
        a = Tensor.alloc((3,), f32)
        gamma = Tensor.alloc((3,), f32)
        beta = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        gamma[0] = 1.0
        gamma[1] = 1.0
        gamma[2] = 1.0
        beta[0] = 0.0
        beta[1] = 0.0
        beta[2] = 0.0

        num.layer_norm(a, out, gamma, beta, 1e-9, 3.0)

        std_val = math.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(out[0], -1.0 / std_val, places=5)
        self.assertAlmostEqual(out[1], 0.0, places=5)
        self.assertAlmostEqual(out[2], 1.0 / std_val, places=5)

    def test_matvec_bias(self):
        matrix = Tensor.alloc((2, 3), f32)
        vector = Tensor.alloc((3,), f32)
        bias = Tensor.alloc((2,), f32)
        out = Tensor.alloc((2,), f32)

        matrix[0, 0] = 1.0
        matrix[0, 1] = 2.0
        matrix[0, 2] = 3.0
        matrix[1, 0] = 4.0
        matrix[1, 1] = 5.0
        matrix[1, 2] = 6.0

        vector[0] = 2.0
        vector[1] = 1.0
        vector[2] = 3.0
        bias[0] = 0.5
        bias[1] = -1.5

        num.matvec_bias(matrix, vector, bias, out)

        self.assertEqual(out[0], 13.5)
        self.assertEqual(out[1], 29.5)

    def test_max_pool2d_generic(self):
        image = Tensor.alloc((3, 3), f32)
        out = Tensor.alloc((2, 2), f32)

        image[0, 0] = 1.0
        image[0, 1] = 3.0
        image[0, 2] = 2.0
        image[1, 0] = 4.0
        image[1, 1] = 2.0
        image[1, 2] = 5.0
        image[2, 0] = 0.0
        image[2, 1] = 1.0
        image[2, 2] = 3.0

        num.max_pool2d(image, out, 2, 2, 1, 1)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 5.0)
        self.assertEqual(out[1, 0], 4.0)
        self.assertEqual(out[1, 1], 5.0)

    def test_avg_pool2d_generic(self):
        image = Tensor.alloc((3, 3), f32)
        out = Tensor.alloc((2, 2), f32)

        image[0, 0] = 1.0
        image[0, 1] = 3.0
        image[0, 2] = 2.0
        image[1, 0] = 4.0
        image[1, 1] = 2.0
        image[1, 2] = 5.0
        image[2, 0] = 0.0
        image[2, 1] = 1.0
        image[2, 2] = 3.0

        num.avg_pool2d(image, out, 2, 2, 1, 1)

        self.assertAlmostEqual(out[0, 0], 2.5, places=5)
        self.assertAlmostEqual(out[0, 1], 3.0, places=5)
        self.assertAlmostEqual(out[1, 0], 1.75, places=5)
        self.assertAlmostEqual(out[1, 1], 2.75, places=5)

    def test_convolve2d_padded(self):
        image = Tensor.alloc((2, 2), f32)
        kernel = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        image[0, 0] = 1.0
        image[0, 1] = 2.0
        image[1, 0] = 3.0
        image[1, 1] = 4.0

        kernel[0, 0] = 1.0
        kernel[0, 1] = 1.0
        kernel[1, 0] = 1.0
        kernel[1, 1] = 1.0

        num.convolve2d_padded(image, kernel, out, 1, 1, 1, 1)

        self.assertAlmostEqual(out[0, 0], 1.0, places=5)

    def test_resize_nearest(self):
        image = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((3, 3), f32)

        image[0, 0] = 10.0
        image[0, 1] = 20.0
        image[1, 0] = 30.0
        image[1, 1] = 40.0

        num.resize_nearest(image, out, 0.5, 0.5)

        self.assertEqual(out[0, 0], 10.0)
        self.assertEqual(out[0, 2], 20.0)
        self.assertEqual(out[2, 0], 30.0)
        self.assertEqual(out[2, 2], 40.0)


if __name__ == "__main__":
    unittest.main()
