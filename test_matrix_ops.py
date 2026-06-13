import unittest
import numpy as np
from matrix_ops import (
    MatrixOperations,
    MatrixDimensionError,
    MatrixValidationError,
)


class TestMatrixOperations(unittest.TestCase):
    def test_add_2x2_lists(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[6, 8], [10, 12]])

    def test_add_3x3_lists(self):
        a = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[2, 2, 3], [4, 6, 6], [7, 8, 10]])

    def test_add_with_numpy_arrays(self):
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1, 1], [1, 1]], dtype=float)
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[2, 3], [4, 5]])

    def test_add_with_negative_numbers(self):
        a = [[-1, -2], [3, -4]]
        b = [[1, 2], [-3, 4]]
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[0, 0], [0, 0]])

    def test_add_with_floats(self):
        a = [[1.5, 2.5], [3.5, 4.5]]
        b = [[0.5, 0.5], [0.5, 0.5]]
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[2.0, 3.0], [4.0, 5.0]])

    def test_subtract_2x2_lists(self):
        a = [[5, 6], [7, 8]]
        b = [[1, 2], [3, 4]]
        result = MatrixOperations.subtract(a, b)
        self.assertEqual(result, [[4, 4], [4, 4]])

    def test_subtract_result_negative(self):
        a = [[1, 2], [3, 4]]
        b = [[10, 20], [30, 40]]
        result = MatrixOperations.subtract(a, b)
        self.assertEqual(result, [[-9, -18], [-27, -36]])

    def test_subtract_with_numpy_arrays(self):
        a = np.array([[10, 20], [30, 40]], dtype=float)
        b = np.array([[1, 2], [3, 4]], dtype=float)
        result = MatrixOperations.subtract(a, b)
        self.assertEqual(result, [[9, 18], [27, 36]])

    def test_add_shape_mismatch_raises(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[1, 2], [3, 4]]
        with self.assertRaises(MatrixDimensionError) as ctx:
            MatrixOperations.add(a, b)
        self.assertIn("维度不匹配", str(ctx.exception))
        self.assertEqual(ctx.exception.shape_a, (2, 3))
        self.assertEqual(ctx.exception.shape_b, (2, 2))

    def test_subtract_shape_mismatch_raises(self):
        a = [[1, 2], [3, 4], [5, 6]]
        b = [[1, 2], [3, 4]]
        with self.assertRaises(MatrixDimensionError) as ctx:
            MatrixOperations.subtract(a, b)
        self.assertIn("维度不匹配", str(ctx.exception))
        self.assertEqual(ctx.exception.shape_a, (3, 2))
        self.assertEqual(ctx.exception.shape_b, (2, 2))

    def test_non_matrix_list_raises(self):
        with self.assertRaises(MatrixValidationError) as ctx:
            MatrixOperations.add([1, 2, 3], [4, 5, 6])
        self.assertIn("二维矩阵", str(ctx.exception))

    def test_invalid_type_raises(self):
        with self.assertRaises(MatrixValidationError):
            MatrixOperations.add("not a matrix", [[1, 2]])

    def test_irregular_matrix_raises(self):
        with self.assertRaises(MatrixValidationError):
            MatrixOperations.add([[1, 2, 3], [4, 5]], [[1, 2], [3, 4]])

    def test_empty_matrix_raises(self):
        with self.assertRaises(MatrixValidationError):
            MatrixOperations.add([[]], [[1]])

    def test_shape_mismatch_error_code_values(self):
        err = MatrixDimensionError(
            "test", shape_a=(3, 3), shape_b=(2, 2)
        )
        self.assertEqual(err.shape_a, (3, 3))
        self.assertEqual(err.shape_b, (2, 2))

    def test_get_shape(self):
        self.assertEqual(MatrixOperations.get_shape([[1, 2], [3, 4]]), (2, 2))
        self.assertEqual(MatrixOperations.get_shape([[1, 2, 3], [4, 5, 6]]), (2, 3))
        self.assertEqual(
            MatrixOperations.get_shape(np.array([[1], [2], [3]], dtype=float)),
            (3, 1),
        )

    def test_add_1xN_matrix(self):
        a = [[1, 2, 3, 4]]
        b = [[5, 6, 7, 8]]
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[6, 8, 10, 12]])

    def test_add_Nx1_matrix(self):
        a = [[1], [2], [3], [4]]
        b = [[10], [20], [30], [40]]
        result = MatrixOperations.add(a, b)
        self.assertEqual(result, [[11], [22], [33], [44]])

    def test_multiply_2x2_x_2x2(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[19, 22], [43, 50]])

    def test_multiply_2x3_x_3x2(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[58, 64], [139, 154]])

    def test_multiply_3x3_x_3x1(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        b = [[1], [2], [3]]
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[14], [32], [50]])

    def test_multiply_1x3_x_3x1(self):
        a = [[1, 2, 3]]
        b = [[4], [5], [6]]
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[32]])

    def test_multiply_with_numpy_arrays(self):
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1, 0], [0, 1]], dtype=float)
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_multiply_identity_matrix(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = MatrixOperations.multiply(a, identity)
        self.assertEqual(result, a)
        result2 = MatrixOperations.multiply(identity, a)
        self.assertEqual(result2, a)

    def test_multiply_zero_matrix(self):
        a = [[1, 2, 3], [4, 5, 6]]
        zero = [[0, 0], [0, 0], [0, 0]]
        result = MatrixOperations.multiply(a, zero)
        self.assertEqual(result, [[0, 0], [0, 0]])

    def test_multiply_with_negative_numbers(self):
        a = [[-1, -2], [3, -4]]
        b = [[2, 0], [0, -3]]
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[-2, 6], [6, 12]])

    def test_multiply_with_floats(self):
        a = [[1.5, 2.5], [3.5, 4.5]]
        b = [[2.0, 0], [0, 2.0]]
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[3.0, 5.0], [7.0, 9.0]])

    def test_multiply_shape_mismatch_raises(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(MatrixDimensionError) as ctx:
            MatrixOperations.multiply(a, b)
        self.assertIn("乘法维度不匹配", str(ctx.exception))
        self.assertEqual(ctx.exception.shape_a, (2, 3))
        self.assertEqual(ctx.exception.shape_b, (2, 3))

    def test_multiply_shape_mismatch_columns_vs_rows(self):
        a = [[1, 2], [3, 4], [5, 6]]
        b = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(MatrixDimensionError) as ctx:
            MatrixOperations.multiply(a, b)
        self.assertIn("列数", str(ctx.exception))
        self.assertIn("行数", str(ctx.exception))
        self.assertEqual(ctx.exception.shape_a, (3, 2))
        self.assertEqual(ctx.exception.shape_b, (3, 2))

    def test_multiply_valid_but_different_from_add_shape(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[1, 2], [3, 4], [5, 6]]
        add_raises = False
        try:
            MatrixOperations.add(a, b)
        except MatrixDimensionError:
            add_raises = True
        self.assertTrue(add_raises)
        result = MatrixOperations.multiply(a, b)
        self.assertEqual(result, [[22, 28], [49, 64]])


if __name__ == "__main__":
    unittest.main()
