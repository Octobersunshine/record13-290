import numpy as np
from typing import List, Union, Tuple, Optional


class MatrixDimensionError(ValueError):
    def __init__(
        self,
        message: str,
        shape_a: Optional[Tuple[int, int]] = None,
        shape_b: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(message)
        self.shape_a = shape_a
        self.shape_b = shape_b


class MatrixValidationError(ValueError):
    pass


class MatrixOperations:
    @staticmethod
    def _to_ndarray(matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        if isinstance(matrix, np.ndarray):
            return matrix
        if not isinstance(matrix, list):
            raise MatrixValidationError("矩阵必须是列表或 NumPy 数组")
        try:
            arr = np.array(matrix, dtype=float)
        except (ValueError, TypeError) as e:
            raise MatrixValidationError(f"矩阵数据格式无效: {e}")
        if arr.ndim != 2:
            raise MatrixValidationError(f"必须是二维矩阵，当前维度为 {arr.ndim}")
        if arr.size == 0:
            raise MatrixValidationError("矩阵不能为空")
        return arr

    @staticmethod
    def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape:
            shape_a = tuple(a.shape)
            shape_b = tuple(b.shape)
            raise MatrixDimensionError(
                f"矩阵维度不匹配: {shape_a} vs {shape_b}，"
                f"相同维度的矩阵才能进行加减运算",
                shape_a=shape_a,
                shape_b=shape_b,
            )

    @staticmethod
    def _validate_multiply_shape(a: np.ndarray, b: np.ndarray) -> None:
        if a.shape[1] != b.shape[0]:
            shape_a = tuple(a.shape)
            shape_b = tuple(b.shape)
            raise MatrixDimensionError(
                f"矩阵乘法维度不匹配: {shape_a} vs {shape_b}，"
                f"矩阵 A 的列数 ({a.shape[1]}) 必须等于矩阵 B 的行数 ({b.shape[0]})",
                shape_a=shape_a,
                shape_b=shape_b,
            )

    @classmethod
    def add(
        cls,
        matrix_a: Union[List[List[float]], np.ndarray],
        matrix_b: Union[List[List[float]], np.ndarray],
    ) -> List[List[float]]:
        a = cls._to_ndarray(matrix_a)
        b = cls._to_ndarray(matrix_b)
        cls._validate_same_shape(a, b)
        result = a + b
        return result.tolist()

    @classmethod
    def subtract(
        cls,
        matrix_a: Union[List[List[float]], np.ndarray],
        matrix_b: Union[List[List[float]], np.ndarray],
    ) -> List[List[float]]:
        a = cls._to_ndarray(matrix_a)
        b = cls._to_ndarray(matrix_b)
        cls._validate_same_shape(a, b)
        result = a - b
        return result.tolist()

    @classmethod
    def multiply(
        cls,
        matrix_a: Union[List[List[float]], np.ndarray],
        matrix_b: Union[List[List[float]], np.ndarray],
    ) -> List[List[float]]:
        a = cls._to_ndarray(matrix_a)
        b = cls._to_ndarray(matrix_b)
        cls._validate_multiply_shape(a, b)
        result = a @ b
        return result.tolist()

    @staticmethod
    def get_shape(matrix: Union[List[List[float]], np.ndarray]) -> tuple:
        if isinstance(matrix, np.ndarray):
            arr = matrix
        else:
            arr = np.array(matrix, dtype=float)
        return tuple(arr.shape)
