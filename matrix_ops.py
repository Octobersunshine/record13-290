import numpy as np
from typing import List, Union


class MatrixOperations:
    @staticmethod
    def _to_ndarray(matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        if isinstance(matrix, np.ndarray):
            return matrix
        if not isinstance(matrix, list):
            raise TypeError("矩阵必须是列表或 NumPy 数组")
        try:
            arr = np.array(matrix, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"矩阵数据格式无效: {e}")
        if arr.ndim != 2:
            raise ValueError(f"必须是二维矩阵，当前维度为 {arr.ndim}")
        return arr

    @staticmethod
    def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape:
            raise ValueError(
                f"矩阵维度不匹配: {a.shape} vs {b.shape}，"
                f"相同维度的矩阵才能进行加减运算"
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

    @staticmethod
    def get_shape(matrix: Union[List[List[float]], np.ndarray]) -> tuple:
        if isinstance(matrix, np.ndarray):
            arr = matrix
        else:
            arr = np.array(matrix, dtype=float)
        return tuple(arr.shape)
