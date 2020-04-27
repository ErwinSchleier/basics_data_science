from typing import List
import math
from typing import Tuple
from typing import Callable

"""
Vectors
"""
Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """ adds two vectors together
    :param v: vector v
    :param type: Vector
    :param w: vector w
    :param type: Vector
    :return: sum of vector v and w
    """
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]
assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """ subtracts vector w from vector v
    :param v: vector v
    :param type: Vector
    :param w: vector w
    :param type: Vector
    :return: difference of vector v and w
    """
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

def vector_sum(vectors: List[Vector]) -> Vector:
    """sums up all input vectors
    :param vectors: list of all vectors
    :param type: list
    :returns: summed up vector
    """
    assert vectors, "no vectors provided!"
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "differenz sizes!"
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """ builds multiplicity of a vector
    :param c: multiplicity
    :param type: float
    :param v: vector to multiply
    :param type: Vector
    :return: multiplicity of v by c
    """
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """ Calculates the mean of a collection of vectors
    :param vectors: list of vectors
    :param type: list
    :returns: averge over all vectors
    """
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """ calculates v_1 * w_1 + ... + v_n * w_n
    :param v: vector v
    :param type: Vector
    :param w: vector w
    :param type: Vector
    :return: dot product of v and w
    """
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
assert dot([1, 2, 3], [4, 5, 6]) == 32 # 1 * 4 + 2 * 5 + 3 * 6

def sum_of_squares(v: Vector) -> float:
    """ squares each component of a vector and sums them together
    :param v: input vector
    :param type: Vector
    :returns: sum of squares of vector v
    """
    return dot(v, v)
assert sum_of_squares([1, 2, 3]) == 14 # 1 * 1 + 2 * 2 + 3 * 3

def magnitude(v: Vector) -> float:
    """ provides the length of a given vector
    :param v: input vector
    :param type: Vector
    :returns: length of v
    """
    return math.sqrt(sum_of_squares(v))
assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    """ (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2
    :param v: vector v
    :param type: Vector
    :param w: vector w
    :param type: Vector
    :return: squared distance between v and w
    """
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """ calculates the distance between two vectors
    :param v: vector v
    :param type: Vector
    :param w: vector w
    :param type: Vector
    :return: distance between v and w
    """
    return math.sqrt(squared_distance(v, w))


"""
Matricies

A = [[1, 2, 3], # Matrix with 2 rows and 3 colums
    [4, 5, 6]]

B = [[1, 2], # Matrix with 3 rows and 2 colums
    [3, 4],
    [5, 6]]
"""
Matrix = List[List[float]]

def shape(A: Matrix) -> Tuple[int, int]:
    """ provides number of rows and number of colums
    :param A: input matrix
    :param type: Matrix
    :returns: number of rows and columns of A
    """
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols
assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)

def get_row(A: Matrix, i: int) -> Vector:
    """ provides i-th row of A as Vector
    :param A: input matrix
    :param type: Matrix
    :param i: target row
    :param type: int
    :returns: i-th row of A
    """
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """ provides j-th column of A as Vector
    :param A: input matrix
    :param type: Matrix
    :param j: target column
    :param type: int
    :returns: i-th column of A
    """
    return[A_i[j]
            for A_i in A]

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """ creates a matrix with num_rows and num_cols and entry_fn as entries at i, j
    :param num_rows: number of rows
    :param type: int
    :param num_cols: number of columns
    :param type: int
    :param entry_fn: entries for i, j
    :param type: object
    :returns: calculated matrix
    """
    return [[entry_fn(i, j)
            for j in range(num_cols)]
            for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    """creates  n x n matrix with 1 at the diagonale
    :param n: shape of matrix
    :param type: int
    :returns: n x n identity matrix
    """
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)
assert identity_matrix(5) == [[1, 0, 0, 0, 0], 
                                [0, 1, 0, 0, 0], 
                                [0, 0, 1, 0, 0], 
                                [0, 0, 0, 1, 0], 
                                [0, 0, 0, 0, 1]]