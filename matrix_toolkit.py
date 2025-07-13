import numpy as np

def create_matrix(rows, cols, fill_value=0):
    """
    Create a matrix of size (rows x cols) filled with a specific value.
    """
    return np.full((rows, cols), fill_value)

def add_matrices(matrix_a, matrix_b):
    """
    Add two matrices element-wise.
    """
    return np.add(matrix_a, matrix_b)

def multiply_matrices(matrix_a, matrix_b):
    """
    Multiply two matrices (dot product).
    """
    return np.dot(matrix_a, matrix_b)

def transpose_matrix(matrix):
    """
    Transpose a given matrix.
    """
    return np.transpose(matrix)

def determinant(matrix):
    """
    Calculate the determinant of a square matrix.
    """
    return np.linalg.det(matrix)

def inverse(matrix):
    """
    Find the inverse of a square matrix.
    """
    return np.linalg.inv(matrix)

def eigen(matrix):
    """
    Find eigenvalues and eigenvectors of a square matrix.
    """
    values, vectors = np.linalg.eig(matrix)
    return values, vectors

# Test the functions
if __name__ == "__main__":
    print("=== MATRIX TOOLKIT ===")

    # Create two 2x2 matrices
    a = create_matrix(2, 2, 3)
    b = create_matrix(2, 2, 5)

    print("\nMatrix A:\n", a)
    print("\nMatrix B:\n", b)

    # Addition
    print("\nAdded Matrices:\n", add_matrices(a, b))

    # Transpose
    print("\nTransposed Matrix A:\n", transpose_matrix(a))

    # Multiplication
    print("\nMatrix Multiplication (A x B):\n", multiply_matrices(a, b))

    # Determinant
    det_a = determinant(a)
    print("\nDeterminant of Matrix A:", det_a)

    # Inverse (check for singularity)
    try:
        inv_a = inverse(a)
        print("\nInverse of Matrix A:\n", inv_a)
    except np.linalg.LinAlgError:
        print("\nMatrix A is singular and cannot be inverted.")

    # Eigenvalues and Eigenvectors
    values, vectors = eigen(a)
    print("\nEigenvalues of Matrix A:\n", values)
    print("\nEigenvectors of Matrix A:\n", vectors)
