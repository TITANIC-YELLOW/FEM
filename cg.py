import numpy as np


def conjugate_gradient(A, b, x0, tol=1e-5, max_iter=1000):
    """
    Conjugate Gradient Method for solving linear system Ax = b.

    Parameters:
        A (numpy.ndarray): The symmetric positive definite matrix.
        b (numpy.ndarray): The right-hand side vector.
        x0 (numpy.ndarray): The initial guess.
        tol (float): Tolerance for stopping criteria (default 1e-5).
        max_iter (int): Maximum number of iterations (default 1000).

    Returns:
        numpy.ndarray: The solution vector.
    """
    x = x0
    r = b - np.dot(A, x)  # Residual
    p = r.copy()
    r_dot_old = np.dot(r, r)

    for k in range(max_iter):
        Ap = np.dot(A, p)
        alpha = r_dot_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_new = np.dot(r, r)
        if np.sqrt(r_dot_new) < tol:
            break
        beta = r_dot_new / r_dot_old
        p = r + beta * p
        r_dot_old = r_dot_new

    return x


# Example usage:
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
b = np.array([5, -7, 3])
x0 = np.zeros_like(b)
solution = conjugate_gradient(A, b, x0)
print("Solution:", solution)
