import numpy as np


def integrate_in_triangle(func) -> float:
    ans = 0.

    array = np.array([
        [0.5,0.5],[0.5,0],[0,0.5]
    ])

    for i in range(3):
        x, y = array[i]
        ans += func(x, y)

    ans = ans / 3
    return ans


def integrate_in_tetrahedron(func) -> float:
    ans = 0.

    a = 0.5854102
    b = 0.1381966
    array = np.array([
        [a, b, b], [b, a, b], [b, b, a], [b, b, b]
    ])

    for i in range(4):
        x, y, z = array[i]
        ans += func(x, y, z)

    ans = ans / 24
    return ans


def c3d10_edge_matrix(coordinate_matrix: np.ndarray, h: float) -> np.ndarray:
    edge = np.zeros((6, 6))


    for row in range(6):
        for col in range(6):
            def func(x: float, y: float) -> float:
                ans = c2d6_shape_function(row, x, y) * c2d6_shape_function(col, x, y) * \
                jacobi_tri(coordinate_matrix, x, y)

                return ans
            
            edge[row, col] = integrate_in_triangle(func)

    return edge * h


def shape_function_derivative(l1: float, l2: float, l3: float) -> np.ndarray:
    '''
    return: 3*10 matrix
    '''
    matrix = np.array([
        [4*l1-1,0,0,-4*(1-l1-l2-l3)+1,4*l2,0,4*l3,4*(1-2*l1-l2-l3),-4*l2,-4*l3],
        [0,4*l2-1,0,-4*(1-l1-l2-l3)+1,4*l1,4*l3,0,-4*l1,4*(1-l1-2*l2-l3),-4*l3],
        [0,0,4*l3-1,-4*(1-l1-l2-l3)+1,0,4*l2,4*l1,-4*l1,-4*l2,4*(1-l1-l2-2*l3)]
    ])
    return matrix



def c3d10_shape_function(index: int, x: float, y: float, z: float) -> float:
    """
    index: from 0 to 9
    """
    l4 = 1-x-y-z
    vector = np.array([
        (2*x-1)*x, (2*y-1)*y, (2*z-1)*z, (2*l4-1)*l4, 4*x*y, 
        4*y*z, 4*x*z, 4*x*l4, 4*y*l4, 4*z*l4 
    ])
    ans = vector[index]
    return ans


def c3d10_volume(coordinate_matrix: np.ndarray) -> float:
    """
    :param coordinate_matrix: row:10, col:3
    :return: volume of c3d10
    """
    def jacobi(x: float, y: float, z: float) -> float:
        det = np.linalg.det(shape_function_derivative(x,y,z) @ coordinate_matrix)
        return abs(det)

    return integrate_in_tetrahedron(jacobi)


def c3d10_mass_matrix(coordinate_matrix: np.ndarray, p:float, c:float) -> np.ndarray:
    m = np.zeros((10, 10))

    for row in range(10):
        for col in range(row+1):
            def func(x: float, y: float, z: float) -> float:
                det = np.linalg.det(shape_function_derivative(x, y, z) @ coordinate_matrix)
                shape_function1 = c3d10_shape_function(row, x, y, z)
                shape_function2 = c3d10_shape_function(col, x, y, z)

                return abs(det) * shape_function1 * shape_function2
            
            ans = integrate_in_tetrahedron(func)
            m[row, col] += ans
            m[col, row] += ans

    for i in range(10):
        m[i, i] *= 0.5

    return m * p * c


def c3d10_stiffness_matrix(coordinate_matrix: np.ndarray, conductivity: list) -> np.ndarray:
    k = np.zeros((10,10))

    for row in range(10):
        for col in range(row + 1):
            def func(x: float, y: float, z: float) -> float:
                jacobi = shape_function_derivative(x, y, z) @ coordinate_matrix
                det = abs(np.linalg.det(jacobi))
                inv_jacobi = np.linalg.inv(jacobi)
                D = inv_jacobi @ shape_function_derivative(x, y, z)

                res = 0.
                for i in range(3):
                    res += det * D[i, row] * D[i, col] * conductivity[i]

                return res

            ans = integrate_in_tetrahedron(func)
            k[row, col] += ans
            k[col, row] += ans

    for idx in range(10):
        k[idx,idx] *= 0.5

    return k


def c3d10_fQ(coordinate_matrix: np.ndarray, Q: float) -> np.ndarray:
    f = np.zeros(10)

    for i in range(10):
        def func(x: float, y: float, z: float) -> float:
            jacobi = shape_function_derivative(x, y, z) @ coordinate_matrix
            det = abs(np.linalg.det(jacobi))
            shape_function = c3d10_shape_function(i, x, y, z)

            return shape_function * det

        f[i] = integrate_in_tetrahedron(func)

    return f * Q


def c2d6_shape_function(index: int, x: float, y: float) -> float:
    """
    index: from 0 to 5
    """
    z = 1-x-y
    vector = np.array([
        (2*x-1)*x, (2*y-1)*y, (2*z-1)*z, 
        4*y*z, 4*x*z, 4*x*y
    ])
    ans = vector[index]
    return ans


def jacobi_tri(coordinate_matrix: np.ndarray, x: float, y: float) -> float:
    derivative = np.array([
        [4*x-1,0,-4*(1-x-y)+1,-4*y,4*(1-2*x-y),4*y],
        [0,4*y-1,-4*(1-x-y)+1,4*(1-x-2*y),-4*x,4*x]
    ])

    o = np.dot(derivative, coordinate_matrix) # (2,3)

    a = np.array([[o[0, 0], o[0, 1]],
                  [o[1, 0], o[1, 1]]])
    b = np.array([[o[0, 1], o[0, 2]],
                  [o[1, 1], o[1, 2]]])
    c = np.array([[o[0, 2], o[0, 0]],
                  [o[1, 2], o[1, 0]]])
    A = np.linalg.det(a)
    B = np.linalg.det(b)
    C = np.linalg.det(c)

    ans = (A ** 2 + B ** 2 + C ** 2) ** 0.5

    return ans


def c3d10_f2(coordinate_matrix: np.ndarray, q: float) -> np.ndarray:
    f = np.zeros(6)

    for i in range(6):
        def func(x: float, y: float) -> float:
            ans = c2d6_shape_function(i, x, y) * jacobi_tri(coordinate_matrix, x, y)
            return ans
        
        f[i] = integrate_in_triangle(func)

    return f * q


def c3d10_f3(coordinate_matrix: np.ndarray, h: float, temperature: float) -> np.ndarray:
    return c3d10_f2(coordinate_matrix, h * temperature)


if __name__ == '__main__':
    x = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
    [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ])
    
    f = c3d10_fQ(x,1)
    print(f)
    print(np.sum(f))
    print(c3d10_mass_matrix(x,1,1))
    