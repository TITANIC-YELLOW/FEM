import numpy as np

const1 = 1. / 3. ** 0.5

loaded_data = np.load('array.npz')
PyramidIntPoints1 = loaded_data['arr1']
PyramidIntPoints2 = loaded_data['arr2']
PyramidIntPoints3 = loaded_data['arr3']
PyramidIntPoints4 = loaded_data['arr4']
TetraIntPoints = loaded_data['arrtetra']
PrismIntPoints = loaded_data['arrprism']
PrismIntPoints27 = loaded_data['arrprism27']
ND_hexa = loaded_data['dN_hexa']


'''QuadIntPoints: Quadrilateral's gaussian integral points'''
QuadIntPoints = np.array([[-const1, -const1],
                          [-const1, const1],
                          [const1, -const1],
                          [const1, const1]])


'''HexaIntPoints: Hexahedron's gaussian integral points'''
HexaIntPoints = np.array([[-const1, -const1, -const1],
                          [-const1, -const1, const1],
                          [-const1, const1, -const1],
                          [-const1, const1, const1],
                          [const1, -const1, -const1],
                          [const1, -const1, const1],
                          [const1, const1, -const1],
                          [const1, const1, const1]])


dN_tetra = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]])


'''pyramidlist contains 256 gaussian integral points in standard pyramid.'''
pyramidlist = [PyramidIntPoints1, PyramidIntPoints2, PyramidIntPoints3, PyramidIntPoints4]



def IntInPyramid(func) -> float:
    '''
    IntInPyramid is a function used to integrate the function in standard pyramid.\n
    It gets a callable function,
    and returns the result of integration.
    '''
    ans = 0.

    for i in range(4):
        for j in range(27):
            x, y, z, const1, const2 = pyramidlist[i][j]
            ans += func(x,y,z) * const1 * const2

    return ans


def TetraShapeFunction(i:int,x:float,y:float,z:float) -> float:
    '''
    i: the number of the standard tetrahedron's shape function\n
    x, y, z: coordinate
    '''
    if i == 0:
        return x
    elif i == 1:
        return y
    elif i == 2:
        return z
    elif i == 3:
        return 1 - x - y - z


def V_tetra(CoordinateArray:np.ndarray) -> float:
    '''
    V_tetra returns the volume of a tetrahedron.\n
    CoordinateArray's shape: (4,3)
    '''
    ones = np.ones((4,1))

    V_matrix = np.concatenate((ones, CoordinateArray), axis=1)

    V = abs(np.linalg.det(V_matrix)) / 6

    return V


def m_tetra(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:
    '''
    m_tetra returns the mass matrix of a tetrahedron.\n
    CoordinateArray's shape: (4,3)\n
    p: density\n
    c: specific heat
    '''
    m = V_tetra(CoordinateArray) * p * c / 20 * np.array([[2, 1, 1, 1],
                                                          [1, 2, 1, 1],
                                                          [1, 1, 2, 1],
                                                          [1, 1, 1, 2]])
    return m



def k_tetra(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:
    '''
    k_tetra returns the stiffness matrix of a tetrahedron.\n
    CoordinateArray's shape: (4,3)\n
    ThermalConductivity: [k_x, k_y, k_z]
    '''
    x1, y1, z1 = CoordinateArray[0]
    x2, y2, z2 = CoordinateArray[1]
    x3, y3, z3 = CoordinateArray[2]
    x4, y4, z4 = CoordinateArray[3]

    b1 = (y2 - y4)*(z3 - z4) - (y3 - y4)*(z2 - z4)
    b2 = (y3 - y4)*(z1 - z4) - (y1 - y4)*(z3 - z4)
    b3 = (y1 - y4)*(z2 - z4) - (y2 - y4)*(z1 - z4)
    b4 = -(b1 + b2 + b3)
    c1 = (x3 - x4)*(z2 - z4) - (x2 - x4)*(z3 - z4)
    c2 = (x1 - x4)*(z3 - z4) - (x3 - x4)*(z1 - z4)
    c3 = (x2 - x4)*(z1 - z4) - (x1 - x4)*(z2 - z4)
    c4 = -(c1 + c2 + c3)
    d1 = (x2 - x4)*(y3 - y4) - (x3 - x4)*(y2 - y4)
    d2 = (x3 - x4)*(y1 - y4) - (x1 - x4)*(y3 - y4)
    d3 = (x1 - x4)*(y2 - y4) - (x2 - x4)*(y1 - y4)
    d4 = -(d1 + d2 + d3)


    k1 = ThermalConductivity[0] * np.array([[b1*b1, b1*b2, b1*b3, b1*b4],
                                            [b1*b2, b2*b2, b2*b3, b2*b4],
                                            [b1*b3, b2*b3, b3*b3, b3*b4],
                                            [b1*b4, b2*b4, b3*b4, b4*b4]])
    k2 = ThermalConductivity[1] * np.array([[c1*c1, c1*c2, c1*c3, c1*c4],
                                            [c1*c2, c2*c2, c2*c3, c2*c4],
                                            [c1*c3, c2*c3, c3*c3, c3*c4],
                                            [c1*c4, c2*c4, c3*c4, c4*c4]])
    k3 = ThermalConductivity[2] * np.array([[d1*d1, d1*d2, d1*d3, d1*d4],
                                            [d1*d2, d2*d2, d2*d3, d2*d4],
                                            [d1*d3, d2*d3, d3*d3, d3*d4],
                                            [d1*d4, d2*d4, d3*d4, d4*d4]])

    k = (k1 + k2 + k3) / (36 * V_tetra(CoordinateArray))

    return k


def Jacobi_tri(coor:np.ndarray) -> float:
    '''
    Jacobi_tri returns Jacobian determinant for coordinate transformation
    of a triangular element.\n
    coor's shape: (3,3)
    '''
    dN = np.array([[-1, 1, 0],
                   [-1, 0, 1]])

    o = np.dot(dN, coor)

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


def edge_tri(CoordinateArray:np.ndarray, h:float) -> np.ndarray:
    '''
    edge_tri is used to calculate the stiffness matrix
    affected by the third type of boundary conditions.\n
    CoordinateArray's shape: (3,3)
    '''
    edge = np.array([[2,1,1],
                     [1,2,1],
                     [1,1,2]])

    J = Jacobi_tri(CoordinateArray)

    edge = edge * h * J / 24

    return edge


def S_tri(coor:np.ndarray) -> float:
    '''
    S_tri returns the area of a triangle in 3D space.\n
    coor's shape: (3,3)
    '''
    x1, y1, z1 = coor[0]
    x2, y2, z2 = coor[1]
    x3, y3, z3 = coor[2]

    a = ((x2 - x1)**  2 + (y2 - y1)**  2 + (z2 - z1)**  2) ** 0.5
    b = ((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2) ** 0.5
    c = ((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2) ** 0.5

    s = (a + b + c) * 0.5

    A = (s * (s - a) * (s - b) * (s - c)) ** 0.5

    return A


def f3_tri(CoordinateArray:np.ndarray, h:float, Tf:float) -> np.ndarray:
    '''
    f3_tri returns a triangular element's load vector
    related to the third type of boundary condition.\n
    CoordinateArray's shape: (3,3)
    '''
    return np.ones(3) * h * Tf * S_tri(CoordinateArray)/ 3


def f2_tri(CoordinateArray:np.ndarray, q:float) -> np.ndarray:
    '''
    f2_tri returns a triangular element's load vector
    related to the second type of boundary condition.\n
    CoordinateArray's shape: (3,3)
    '''
    return np.ones(3) * q * S_tri(CoordinateArray)/ 3


def fQ_tetra(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:
    '''
    fQ_tetra returns a tetrahedral element's load vector
    related to the internal heat source.\n
    CoordinateArray's shape: (4,3)
    '''
    coefficient = c * Q * V_tetra(CoordinateArray) * 0.25
    fQ = np.ones(4) * coefficient

    return fQ


def dN_prism(x:float, y:float, z:float) -> np.ndarray:
    '''
    dN_prism returns the derivatives of the shape functions of a standard prism
    in the natural coordinates x, y, and z directions.
    '''
    dN = np.zeros((3, 6))

    t1 = 0.5 * (z - 1)
    t2 = -0.5 * (z + 1)
    dN[0, 0] = t1
    dN[0, 1] = -t1
    dN[0, 3] = t2
    dN[0, 4] = -t2

    dN[1, 0] = t1
    dN[1, 2] = -t1
    dN[1, 3] = t2
    dN[1, 5] = -t2

    dN[2, 0] = 0.5 * (x + y - 1)
    dN[2, 1] = -0.5 * x
    dN[2, 2] = -0.5 * y
    dN[2, 3] = -dN[2, 0]
    dN[2, 4] = 0.5 * x
    dN[2, 5] = 0.5 * y

    return dN


def V_prism(CoordinateArray:np.ndarray) -> float:
    '''
    V_prism returns the volume of a prism.\n
    CoordinateArray's shape: (6,3)
    '''
    def JacobiDet(x,y,z):

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det = abs(np.linalg.det(J))

        return det

    result = 0.
    for i in range(8):
        result += JacobiDet(PrismIntPoints[i, 0], PrismIntPoints[i,1], PrismIntPoints[i,2]) \
                  * PrismIntPoints[i,3]

    return result


def V_prism27(CoordinateArray:np.ndarray) -> float:
    '''
    V_prism27 returns the volume of a prism,
    which is more accurate than the result of V_prism.\n
    CoordinateArray's shape: (6,3)
    '''
    def JacobiDet(x,y,z):

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det = abs(np.linalg.det(J))

        return det

    result = 0.
    for i in range(27):
        result += JacobiDet(PrismIntPoints27[i, 0], PrismIntPoints27[i,1], PrismIntPoints27[i,2]) \
                  * PrismIntPoints27[i,3] * PrismIntPoints27[i,4]

    return result


def PrismShapeFunction(num:int, x:float, y:float, z:float) -> float:
    '''
    PrismShapeFunction returns prism's shape function,
    where 'num' ,from 0 to 5, means the order number of the shape function.
    '''
    ans = 0.5
    if num < 3:
        ans *= 1 - z
    if num >= 3:
        ans *= 1 + z
    if num == 0 or num == 3:
        ans *= 1 - x - y
    if num == 1 or num == 4:
        ans *= x
    if num == 2 or num == 5:
        ans *= y

    return ans


def k_prism(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:
    '''
    k_prism returns the stiffness matrix of a prism.\n
    CoordinateArray's shape: (6,3)\n
    ThermalConductivity: [k_x, k_y, k_z]
    '''
    k = np.zeros((6, 6))

    for j in range(8):
        x = PrismIntPoints[j, 0]
        y = PrismIntPoints[j, 1]
        z = PrismIntPoints[j, 2]
        const = PrismIntPoints[j, 3]

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det_J = abs(np.linalg.det(J))

        inv_J = np.linalg.inv(J)  # inv_J为Jacobi矩阵逆矩阵
        D = np.dot(inv_J, dN_prism(x, y, z))  # D为形函数对全局坐标的导数组成的3*6矩阵

        for row in range(6):
            for col in range(row+1):

                for i in range(3):
                    ans = ThermalConductivity[i] * det_J * D[i, row] * D[i, col] * const
                    k[row, col] += ans
                    k[col, row] += ans

    for idx in range(6):
        k[idx,idx] *= 0.5

    return k


def k_prism27(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:
    '''
    k_prism27 returns the stiffness matrix of a prism,
    which is more accurate than the result of k_prism.\n
    CoordinateArray's shape: (6,3)\n
    ThermalConductivity: [k_x, k_y, k_z]
    '''

    k = np.zeros((6, 6))

    for j in range(27):
        x = PrismIntPoints27[j, 0]
        y = PrismIntPoints27[j, 1]
        z = PrismIntPoints27[j, 2]
        const = PrismIntPoints27[j, 3]
        const2 = PrismIntPoints27[j, 4]

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det_J = abs(np.linalg.det(J))

        inv_J = np.linalg.inv(J)  # inv_J为Jacobi矩阵逆矩阵
        D = np.dot(inv_J, dN_prism(x, y, z))  # D为形函数对全局坐标的导数组成的3*6矩阵

        for row in range(6):
            for col in range(row+1):

                for i in range(3):
                    ans = ThermalConductivity[i] * det_J * D[i, row] * D[i, col] * const *const2
                    k[row, col] += ans
                    k[col, row] += ans

    for idx in range(6):
        k[idx,idx] *= 0.5

    return k


def m_prism(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:
    '''
    m_prism returns the mass matrix of a prism.\n
    CoordinateArray's shape: (6,3)
    '''
    m = np.zeros((6, 6))

    for i in range(8):
        x = PrismIntPoints[i, 0]
        y = PrismIntPoints[i, 1]
        z = PrismIntPoints[i, 2]
        const = PrismIntPoints[i, 3]
        # coefficient = PrismIntPoints[i, 4]

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det = abs(np.linalg.det(J))

        for row in range(6):
            for col in range(row+1):
                N1 = PrismShapeFunction(row, x, y, z)
                N2 = PrismShapeFunction(col, x, y, z)

                ans = N1 * N2 * det * const
                m[row, col] += ans
                m[col, row] += ans

    for j in range(6):
        m[j, j] *= 0.5

    m *= p * c
    return m


def m_prism27(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:
    '''
    m_prism27 returns the mass matrix of a prism
    which is more accurate than the result of m_prism.\n
    CoordinateArray's shape: (6,3)
    '''

    m = np.zeros((6, 6))

    for i in range(27):
        x = PrismIntPoints27[i, 0]
        y = PrismIntPoints27[i, 1]
        z = PrismIntPoints27[i, 2]
        const1 = PrismIntPoints27[i, 3]
        const2 = PrismIntPoints27[i, 4]

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det = abs(np.linalg.det(J))

        for row in range(6):
            for col in range(row + 1):
                N1 = PrismShapeFunction(row, x, y, z)
                N2 = PrismShapeFunction(col, x, y, z)

                ans = N1 * N2 * det * const1 * const2
                m[row, col] += ans
                m[col, row] += ans

    for j in range(6):
        m[j, j] *= 0.5

    m *= p * c

    return m


def fQ_prism(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:
    '''
    fQ_prism returns a prism element's load vector
    related to the internal heat source.\n
    CoordinateArray's shape: (6,3)
    '''

    f = np.zeros(6)

    for idx in range(8):
        x,y,z,const = PrismIntPoints[idx]
        for i in range(6):
            J = np.dot(dN_prism(x, y, z), CoordinateArray)
            det = abs(np.linalg.det(J))
            f[i] += det * PrismShapeFunction(i, x, y, z) * const

    return f * c * Q


def dN_pyramid(x:float, y:float, z:float) -> np.ndarray:
    '''
    dN_pyramid returns the derivatives of the shape functions of a standard pyramid
    in the natural coordinates x, y, and z directions.
    '''
    dN = np.zeros((3, 5))

    dN[0, 0] = -(1-y-z)/(4*(1-z))
    dN[0, 1] = -dN[0, 0]
    dN[0, 2] = (1+y-z)/(4*(1-z))
    dN[0, 3] = -dN[0, 2]
    dN[0, 4] = 0

    dN[1, 0] = -(1 - x - z) / (4 * (1 - z))
    dN[1, 1] = -(1 + x - z) / (4 * (1 - z))
    dN[1, 2] = -dN[1, 1]
    dN[1, 3] = -dN[1, 0]
    dN[1, 4] = 0

    dN[2, 0] = ((2*z - (1-y)-(1-x))*(1-z) + (1-x-z)*(1-y-z)) / (4*(1-z)**2)
    dN[2, 1] = ((2*z - (1-y)-(1+x))*(1-z) + (1+x-z)*(1-y-z)) / (4*(1-z)**2)
    dN[2, 2] = ((2*z - (1+y)-(1+x))*(1-z) + (1+x-z)*(1+y-z)) / (4*(1-z)**2)
    dN[2, 3] = ((2*z - (1+y)-(1-x))*(1-z) + (1-x-z)*(1+y-z)) / (4*(1-z)**2)
    dN[2, 4] = 1

    return dN


def V_pyramid(CoordinateArray:np.ndarray) -> float:
    '''
    V_pyramid returns the volume of a pyramid.\n
    CoordinateArray's shape: (5,3)
    '''

    def func(x, y, z):
        J = np.dot(dN_pyramid(x, y, z), CoordinateArray)
        det = np.linalg.det(J)
        return abs(det)

    ans = IntInPyramid(func)
    return ans


def k_pyramid(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:
    '''
    k_pyramid returns the stiffness matrix of a pyramid.\n
    CoordinateArray's shape: (5,3)
    '''

    k = np.zeros((5, 5))

    for row in range(5):
        for col in range(row + 1):

            def func(x,y,z):
                ND = dN_pyramid(x, y, z)
                J = np.dot(ND, CoordinateArray)
                det_J = abs(np.linalg.det(J))

                inv_J = np.linalg.inv(J)  # inv_J为Jacobi矩阵逆矩阵
                D = np.dot(inv_J, dN_pyramid(x, y, z))  # D为形函数对全局坐标的导数组成的3*6矩阵

                res = 0.
                for i in range(3):
                    res += det_J * D[i, row] * D[i, col] * ThermalConductivity[i]
                # res = det_J * D[2, row] * D[2, col]
                return res

            ans = IntInPyramid(func)
            k[row, col] += ans
            k[col, row] += ans

    for idx in range(5):
        k[idx,idx] *= 0.5

    return k


def PyramidShapeFunction(i:int, x:float, y:float, z:float) -> float:
    '''
    PyramidShapeFunction returns pyramid's shape function,
    where 'i' ,from 0 to 4, means the order number of the shape function.
    '''
    ans = 0.
    if i == 0:
        ans = (1-x-z)*(1-y-z)/(4*(1-z))
    if i == 1:
        ans = (1+x-z)*(1-y-z)/(4*(1-z))
    if i == 2:
        ans = (1+x-z)*(1+y-z)/(4*(1-z))
    if i == 3:
        ans = (1-x-z)*(1+y-z)/(4*(1-z))
    if i == 4:
        ans = z
    return ans


def m_pyramid(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:
    '''
    m_pyramid returns the mass matrix of a pyramid.\n
    CoordinateArray's shape: (5,3)
    '''

    m = np.zeros((5, 5))

    for row in range(5):
        for col in range(row+1):
            def func(x, y, z):
                J = np.dot(dN_pyramid(x, y, z), CoordinateArray)
                det = abs(np.linalg.det(J))
                res = det * PyramidShapeFunction(row, x, y, z) * PyramidShapeFunction(col, x, y, z)
                return res

            ans = IntInPyramid(func)
            m[row, col] += ans
            m[col, row] += ans

    for j in range(5):
        m[j,j] *= 0.5

    return m * p * c


def fQ_pyramid(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:
    '''
    fQ_pyramid returns a pyramid element's load vector related to the internal heat source.\n
    CoordinateArray's shape: (5,3)
    '''

    f = np.zeros(5)

    for i in range(5):
        def func(x,y,z):
            J = np.dot(dN_pyramid(x, y, z), CoordinateArray)
            det = abs(np.linalg.det(J))
            ans = det * PyramidShapeFunction(i, x, y, z)
            return ans
        f[i] = IntInPyramid(func)
    f *= c * Q
    return f


'''Jacobi_hexa'''
def Jacobi_hexa(i:int, coordinate:np.ndarray) -> np.ndarray:
    dN = ND_hexa[3*i:3*i+3, :]

    J = np.dot(dN, coordinate)

    return J


'''V_hexa returns the volume of a hexahedron.'''
def V_hexa(CoordinateArray:np.ndarray) -> float:
    V = 0.

    for i in range(8):
        det_J = abs(np.linalg.det(Jacobi_hexa(i, CoordinateArray)))
        V += det_J

    return V


abc = np.array([[-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1]])


def HexaShapeFunction(idx:int, x:float, y:float, z:float) -> float:
    ans = (1+abc[idx, 0]*x) * (1+abc[idx, 1]*y) * (1+abc[idx, 2]*z) * 0.125
    return ans


'''m_hexa returns the mass matrix of a hexahedron.'''
def m_hexa(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:
    m = np.zeros((8, 8))

    for i in range(8):
        x, y, z = HexaIntPoints[i]

        det_J = abs(np.linalg.det(Jacobi_hexa(i, CoordinateArray)))

        for row in range(8):
            for col in range(row+1):

                N1 = (1+abc[row, 0]*x) * (1+abc[row, 1]*y) * (1+abc[row, 2]*z)
                N2 = (1+abc[col, 0]*x) * (1+abc[col, 1]*y) * (1+abc[col, 2]*z)
                ans = det_J  * N1 * N2
                m[row, col] += ans
                m[col, row] += ans

    for j in range(8):
        m[j,j] *= 0.5

    m = m * p * c * 0.125**2
    return m


'''k_hexa returns the stiffness matrix of a hexahedron.'''
def k_hexa(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:

    k = np.zeros((8, 8))

    for j in range(8):
        J = Jacobi_hexa(j, CoordinateArray)
        det_J = abs(np.linalg.det(J))
        inv_J = np.linalg.inv(J)  # inv_J为Jacobi矩阵逆矩阵
        D = np.dot(inv_J, ND_hexa[3*j:3*j+3, :])  # D为形函数对全局坐标的导数组成的3*8矩阵

        for row in range(8):
            for col in range(row+1):

                for i in range(3):
                    ans = ThermalConductivity[i] * det_J * D[i, row] * D[i, col]
                    k[row, col] += ans
                    k[col, row] += ans
    for idx in range(8):
        k[idx,idx] *= 0.5
    return k


'''fQ_hexa returns a hexahedron element's load vector related to the internal heat source.'''
def fQ_hexa(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:
    fQ = np.zeros(8)

    for i in range(8):
        x, y, z = HexaIntPoints[i]

        det_J = abs(np.linalg.det(Jacobi_hexa(i, CoordinateArray)))

        for row in range(8):
                N = (1 + abc[row, 0] * x) * (1 + abc[row, 1] * y) * (1 + abc[row, 2] * z)

                fQ[row] += det_J * N

    fQ = fQ * Q * c * 0.125

    return fQ


ab = np.array([[-1, -1],
               [ 1, -1],
               [ 1,  1],
               [-1,  1]])


'''Jacobi_quad'''
def Jacobi_quad(i:int, coor:np.ndarray) -> float:

    x = QuadIntPoints[i, 0]
    y = QuadIntPoints[i, 1]

    dN = np.zeros((2, 4))

    dN[0, 0] = y - 1
    dN[0, 1] = 1 - y
    dN[0, 2] = 1 + y
    dN[0, 3] = -1 - y

    dN[1, 0] = x - 1
    dN[1, 1] = -1 - x
    dN[1, 2] = 1 + x
    dN[1, 3] = 1 - x

    dN = dN * 0.25

    o = np.dot(dN, coor) #(2,3)

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


'''f3_quad returns a quadrilateral element's load vector related to the third type of boundary condition.'''
def f3_quad(CoordinateArray:np.ndarray, h:float, Tf:float) -> np.ndarray:
    F = np.zeros(4)

    for j in range(4):
        for i in range(4):

            x = QuadIntPoints[i, 0]
            y = QuadIntPoints[i, 1]

            F[j] += Jacobi_quad(i, CoordinateArray) * 0.25 * (1 + ab[j, 0] * x) * (1 + ab[j, 1] * y)


    F *= h * Tf
    return F


def S_quad3(CoordinateArray:np.ndarray) -> float:
    '''
    CoordinateArray's shape: (4,3)\n
    get the area of quad in 3d-space
    '''
    a1 = CoordinateArray[0:3,:]
    a2 = np.delete(CoordinateArray, 1, axis=0)
    return S_tri(a1)+S_tri(a2)


def f3_quad_test(CoordinateArray:np.ndarray, h:float, Tf:float) -> np.ndarray:
    return np.ones(4) * h * Tf * S_quad3(CoordinateArray) / 4
    # F = np.zeros(4)
    #
    # for i in range(4):
    #     def func(x, y):
    #         o = np.dot(dN_quad(x,y), CoordinateArray)
    #
    #         a = np.array([[o[0, 0], o[0, 1]],
    #                       [o[1, 0], o[1, 1]]])
    #         b = np.array([[o[0, 1], o[0, 2]],
    #                       [o[1, 1], o[1, 2]]])
    #         c = np.array([[o[0, 2], o[0, 0]],
    #                       [o[1, 2], o[1, 0]]])
    #         A = np.linalg.det(a)
    #         B = np.linalg.det(b)
    #         C = np.linalg.det(c)
    #
    #         Jacobi = (A ** 2 + B ** 2 + C ** 2) ** 0.5
    #         res = QuadShapeFunction(i, x, y) * Jacobi
    #         return res
    #
    #     ans = IntInQuad(func)
    #     F[i] = ans
    # F *= h * Tf
    # return F


'''edge_quad is used to correct the stiffness matrix affected by the third type of boundary conditions.'''
def edge_quad(CoordinateArray:np.ndarray, h:float) -> np.ndarray:
    Edge = np.zeros((4, 4))

    for i in range(4):
        x = QuadIntPoints[i, 0]
        y = QuadIntPoints[i, 1]
        J = Jacobi_quad(i, CoordinateArray)
        for row in range(4):
            for col in range(4):

                N1 = (1 + ab[row, 0] * x)*(1 + ab[row, 1] * y)
                N2 = (1 + ab[col, 0] * x)*(1 + ab[col, 1] * y)
                ans = J * 0.25**2 * N1 * N2
                Edge[row, col] += ans
                # Edge[row, col] += ans

    # for idx in range(4):
    #     Edge[idx,idx] *= 0.5

    Edge *= h
    return Edge


def edge_quad_test(CoordinateArray:np.ndarray, h:float) -> np.ndarray:
    Edge = np.zeros((4, 4))

    for row in range(4):
        for col in range(row+1):
            def func(x,y):
                N1 = QuadShapeFunction(row,x,y)
                N2 = QuadShapeFunction(col,x,y)
                o = np.dot(dN_quad(x, y), CoordinateArray)

                a = np.array([[o[0, 0], o[0, 1]],
                              [o[1, 0], o[1, 1]]])
                b = np.array([[o[0, 1], o[0, 2]],
                              [o[1, 1], o[1, 2]]])
                c = np.array([[o[0, 2], o[0, 0]],
                              [o[1, 2], o[1, 0]]])
                A = np.linalg.det(a)
                B = np.linalg.det(b)
                C = np.linalg.det(c)

                Jacobi = (A ** 2 + B ** 2 + C ** 2) ** 0.5
                res = N1 * N2 * Jacobi
                return res

            ans = IntInQuad(func,4)
            Edge[row, col] += ans
            Edge[col, row] += ans

    for idx in range(4):
        Edge[idx,idx] *= 0.5

    Edge *= h
    return Edge


def f2_quad(CoordinateArray:np.ndarray, q:float) -> np.ndarray:
    F = np.zeros(4)

    for j in range(4):
        for i in range(4):

            x = QuadIntPoints[i, 0]
            y = QuadIntPoints[i, 1]

            F[j] += Jacobi_quad(i, CoordinateArray) * 0.25 * (1 + ab[j, 0] * x) * (1 + ab[j, 1] * y)

    F *= q

    return F


def flow_tetra(CoordinateArray:np.ndarray, p:float, c:float, speed:np.ndarray) -> np.ndarray:
    flow = np.zeros((4, 4))


    J = np.dot(dN_tetra, CoordinateArray)
    det_J = abs(np.linalg.det(J))
    inv_J = np.linalg.inv(J)
    D = np.dot(inv_J, dN_tetra)

    for row in range(4):
        for col in range(4):

            def func(x,y,z):
                N = TetraShapeFunction(row, x, y, z)

                res = 0.
                for i in range(3):
                    res += det_J * N * D[i, col] * speed[col, i]

                return res

            ans = IntInTetra(func)
            flow[row, col] = ans

    flow *= p * c
    return flow


def flow_hexa(CoordinateArray:np.ndarray, p:float, c:float, speed:np.ndarray) -> np.ndarray:
    flow = np.zeros((8, 8))

    for i in range(8):
        x, y, z = HexaIntPoints[i]

        J = Jacobi_hexa(i, CoordinateArray)
        det_J = abs(np.linalg.det(J))
        inv_J = np.linalg.inv(J)
        D = np.dot(inv_J, ND_hexa[3 * i:3 * i + 3, :])

        for row in range(8):
            for col in range(8):
                N = (1 + abc[row, 0] * x) * (1 + abc[row, 1] * y) * (1 + abc[row, 2] * z) / 8

                for j in range(3):
                    ans = det_J * speed[col, j] * D[j, col] * N
                    flow[row, col] += ans

    flow *= p * c
    return flow


def flow_prism(CoordinateArray:np.ndarray, p:float, c:float, speed:np.ndarray) -> np.ndarray:
    flow = np.zeros((6, 6))

    for j in range(8):
        x = PrismIntPoints[j, 0]
        y = PrismIntPoints[j, 1]
        z = PrismIntPoints[j, 2]
        const = PrismIntPoints[j, 3]

        ND = dN_prism(x, y, z)
        J = np.dot(ND, CoordinateArray)
        det_J = abs(np.linalg.det(J))

        inv_J = np.linalg.inv(J)
        D = np.dot(inv_J, dN_prism(x, y, z))

        for row in range(6):
            N = PrismShapeFunction(row, x, y, z)
            for col in range(6):

                for i in range(3):
                    ans = speed[col, i] * det_J * D[i, col] * N * const
                    flow[row, col] += ans

    flow *= p * c
    return flow


def flow_pyramid(CoordinateArray:np.ndarray, p:float, c:float, speed:np.ndarray) -> np.ndarray:
    flow = np.zeros((5, 5))

    for row in range(5):
        for col in range(5):

            def func(x,y,z):
                N = PyramidShapeFunction(row, x, y, z)

                ND = dN_pyramid(x, y, z)
                J = np.dot(ND, CoordinateArray)
                det_J = abs(np.linalg.det(J))

                inv_J = np.linalg.inv(J)
                D = np.dot(inv_J, dN_pyramid(x, y, z))

                res = 0.
                for i in range(3):
                    res += det_J * N * D[i, col] * speed[col, i]
                # res = det_J * D[2, row] * D[2, col]
                return res

            ans = IntInPyramid(func)
            flow[row, col] = ans


    flow *= p * c
    return flow


'''axial symmetry'''
def QuadShapeFunction(idx:int, x:float, y:float) -> float:
    factor1, factor2 = ab[idx]
    ans = (1 + x * factor1) * (1 + y * factor2) / 4
    return ans


def IntInQuad(func, num_of_nodes:int) -> float:
    '''calculate the integration of func(x, y) in standard quadrilateral.'''
    ans = 0.

    if num_of_nodes == 4:
        gauss_points = np.array([[const1, const1],
                                 [const1, -const1],
                                 [-const1, const1],
                                 [-const1, -const1]])
        for i in range(4):
            x, y = gauss_points[i]
            ans += func(x, y)

    if num_of_nodes == 9:
        c = 0.6**0.5
        w1 = 25/81
        w2 = 40/81
        w3 = 64/81
        gauss_points9 = np.array([
            [-c,-c,w1],[-c,0,w2],[-c,c,w1],
            [0,-c,w2],[0,0,w3],[0,c,w2],
            [c,-c,w1],[c,0,w2],[c,c,w1],
        ])
        for i in range(9):
            x, y, weight = gauss_points9[i]
            ans += weight * func(x, y)

    return ans


def dN_quad(x:float, y:float) -> np.ndarray:
    dN = np.zeros((2, 4))

    dN[0, 0] = y - 1
    dN[0, 1] = 1 - y
    dN[0, 2] = 1 + y
    dN[0, 3] = -1 - y

    dN[1, 0] = x - 1
    dN[1, 1] = -1 - x
    dN[1, 2] = 1 + x
    dN[1, 3] = 1 - x

    dN = dN * 0.25
    return dN


def S_quad(CoordinateArray:np.ndarray) -> float:
    def func(x, y):
        J = np.dot(dN_quad(x, y), CoordinateArray)
        det = np.linalg.det(J)
        return abs(det)

    ans = IntInQuad(func,4)
    return ans


def m_quad(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:
    '''CoordinateArray's size:4*2'''
    m = np.zeros((4, 4))

    for row in range(4):
        for col in range(row+1):
            def func(x, y):
                J = np.dot(dN_quad(x, y), CoordinateArray)
                det = abs(np.linalg.det(J))
                res = det * QuadShapeFunction(row, x, y) * QuadShapeFunction(col, x, y)
                return res

            ans = IntInQuad(func,4)

            m[row, col] += ans
            m[col, row] += ans

    for j in range(4):
        m[j,j] *= 0.5

    m = m * p * c
    return m


def k_quad(CoordinateArray:np.ndarray, conductivity:float) -> np.ndarray:
    k = np.zeros((4,4))

    for row in range(4):
        for col in range(row + 1):
            def func(x, y):
                ND = dN_quad(x, y)
                J = np.dot(ND, CoordinateArray)
                det_J = abs(np.linalg.det(J))

                inv_J = np.linalg.inv(J) # 2*2
                D = np.dot(inv_J, dN_quad(x, y)) # 2*4

                res = 0.
                for i in range(2):
                    res += det_J * D[i, row] * D[i, col] * conductivity

                return res

            ans = IntInQuad(func,4)
            k[row, col] += ans
            k[col, row] += ans

    for idx in range(4):
        k[idx,idx] *= 0.5

    return k


def fQ_quad(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:
    f = np.zeros(4)

    for i in range(4):
        def func(x, y):
            J = np.dot(dN_quad(x, y), CoordinateArray)
            det = abs(np.linalg.det(J))
            ans = det * QuadShapeFunction(i, x, y)
            return ans
        f[i] = IntInQuad(func,4)

    return f * c * Q


def S_2d_tri(CoordinateArray:np.ndarray) -> float:
    '''
    CoordinateArray: (3,2)
    '''
    x1, y1 = CoordinateArray[0]
    x2, y2 = CoordinateArray[1]
    x3, y3 = CoordinateArray[2]

    # 三角形单元面积
    S = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

    return S


def m_tri(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:

    m = p * c * S_2d_tri(CoordinateArray) / 24 * np.array([[2, 1, 1],
                                                           [1, 2, 1],
                                                           [1, 1, 2]])

    return m


def k_tri(CoordinateArray:np.ndarray, *conductivity) -> np.ndarray:
    if len(conductivity) == 1:
        con_x = conductivity[0]
        con_y = con_x
    else:
        con_x = conductivity[0]
        con_y = conductivity[1]

    x1, y1 = CoordinateArray[0]
    x2, y2 = CoordinateArray[1]
    x3, y3 = CoordinateArray[2]

    b1 = y2 - y3
    c1 = x3 - x2
    b2 = y3 - y1
    c2 = x1 - x3
    b3 = y1 - y2
    c3 = x2 - x1

    k1 = np.array([[b1 * b1, b1 * b2, b1 * b3],
                   [b1 * b2, b2 * b2, b2 * b3],
                   [b1 * b3, b2 * b3, b3 * b3]])

    k2 = np.array([[c1 * c1, c1 * c2, c1 * c3],
                   [c1 * c2, c2 * c2, c2 * c3],
                   [c1 * c3, c2 * c3, c3 * c3]])

    k = (k1 * con_x + k2 * con_y) / (4 * S_2d_tri(CoordinateArray))

    return k


def fQ_tri(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:
    f = S_2d_tri(CoordinateArray) * np.ones(3) / 3

    return f * c * Q


def length_of_line(CoordinateArray:np.ndarray) -> float:
    '''calculate the length of a line.\n
    CoordinateArray's size: 2*2.
    '''
    x1, y1 = CoordinateArray[0]
    x2, y2 = CoordinateArray[1]

    length = ((x2-x1)**2+(y2-y1)**2)**0.5

    return length


def f2_line(CoordinateArray:np.ndarray, q:float) -> np.ndarray:

    return np.ones(2) * q * length_of_line(CoordinateArray) / 2


def f3_line(CoordinateArray:np.ndarray, h:float, Tf:float) -> np.ndarray:

    return np.ones(2) * h * Tf * length_of_line(CoordinateArray) / 2


def IntInLine(func):
    ans = func(-const1) + func(const1)
    return ans


def edge_line(CoordinateArray:np.ndarray, h:float) -> np.ndarray:
    '''CoordinateArray's size: 2*2'''
    edge = length_of_line(CoordinateArray) * h * np.array([[1/3,1/6],[1/6,1/3]])

    return edge


if __name__=='__main__':
    # x = np.array([[-1,-1],
    #               [ 1,-1],
    #               [ 1, 1],
    #               [-1, 1]])
    # # print(S_quad(x))
    # # print(m_quad(x,1,1))
    # # print(k_quad(x,1))
    # # print(fQ_quad(x,1,1))
    y = np.array([[0,0,2],
                  [2,0,2],
                  [2,4,1],
                  [0,4,1]])
    print(f3_quad(y,1,1))
    #
    print(f3_quad_test(y,1,1))
    print(edge_quad(y,1))
    print(edge_quad_test(y,1))
    # # z = np.array([[0,0],[1,20]])
    # # print(edge_line(z,1))
    z = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[10,0,1],[1,1,1],[0,1,1],
    ])
