from multiprocessing import Pool, freeze_support, cpu_count
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as ssl
import time
# import HCMVGdemo2


def V_tetra(CoordinateArray:np.ndarray) -> float:

    x1, y1, z1 = (CoordinateArray[0, 0], CoordinateArray[0, 1], CoordinateArray[0, 2])
    x2, y2, z2 = (CoordinateArray[1, 0], CoordinateArray[1, 1], CoordinateArray[1, 2])
    x3, y3, z3 = (CoordinateArray[2, 0], CoordinateArray[2, 1], CoordinateArray[2, 2])
    x4, y4, z4 = (CoordinateArray[3, 0], CoordinateArray[3, 1], CoordinateArray[3, 2])

    V_matrix = np.array([[1, x1, y1, z1],
                         [1, x2, y2, z2],
                         [1, x3, y3, z3],
                         [1, x4, y4, z4]])

    V = abs(np.linalg.det(V_matrix)) / 6  # 单元体积

    return V


'''m_tetra returns the mass matrix of a tetrahedron.'''
def m_tetra(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:

    m = V_tetra(CoordinateArray) * p * c / 20 * np.array([[2, 1, 1, 1],
                                                          [1, 2, 1, 1],
                                                          [1, 1, 2, 1],
                                                          [1, 1, 1, 2]])
    return m


'''k_tetra returns the stiffness matrix of a tetrahedron.'''
def k_tetra(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:
    x1, y1, z1 = (CoordinateArray[0, 0], CoordinateArray[0, 1], CoordinateArray[0, 2])
    x2, y2, z2 = (CoordinateArray[1, 0], CoordinateArray[1, 1], CoordinateArray[1, 2])
    x3, y3, z3 = (CoordinateArray[2, 0], CoordinateArray[2, 1], CoordinateArray[2, 2])
    x4, y4, z4 = (CoordinateArray[3, 0], CoordinateArray[3, 1], CoordinateArray[3, 2])

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
    if V_tetra(CoordinateArray) == 0:
        print('!!!!!!!!')
    k = (k1 + k2 + k3) / (36 * V_tetra(CoordinateArray))

    return k


points = np.load('points.npy')
cells1 = np.load('cells1.npy')
cells2 = np.load('cells2.npy')
boundary = np.load('boundary.npy')
cells1 = cells1 - int(1)
cells2 = cells2 - int(1)
boundary = boundary - int(1)

n1 = len(points)


def create_m1(i:int) -> np.ndarray:
    if i%1000 == 0:
        print(f'm1:{i}')
    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells1[i, j]), k]

    p1 = 1.3
    c1 = 1000
    m_e = m_tetra(coor,p1,c1)

    return m_e


def create_k1(i:int) -> np.ndarray:
    if i%1000 == 0:
        print(f'k1:{i}')
    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells1[i, j]), k]

    k1 = 0.0267
    k_e = k_tetra(coor,[k1,k1,k1])

    return k_e


def create_m2(i:int) -> np.ndarray:
    if i%1000 == 0:
        print(f'm2:{i}')
    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells2[i, j]), k]

    p2 = 7900
    c2 = 489
    m_e = m_tetra(coor,p2,c2)

    return m_e


def create_k2(i:int) -> np.ndarray:
    if i%1000 == 0:
        print(f'k2:{i}')
    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells2[i, j]), k]

    k2 = 37
    k_e = k_tetra(coor,[k2,k2,k2])

    return k_e


def multicore() -> tuple:
    num_process = 2 * cpu_count() // 3

    M1 = sp.lil_matrix((n1, n1))

    pool = Pool(processes=num_process)    #   建立进程池pool

    res = pool.map(create_m1, range(cells1.shape[0]))

    print('calculate completed')
    for i in range(cells1.shape[0]):
        for row in range(4):
            for col in range(4):
                M1[int(cells1[i, row]), int(cells1[i, col])] += res[i][row, col]
    print('1111111111')
    pool.close()
    pool.join()
    print('m1 completed')

    pool = Pool(processes=num_process)    #   建立进程池pool

    res3 = pool.map(create_m2, range(cells2.shape[0]))

    for i in range(cells2.shape[0]):
        for row in range(4):
            for col in range(4):
                M1[int(cells2[i, row]), int(cells2[i, col])] += res3[i][row, col]

    pool.close()
    pool.join()


    K1 = sp.lil_matrix((n1, n1))

    pool = Pool(processes=num_process)    #   建立进程池pool
    res2 = pool.map(create_k1, range(cells1.shape[0]))


    for i in range(cells1.shape[0]):
        for row in range(4):
            for col in range(4):
                K1[int(cells1[i, row]), int(cells1[i, col])] += res2[i][row, col]

    pool.close()
    pool.join()

    pool = Pool(processes=num_process)    #   建立进程池pool

    res4 = pool.map(create_k2, range(cells2.shape[0]))

    for i in range(cells2.shape[0]):
        for row in range(4):
            for col in range(4):
                K1[int(cells2[i, row]), int(cells2[i, col])] += res4[i][row, col]


    pool.close()
    pool.join()

    return M1, K1


def solve(K:sp.lil_matrix, M:sp.lil_matrix, u:sp.lil_matrix) -> np.ndarray:

    M_csr = sp.csr_matrix(M)
    K_csr = sp.csr_matrix(K)
    u_csr = sp.csr_matrix(u)

    F = np.zeros((n1,1))

    big_number = 1e18
    fixed_temperature = 1000.0
    for i in range(len(boundary)):
        idx = boundary[i]

        K_csr[idx, idx] = big_number * K_csr[idx, idx]
        temp = K_csr[idx, idx]
        F[idx] = fixed_temperature * temp

    # K_new, F_new = MultiplyBigNumber.modifiedAandB(K_csr, F, boundary, 1000)
    F_csr = sp.csr_matrix(F)
    # M_csr = sp.csr_matrix(M_new)


    # ans = np.ones(n1) * 20

    dt = 50
    t_max = 20000
    t = np.arange(dt, t_max + dt, dt)

    txt = np.zeros((n1,int(t_max/dt)))


    sum1 = K_csr + M_csr / dt
    lu = ssl.splu(sum1)
    tic1 = time.time()
    for ti in t:

        sum2 = F_csr + M_csr.dot(u_csr) / dt
        sum2 = sum2.toarray()   #   在ssl.cg中向量为非稀疏矩阵格式

        # answer = ssl.gmres(sum1,sum2,x0=answer)[0]
        # ans = ssl.spsolve(sum1,sum2)
        ans = lu.solve(sum2)

        print(np.mean(ans))
        percent = ti * 100 / t_max
        print(f"{percent}%")
        u_csr[:, 0] = ans
        txt[:, int(ti/dt)-1] = ans[:,0]

        if ti == t_max:
            np.savetxt('last.txt', ans)
    toc1 = time.time()
    print(toc1 - tic1)

    print(txt[:,-1])
    return txt


if __name__=='__main__':
    freeze_support()

    tic = time.time()

    M, K = multicore()

    tpc = time.time()
    print(tpc-tic)

    u = np.ones((n1, 1)) * 20

    temperature = solve(K, M, u)

    # print(temperature)
    print(temperature[:,-1].max(), temperature[:,-1].min(), temperature[:,-1].mean())
    np.savetxt('result.txt', temperature)
