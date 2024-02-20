from multiprocessing import Pool, freeze_support, cpu_count
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as ssl
import time
import HCMVGdemo2


points = np.load('D:/pydtest/tetrahedron/points.npy')
cells1 = np.load('D:/pydtest/tetrahedron/cells1.npy')
cells2 = np.load('D:/pydtest/tetrahedron/cells2.npy')
boundary = np.load('D:/pydtest/tetrahedron/boundary.npy')
cells1 = cells1 - int(1)
cells2 = cells2 - int(1)
boundary = boundary - int(1)

n1 = 4973


def create_m1(i:int) -> np.ndarray:

    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells1[i, j]), k]

    p1 = 1.3
    c1 = 1000
    m_e = HCMVGdemo2.m_tetra(coor,p1,c1)

    return m_e


def create_k1(i:int) -> np.ndarray:

    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells1[i, j]), k]

    k1 = 0.0267
    k_e = HCMVGdemo2.k_tetra(coor,[k1,k1,k1])

    return k_e


def create_m2(i:int) -> np.ndarray:

    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells2[i, j]), k]

    p2 = 7900
    c2 = 489
    m_e = HCMVGdemo2.m_tetra(coor,p2,c2)

    return m_e


def create_k2(i:int) -> np.ndarray:

    coor = np.zeros((4, 3))

    for j in range(4):
        for k in range(3):
            coor[j, k] = points[int(cells2[i, j]), k]

    k2 = 37
    k_e = HCMVGdemo2.k_tetra(coor,[k2,k2,k2])

    return k_e


def multicore() -> tuple:
    num_process = 2 * cpu_count() // 3
    pool = Pool(processes=num_process)    #   建立进程池pool


    res = pool.map(create_m1, range(cells1.shape[0]))

    M1 = sp.lil_matrix((n1, n1))

    for i in range(cells1.shape[0]):
        for row in range(4):
            for col in range(4):
                M1[int(cells1[i, row]), int(cells1[i, col])] += res[i][row, col]


    res2 = pool.map(create_k1, range(cells1.shape[0]))


    # K1 = np.zeros((n1, n1))
    K1 = sp.lil_matrix((n1, n1))

    for i in range(cells1.shape[0]):
        for row in range(4):
            for col in range(4):
                K1[int(cells1[i, row]), int(cells1[i, col])] += res2[i][row, col]


    res3 = pool.map(create_m2, range(cells2.shape[0]))


    for i in range(cells2.shape[0]):
        for row in range(4):
            for col in range(4):
                M1[int(cells2[i, row]), int(cells2[i, col])] += res3[i][row, col]


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

    big_number = 1e17
    fixed_temperature = 1000.0
    for i in range(len(boundary)):
        idx = boundary[i]

        K_csr[idx, idx] = big_number * K_csr[idx, idx]
        temp = K_csr[idx, idx]
        F[idx] = fixed_temperature * temp

    # K_new, F_new = MultiplyBigNumber.modifiedAandB(K_csr, F, boundary, 1000)
    F_csr = sp.csr_matrix(F)
    # M_csr = sp.csr_matrix(M_new)


    answer = np.ones(n1) * 20

    dt = 50
    t_max = 20000
    t = np.arange(dt, t_max + dt, dt)

    txt = np.zeros((n1,int(t_max/dt)))


    sum1 = K_csr + M_csr / dt
    tic1 = time.time()
    for ti in t:

        sum2 = F_csr + M_csr.dot(u_csr) / dt
        sum2 = sum2.toarray()   #   在ssl.cg中向量为非稀疏矩阵格式

        # answer = ssl.gmres(sum1,sum2,x0=answer)[0]
        ans = ssl.spsolve(sum1,sum2)
        print(np.mean(ans))
        u_csr[:, 0] = ans
        txt[:, int(ti/dt)-1] = ans

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
