from multiprocessing import Pool, freeze_support, cpu_count
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as ssl
import time
# import read_tetra_dat
from HCMVG import m_tetra, k_tetra, edge_tri, f3_tri
from check import load_vector


#   进程数
num_process = 2 * cpu_count() // 3

cells = np.loadtxt('cell.txt')
cells = cells - np.ones((len(cells), 4))
points = np.loadtxt('coor.txt')
surface = np.loadtxt('surface.txt')
# points, cells, faces = read_tetra_dat.getdata('Tetra.dat')

n1 = points.shape[0]    # n1:numbers of points
n2 = cells.shape[0]     # n2:numbers of cells
# n3 = faces.shape[0]     # n3:numbers of faces


# 材料参数
c = 489  # 比热
p = 7900  # 密度
lamda = 37  # 热导率
h = 100
q = 1000


def create_m(i: int) -> np.ndarray:

    coor = np.zeros((4, 3))

    for j in range(4):
        coor[j] = points[int(cells[i, j])]

    m_e = m_tetra(coor, p, c)

    return m_e


def create_k(i: int) -> np.ndarray:

    coor = np.zeros((4, 3))  # 获取8节点坐标数组

    for j in range(4):
        coor[j] = points[int(cells[i, j])]

    k_e = k_tetra(coor, [lamda, lamda, lamda])

    return k_e


# def create_f(i: int) -> np.ndarray:
#
#     coor = np.zeros((3, 3))
#
#     for j in range(3):
#         coor[j] = points[int(faces[i, j])]
#
#     f_e = f3_tri(coor, h, 1000)
#
#     return f_e


def multicore() -> tuple:
    pool = Pool(processes=num_process)

    res = pool.map(create_m, range(n2))
    M = sp.lil_matrix((n1, n1))

    for i in range(n2):
        for row in range(4):
            for col in range(4):
                M[int(cells[i, row]), int(cells[i, col])] += res[i][row, col]

    print('M completed')

    res2 = pool.map(create_k, range(n2))
    K = sp.lil_matrix((n1, n1))

    for i in range(n2):
        for row in range(4):
            for col in range(4):
                K[int(cells[i, row]), int(cells[i, col])] += res2[i][row, col]

    print('K completed')

    pool.close()
    pool.join()

    return M, K


def solve(K: np.ndarray, M: np.ndarray, u: np.ndarray) -> np.ndarray:

    M_csr = sp.csr_matrix(M)
    K_csr = sp.csr_matrix(K)
    # F_csr = sp.csr_matrix(F)
    u_csr = sp.csr_matrix(u)

    dt = 1
    t_max = 200
    t = np.arange(dt, t_max + dt, dt)

    result = np.zeros((n1, int(t_max/dt)))

    tic1 = time.time()

    sum1 = K_csr + M_csr / dt
    for ti in t:
        # '''中心差分'''
        # sum1 = (K_csr) / 2 + M_csr / dt
        # sum2 = np.dot(M_csr / dt - (K_csr) / 2, u_csr) + F_csr

        # '''伽辽金'''
        # sum1 = (K_csr) * 2 / 3 + M_csr / dt
        # sum2 = np.dot(M_csr / dt - (K_csr) / 3, u_csr) + F_csr

        # '''前差分'''
        # sum1 = M_csr / dt
        # sum2 = np.dot(M_csr / dt - (K_csr), u_csr) + F_csr
        x0 = 0.01 + ti*2/10000
        F = load_vector(surface, n1, 100, [x0, 0.004], 0.001)
        F_csr = sp.csr_matrix(F)
        '''后差分'''

        sum2 = M_csr.dot(u_csr) / dt + F_csr

        sum2 = sum2.toarray()  # 在ssl.cg中向量为非稀疏矩阵格式

        answer = ssl.cg(sum1,sum2)[0]
        # answer = np.linalg.solve(sum1.toarray(), sum2)

        # answer = ssl.spsolve(sum1,sum2)
        print(np.min(answer), np.max(answer))

        u_csr[:, 0] = answer
        result[:, int(ti/dt)-1] = answer

    toc1 = time.time()
    print(toc1 - tic1)

    return result


if __name__ == '__main__':
    freeze_support()

    M, K = multicore()

    p = np.loadtxt('p.txt')
    p_T = p.transpose()
    np.savetxt('pmp.txt', p_T @ M @ p)
    np.savetxt('pkp.txt', p_T @ K @ p)
    np.savetxt('pm.txt', p_T @ M)


    u = np.ones((n1, 1)) * 20

    temperature = solve(K, M, u)

    np.savetxt('result.txt', temperature)
    # print(f'max:{np.max(temperature)}')
    # print(f'min:{np.min(temperature)}')
    # print(f'mean:{np.mean(temperature)}')

    # print(temperature)
    # print(temperature[:, -1].max(), temperature[:, -1].min(), temperature[:, -1].mean())

