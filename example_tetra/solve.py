from multiprocessing import Pool, freeze_support, cpu_count
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as ssl
import time
import read_tetra_dat
from HCMVG import m_tetra, k_tetra, edge_tri, f3_tri


#   进程数
num_process = 2 * cpu_count() // 3


points, cells, faces = read_tetra_dat.getdata('Tetra.dat')

n1 = points.shape[0]    # n1:numbers of points
n2 = cells.shape[0]     # n2:numbers of cells
n3 = faces.shape[0]     # n3:numbers of faces


# 材料参数
c = 489  # 比热
p = 7900  # 密度
lamda = 37  # 热导率
h = 100


def create_m(i:int) -> np.ndarray:

    coor = np.zeros((4, 3))     #   获取8节点坐标数组

    for j in range(4):
        coor[j] = points[int(cells[i, j])]

    m_e = m_tetra(coor,p,c)

    return m_e


def create_k(i:int) -> np.ndarray:

    coor = np.zeros((4, 3))  # 获取8节点坐标数组

    for j in range(4):
        coor[j] = points[int(cells[i, j])]

    k_e = k_tetra(coor,[lamda,lamda,lamda])


    return k_e


def create_edge(i:int) -> np.ndarray:

    coor = np.zeros((3, 3))

    for j in range(3):
        coor[j] = points[int(faces[i, j])]

    edge_e = edge_tri(coor,h)

    return edge_e


def create_f(i:int) -> np.ndarray:

    coor = np.zeros((3, 3))

    for j in range(3):
        coor[j] = points[int(faces[i, j])]

    f_e = f3_tri(coor, h, 1000)

    return f_e


def multicore() -> tuple:
    pool = Pool(processes=num_process)    #   建立进程池pool


    res = pool.map(create_m, range(n2))
    M = sp.lil_matrix((n1, n1))

    for i in range(n2):
        for row in range(4):
            for col in range(4):
                M[int(cells[i, row]), int(cells[i, col])] += res[i][row, col]

    print('M completed')


    res2 = pool.map(create_k, range(n2))
    K1 = sp.lil_matrix((n1, n1))

    for i in range(n2):
        for row in range(4):
            for col in range(4):
                K1[int(cells[i, row]), int(cells[i, col])] += res2[i][row, col]

    print('K completed')


    res3 = pool.map(create_edge, range(n3))
    Edge = sp.lil_matrix((n1, n1))

    for i in range(n3):
        for row in range(3):
            for col in range(3):
                Edge[int(faces[i, row]), int(faces[i, col])] += res3[i][row, col]

    print('Edge completed')


    res4 = pool.map(create_f, range(n3))
    F = sp.lil_matrix((n1, 1))

    for i in range(n3):
        for row in range(3):
            F[int(faces[i, row]), 0] += res4[i][row]

    print('F completed')


    pool.close()
    pool.join()


    K = K1 + Edge

    location_x, location_y = M.nonzero()
    np.savetxt('x.txt', location_x)
    np.savetxt('y.txt', location_y)
    # location_xk, location_yk = K.nonzero()
    # np.savetxt('x_k.txt', location_xk)
    # np.savetxt('y_k.txt', location_yk)

    # delta_x = location_x - location_xk
    # delta_y = location_y - location_yk
    #
    # num = 0
    # for i in delta_x:
    #     if i != 0:
    #         num+=1
    # print(f"num:{num}")
    # num2 = 0
    # for i in delta_y:
    #     if i != 0:
    #         num2+=1
    # print(f"num:{num2}")


    np.savetxt('F.txt', F.toarray())
    # print(M1.data)
    # np.savetxt('M.txt', M1.data)
    # np.savetxt('K.txt', K.data)

    np.savetxt('M0.txt', M.toarray())
    np.savetxt('K0.txt', K.toarray())
    return M, K, F


def solve(K:np.ndarray, M:np.ndarray, F:np.ndarray, u:np.ndarray) -> np.ndarray:

    M_csr = sp.csr_matrix(M)
    K_csr = sp.csr_matrix(K)
    F_csr = sp.csr_matrix(F)
    u_csr = sp.csr_matrix(u)

    np.savetxt('M.txt', M_csr.data)
    np.savetxt('K.txt', K_csr.data)
    # answer = np.ones(n1) * 1000

    # answer = np.ones(n1) * 1000

    dt = 10
    t_max = 2000
    t = np.arange(dt, t_max + dt, dt)

    result = np.zeros((n1,int(t_max/dt)))


    tic1 = time.time()
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

        '''后差分'''
        sum1 = K_csr + M_csr / dt
        sum2 = M_csr.dot(u_csr) / dt + F_csr

        sum2 = sum2.toarray()   #   在ssl.cg中向量为非稀疏矩阵格式


        # answer = ssl.cg(sum1,sum2)[0]
        answer = np.linalg.solve(sum1.toarray(), sum2)

        # answer = ssl.spsolve(sum1,sum2)
        print(np.min(answer),np.max(answer))
        # print(f'max:{np.max(answer)}')
        # print(f'min:{np.min(answer)}')
        # print(f'mean:{np.mean(answer)}')

        # np.savetxt('result.txt', answer)
        u_csr[:, 0] = answer
        result[:, int(ti/dt)-1] = answer[:,0]

    toc1 = time.time()
    print(toc1 - tic1)

    # np.savetxt('result.txt', result)
    return result


if __name__=='__main__':
    freeze_support()

    M, K, F = multicore()

    # print(M)
    # print(K_sum)

    u = np.ones((n1, 1)) * 20

    temperature = solve(K, M, F, u)

    np.savetxt('result.txt', temperature)
    # print(f'max:{np.max(temperature)}')
    # print(f'min:{np.min(temperature)}')
    # print(f'mean:{np.mean(temperature)}')

    # print(temperature)
    print(temperature[:, -1].max(), temperature[:, -1].min(), temperature[:, -1].mean())
    # row = M.getrow()
    # col = M.getcol()
    #
