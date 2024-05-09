import numpy as np
from scipy.sparse.linalg import svds, cg
import time
from check import load_vector
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd


def pod(snapshot: np.ndarray, percent: float) -> np.ndarray:

    u, s, v = randomized_svd(snapshot, snapshot.shape[1])

    print(np.shape(u))
    print(np.shape(v))
    # s = s[::-1]
    num = 0
    for i in range(len(s)):
        epsilon = 100 * np.sum(s[0:i]) / np.sum(s)

        if epsilon > percent:
            num = i
            break

    print(num)
    print(s)

    # sorted_indices = np.argsort(s)[::-1]
    # u_sorted = u[:, sorted_indices]
    # p = u_sorted[:, 0:num]
    p = u[:, 0:num]
    return p


def reduced_matrix_solve(T0: np.ndarray, dt: float, t_max: float) -> np.ndarray:
    '''
    n refers to the number of points.\n
    :param p: (n, s)
    :param M: (n, n)
    :param K: (n, n)
    :param F: (n, 1)
    :param T0: (n, 1)
    :param dt: the size of one time step
    :param t_max: time of heat conduction
    :return: (n, int(t_max/dt))
    '''
    p = np.loadtxt('p.txt')
    pmp0 = np.loadtxt('pmp.txt')
    pkp = np.loadtxt('pkp.txt')
    pm = np.loadtxt('pm.txt')
    surface = np.loadtxt('surface.txt')

    start = time.time()

    p_T = p.transpose()  # (s, n)
    pmp = pmp0 / dt  # (s, s)

    A = pkp + pmp  # (s, s)
    # print(np.shape(A))
    result = np.zeros((len(p), int(t_max/dt)))
    T = T0  # (n, 1)
    t = np.arange(dt, t_max + dt, dt)

    for ti in t:
        x0 = 0.01 + ti * 2 / 10000
        F = load_vector(surface, p.shape[0], 100, [x0, 0.004], 0.001)
        # print(np.shape(p_T@F))
        # print(np.shape(pm@T))
        # F_csr = sp.csr_matrix(F)
        B = p_T @ F + pm @ T / dt  # (s, 1)
        # print(np.shape(B))
        ans = np.linalg.solve(A, B)  # (s, 1)
        # ans = cg(A, B)[0]  # (s, 1)
        # print(np.shape(ans))
        # ans = cg.conjugate_gradient(A,B,x0=np.zeros(p.shape[1]))
        T = p @ ans  # (n, 1)
        # print(np.shape(T))
        result[:, int(ti/dt)-1] = T[:, 0]
        # result[:, int(ti/dt)-1] = T
        print(np.min(T), np.max(T))

    end = time.time()
    print(f'cost time:{end-start}')

    np.savetxt('pod_result.txt', result)


if __name__ == '__main__':
    p = np.loadtxt('p.txt')
    T0 = np.ones((len(p), 1)) * 20
    reduced_matrix_solve(T0, 1, 300)

    # x = np.loadtxt('result.txt')
    # p = pod(x, 99.99)
    # np.savetxt('p.txt', p)

