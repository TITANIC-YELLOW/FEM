import numpy as np


def pod(snapshot: np.ndarray) -> np.ndarray:

    u, s, v = np.linalg.svd(snapshot)

    num = 0
    for i in range(len(s)):
        epsilon = 100 * np.sum(s[0:i]) / np.sum(s)

        if epsilon > 99.99:
            num = i
            break

    print(num)
    print(s)

    p = u[:, 0:num]
    return p


def reduced_matrix(p: np.ndarray,
                   M:np.ndarray, K:np.ndarray, F:np.ndarray, T0:np.ndarray,
                   dt:float, t_max:float) -> np.ndarray:

    p_T = p.transpose()
    pmp = p_T @ M @ p / dt

    A = p_T @ K @ p + pmp

    T = T0
    t = np.arange(dt, t_max + dt, dt)
    for ti in t:

        B = p_T @ F + p_T @ M @ T / dt
        ans = np.linalg.solve(A, B)

        T = p @ ans

        print(np.max(T),np.min(T))


if __name__=='__main__':

    x = np.loadtxt('result.txt')
    p = pod(x)

    M = np.loadtxt('M0.txt')
    K = np.loadtxt('K0.txt')
    F = np.loadtxt('F.txt')
    T0 = np.ones(len(p)) * 20
    reduced_matrix(p, M, K, F, T0, 10, 2000)




