from multiprocessing import Pool, cpu_count
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as ssl
import time
import HCMVG


class Solve:
    def __init__(self, points: np.ndarray, cells: np.ndarray, boundaries: np.ndarray) -> None:
        self.num_process = 2 * cpu_count() // 3

        self.points = points
        self.cells = cells
        self.boundaries = boundaries

        self.n1 = self.points.shape[0]
        self.n2 = self.cells.shape[0]
        self.n3 = self.boundaries.shape[0]

        self.element_type = ""
        self.element_node_num = 0
        self.boundary_node_num = 0
        self.dimension = 0

        self.dt = 0.1
        self.t_max = 100.0

        self.density = 0.0
        self.specific_heat = 0.0
        self.conductivity = []
        self.heat_transfer_coefficient = 0.0
        self.environment_temperature = 0.0

        self.initial_temperature = 0.0

        self.function_m = HCMVG.m_hexa
        self.function_k = HCMVG.k_hexa
        self.function_edge = HCMVG.edge_quad
        self.function_f = HCMVG.f3_quad

    def setting_num_process(self, num_process: int) -> None:
        if num_process > int(cpu_count * 0.9):
            print("the value is too big, the number of process will be the default value")
        else:
            self.num_process = num_process

    def setting_element_type(self, element_type: str) -> None:
        self.element_type = element_type

        if self.element_type == "hexahedron":
            self.element_node_num = 8
            self.boundary_node_num = 4
            self.dimension = 3

        elif self.element_type == "tetrahedron":
            self.element_node_num = 4
            self.boundary_node_num = 3
            self.dimension = 3

            self.function_m = HCMVG.m_tetra
            self.function_k = HCMVG.k_tetra
            self.function_edge = HCMVG.edge_tri
            self.function_f = HCMVG.f3_tri

        elif self.element_type == "quadrilateral" or self.element_type == "axisymmetric_quadrilateral":
            self.element_node_num = 4
            self.boundary_node_num = 2
            self.dimension = 2

            if self.element_type == "quadrilateral":
                self.function_m = HCMVG.m_quad
                self.function_k = HCMVG.k_quad
                self.function_edge = HCMVG.edge_line
                self.function_f = HCMVG.f3_line

            elif self.element_type == "axisymmetric_quadrilateral":
                self.function_m = HCMVG.m_axisymmetric_quad
                self.function_k = HCMVG.k_axisymmetric_quad
                self.function_edge = HCMVG.edge_axisymmetric_line
                self.function_f = HCMVG.f3_axisymmetric_line

    def setting_total_time(self, total_time: float) -> None:
        self.t_max = total_time

    def setting_time_step(self, time_step: float) -> None:
        self.dt = time_step

    def create_m(self, i: int) -> np.ndarray:
        coor = np.zeros((self.element_node_num, self.dimension))

        for j in range(self.element_node_num):
            coor[j] = self.points[int(self.cells[i, j])]

        m_e = self.function_m(coor, self.density, self.specific_heat)
        return m_e

    def create_k(self, i: int) -> np.ndarray:
        coor = np.zeros((self.element_node_num, self.dimension))

        for j in range(self.element_node_num):
            coor[j] = self.points[int(self.cells[i, j])]

        k_e = self.function_k(coor, self.conductivity)

        return k_e

    def create_edge(self, i: int) -> np.ndarray:

        coor = np.zeros((self.boundary_node_num, self.dimension))

        for j in range(self.boundary_node_num):
            coor[j] = self.points[int(self.boundaries[i, j])]

        edge_e = self.function_edge(coor, self.heat_transfer_coefficient)

        return edge_e

    def create_f(self, i: int) -> np.ndarray:

        coor = np.zeros((self.boundary_node_num, self.dimension))

        for j in range(self.boundary_node_num):
            coor[j] = self.points[int(self.boundaries[i, j])]

        f_e = self.function_f(coor, self.heat_transfer_coefficient, self.environment_temperature)

        return f_e

    def solve(self) -> np.ndarray:
        print("start assembling......")
        tic1 = time.time()

        pool = Pool(processes=self.num_process)

        res = pool.map(self.create_m, range(self.n2))
        M = sp.lil_matrix((self.n1, self.n1))

        for i in range(self.n2):
            for row in range(self.element_node_num):
                for col in range(self.element_node_num):
                    M[int(self.cells[i, row]), int(self.cells[i, col])] += res[i][row, col]

        print('mass_matrix assembled')

        res2 = pool.map(self.create_k, range(self.n2))
        K1 = sp.lil_matrix((self.n1, self.n1))

        for i in range(self.n2):
            for row in range(self.element_node_num):
                for col in range(self.element_node_num):
                    K1[int(self.cells[i, row]), int(self.cells[i, col])] += res2[i][row, col]

        print('stiffness_matrix assembled')

        res3 = pool.map(self.create_edge, range(self.n3))
        Edge = sp.lil_matrix((self.n1, self.n1))

        for i in range(self.n3):
            for row in range(self.boundary_node_num):
                for col in range(self.boundary_node_num):
                    Edge[int(self.boundaries[i, row]), int(self.boundaries[i, col])] += res3[i][row, col]

        print('edge_matrix assembled')

        res4 = pool.map(self.create_f, range(self.n3))
        F = sp.lil_matrix((self.n1, 1))

        for i in range(self.n3):
            for row in range(self.boundary_node_num):
                F[int(self.boundaries[i, row]), 0] += res4[i][row]

        print('load_vector assembled')

        pool.close()
        pool.join()

        K = K1 + Edge

        toc1 = time.time()

        M_csr = sp.csr_matrix(M)
        K_csr = sp.csr_matrix(K)
        F_csr = sp.csr_matrix(F)
        u_csr = sp.csr_matrix(np.ones((self.n1, 1)) * self.initial_temperature)

        dt = self.dt
        t_max = self.t_max
        t = np.arange(dt, t_max + dt, dt)

        result = np.zeros((self.n1, int(t_max/dt)))

        tic2 = time.time()

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

            sum2 = sum2.toarray()   # 在ssl.cg中向量为非稀疏矩阵格式

            answer = ssl.cg(sum1, sum2)[0]

            print(f't={ti}s, min={np.min(answer)}, mean={np.mean(answer)}, max={np.max(answer)}')

            u_csr[:, 0] = answer
            result[:, int(ti/dt)-1] = answer

        toc2 = time.time()
        print()
        print(f'assembling matrix takes time:{toc1 - tic1}s')
        print(f'solving the equation takes time:{toc2 - tic2}s')

        return result
