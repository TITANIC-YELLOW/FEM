import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
from scipy import interpolate
import meshio
import gmsh
import sys
import math
import os
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QInputDialog
import time
# import xlrd
import wmi
import hashlib

import scipy.sparse as sp
import scipy.sparse.linalg as ssl

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# class MyFigure(FigureCanvas):
#     def __init__(self,width=5, height=4, dpi=100):
#         #第一步：创建一个创建Figure
#         self.fig = Figure(figsize=(width, height), dpi=dpi)
#         #第二步：在父类中激活Figure窗口
#         super().__init__(self.fig)
#         self.axes = self.fig.add_subplot(111)


# 获取设备的主板ID
m_wmi = wmi.WMI()

board_info = m_wmi.Win32_BaseBoard()
if len(board_info) > 0:
    board_id = board_info[0].SerialNumber.strip().strip('.')


md5 = hashlib.md5('bigyellow'.encode('utf-8'))  # 选择加密方式，初始化一个加密
md5.update(board_id.encode('utf-8'))  # 将要加密的内容，添加到m中
result = md5.hexdigest()


sha256 = hashlib.sha256('bigyellow'.encode('utf-8'))
sha256.update(result.encode('utf-8'))
result2 = sha256.hexdigest()


# 获取屏幕分辨率
app1 = QApplication([])
screen = app1.desktop().screenGeometry()

bili = screen.width() / screen.height()
bili2 = 1.25 - bili / 3

width = int(screen.width() * bili2)
height = int(screen.height() * 0.8)
pic_size = int(height * 0.41)


# 全局变量：

name2 = 0
name3 = 0


# 材料参数
h = 0
lamda = 0
t0 = 0
tf = 0
time_hc = 0
# original coordinate
oc = 0

# 初始顶点坐标
ppp = 0

# 圆角倒角数据
sss = []
ttt = []
time_c1 = 0

uuu = []
vvv = []
www = []
time_d1 = 0


# 角度数据
xxx = []
yyy = []
zzz = []
time_ch = 0


j = 0


t = 0

# 初始化程序开始与结束时间
b_time = 0
n_time = 0


# 网格密度
density = 50  # 默认值为 50
# 时间步数
stepnum = 10  # 默认值为 10

X = 0
Y = 0
spe = []
arc = []

ChangedArr = 0
ChangedArr2 = 0

Dc_matrix = np.zeros((1, 2))
beta_matrix = np.zeros((1, 2))
cf_matrix = np.zeros((1, 2))


# 运算示例数据
# Dc_matrix = np.array([[0, 1e-11], [2, 2e-11]])
# beta_matrix = np.array([[0, 1e-8], [2, 2e-8]])
# cf_matrix = np.array([[0, 1.2], [100000, 1.6]])
t0 = 0.17
time_hc = 100000


yesORno = 0

x_max = 0
x_min = 0
y_max = 0
y_min = 0

canvas = 0


# class MyDialog(QtWidgets.QDialog):
#     mt = MyThread()
#     def closeEvent(self,event):
#         # super().closeEvent()
#         try:
#             if len(self.matrix) != 0:
#
#                 MyDialog.mt.result.emit()
#                 MyDialog.mt.start()
#
#                 event.accept()
#
#         except:
#             # self.msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, 'WARNING','未保存，确认退出？',
#             #                                 QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
#             # self.msg_box.exec_()
#             # if self.msg_box == QtWidgets.QMessageBox.Yes:
#             #     event.accept()
#             # else:
#             #
#             #     event.ignore()
#             # if msg_box == QtWidgets.QMessageBox.No:
#             #     msg_box.close()
#
#             # msg_box.setWindowIcon(QtGui.QIcon('logo.ico'))  # 加载图标
#
#
#             event.accept()


# 重写双击事件
# class MyMouseDoubleClicked(QtWidgets.QLabel):
#     my_signal = pyqtSignal(str)
#
#     def __init__(self):
#         super().__init__()
#
#
#     def mouseDoubleClickEvent(self, event):
#         objectname = self.objectName()
#         self.my_signal.emit(objectname)


class send(QThread):
    sig = pyqtSignal(float)

    def __init__(self):
        super().__init__()

    #def run(self):
        #for i in range(100):
         #   time.sleep(0.01)
          #  self.signal.emit(int(i))

    def run(self):

        global b_time
        b_time = time.time()  # 开始时间


        # 插值得到各时间点的Cf
        canshu = np.zeros((stepnum, 1))

        row = cf_matrix.shape[0]
        if row > stepnum + 1:
            pass
        else:
            for i in range(stepnum):
                for j in range(row-1):
                    delta_x1 = (i + 1) / stepnum - j / (row-1)
                    delta_x2 = (j + 1) / (row-1) - (i + 1) / stepnum

                    # 插值
                    if delta_x1 >= 0 and delta_x2 > 0:
                        canshu[i, 0] = (row-1) * (delta_x1 * cf_matrix[j + 1, 1] + delta_x2 * cf_matrix[j, 1])


            canshu[-1, 0] = cf_matrix[-1, 1]


            # print(canshu)


            # 将全局变量ppp赋值给points
            # ppp已在show1时被修改
            points = ppp

            # 毫米(mm)--->米(m)
            points = points / 1000

            hang = points.shape[0]

            # 列表final记录经过倒角、圆角后点的坐标
            final = []
            for i in range(hang):
                a = np.zeros((2, 1))
                a[0, 0] = points[i, 0]
                a[1, 0] = points[i, 1]

                final.append(a)

            # 使用circle_center和cut函数修改final的元素
            for i in range(len(sss)):
                final[sss[i]] = circle_center(points, sss[i], ttt[i] / 1000)

            for i in range(len(uuu)):
                final[uuu[i]] = cut(points, uuu[i], vvv[i] / 1000, www[i] / 1000)

            # time_p为实际的直线段个数
            time_p = hang + time_d1

            # 将final中的元素排布成点的数组pp
            # pp是为了找到所有直线段
            # 圆弧端点间的直线段也要考虑
            pp = np.zeros((time_p + time_c1, 2))

            r = 0
            for i in range(hang):
                if final[i].shape[0] == 2 and final[i].shape[1] == 1:
                    pp[i + r, 0] = final[i][0, 0]
                    pp[i + r, 1] = final[i][1, 0]
                if final[i].shape[0] == 2 and final[i].shape[1] == 2:
                    pp[i + r, 0] = final[i][0, 0]
                    pp[i + r, 1] = final[i][0, 1]
                    pp[i + r + 1, 0] = final[i][1, 0]
                    pp[i + r + 1, 1] = final[i][1, 1]

                    r += 1
                if final[i].shape[0] == 3:
                    pp[i + r, 0] = final[i][1, 0]
                    pp[i + r, 1] = final[i][1, 1]
                    pp[i + r + 1, 0] = final[i][2, 0]
                    pp[i + r + 1, 1] = final[i][2, 1]

                    r += 1

            self.sig.emit(float(10))


            # 调用gmsh库生成网格
            gmsh.initialize()
            gmsh.model.add("t3")
            mid = (points[:, 0].max() - points[:, 0].min() + points[:, 1].max() - points[:, 1].min()) / 2
            lc = mid / density


            P = gmsh.model.geo.addPoint
            L = gmsh.model.geo.addLine
            C = gmsh.model.geo.addCircleArc

            # LC为储存点集的列表，每个元素是一个点集
            # 点集也是列表
            # 若为线段，则点集里的元素个数是2
            # 若是圆弧，则点集里的元素个数是3
            LC = []

            # addPoint
            num_P = 1  # num_P为点的编号，由1开始
            for i in range(hang):
                # 当该列表元素为一个点的坐标时：
                if final[i].shape[0] == 2 and final[i].shape[1] == 1:
                    if i != hang - 1:
                        P(final[i][0, 0], final[i][1, 0], 0, lc, num_P)
                        LC.append([num_P, num_P + 1])
                        num_P += 1
                    # 特殊情况：最后一个点
                    if i == hang - 1:
                        P(final[i][0, 0], final[i][1, 0], 0, lc, num_P)
                        LC.append([num_P, 1])
                # 当该列表元素为两个点的坐标时（即cut）：
                if final[i].shape[0] == 2 and final[i].shape[1] == 2:
                    P(final[i][0, 0], final[i][0, 1], 0, lc, num_P)
                    P(final[i][1, 0], final[i][1, 1], 0, lc, num_P + 1)

                    if i != hang - 1:
                        LC.append([num_P, num_P + 1, num_P + 2, 0])
                        num_P += 2
                    else:
                        LC.append([num_P, num_P + 1, 1, 0])
                # 当该列表元素为三个点的坐标时(circle)：
                if final[i].shape[0] == 3:
                    P(final[i][1, 0], final[i][1, 1], 0, lc, num_P)
                    P(final[i][0, 0], final[i][0, 1], 0, lc, num_P + 1)
                    P(final[i][2, 0], final[i][2, 1], 0, lc, num_P + 2)

                    LC.append([num_P, num_P + 1, num_P + 2])

                    if i != hang - 1:
                        LC.append([num_P + 2, num_P + 3])
                        num_P += 3
                    else:
                        LC.append([num_P + 2, 1])

            # addLine AND addCircleArc
            num_LC = 1  # num_LC为线段或圆弧段编号，由1开始
            for i in range(len(LC)):
                # 线段
                if len(LC[i]) == 2:
                    L(LC[i][0], LC[i][1], num_LC)
                    num_LC += 1
                # 圆弧段
                if len(LC[i]) == 3:
                    C(LC[i][0], LC[i][1], LC[i][2], num_LC)
                    num_LC += 1
                # 倒角段
                if len(LC[i]) == 4:
                    L(LC[i][0], LC[i][1], num_LC)
                    L(LC[i][1], LC[i][2], num_LC + 1)
                    num_LC += 2

            Loop = []
            for i in range(time_p + time_c1):
                Loop.append(i + 1)

            gmsh.model.geo.addCurveLoop(Loop, 1)
            gmsh.model.geo.addPlaneSurface([1], 1)
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)

            t1 = ''.join([name2, '/', 't3.vtk'])
            gmsh.write(t1)

            self.sig.emit(float(20))


            # 保存网格及单元数据至txt文件

            mesh = meshio.read(t1)

            p1 = ''.join([name2, '/', 'points.txt'])
            np.savetxt(p1, mesh.points)

            c1 = ''.join([name2, '/', 'cells.txt'])
            np.savetxt(c1, mesh.cells_dict["triangle"])

            l1 = ''.join([name2, '/', 'lines.txt'])
            np.savetxt(l1, mesh.cells_dict["line"])


            self.sig.emit(float(25))


            # 圆弧数组记录圆心坐标和半径
            # 第一第二列记录圆心的xy坐标，第三列记录半径r
            # 第四列记录换热系数，第五列记录边界初始温度
            circle = np.zeros((time_c1, 5))

            for i in range(time_c1):
                circle[i, 0] = final[sss[i]][0, 0]
                circle[i, 1] = final[sss[i]][0, 1]
                circle[i, 2] = ttt[i] / 1000

                circle[i, 3] = h
                circle[i, 4] = tf

            # kb数组记录每个边的k,b值以及换热系数和边界初始温度
            kb = np.zeros((hang + time_d1 + time_c1, 4))

            for i in range(hang + time_d1 + time_c1):
                if i != hang + time_d1 + time_c1 - 1:
                    if pp[i, 0] - pp[i + 1, 0] != 0:
                        kb[i, 0] = (pp[i, 1] - pp[i + 1, 1]) / (pp[i, 0] - pp[i + 1, 0])
                        kb[i, 1] = pp[i, 1] - kb[i, 0] * pp[i, 0]
                    if pp[i, 0] - pp[i + 1, 0] == 0:
                        kb[i, 0] = 1e8
                        kb[i, 1] = pp[i, 0]

                # 特殊情况：最后一点与第一个点的连线
                if i == hang + time_d1 + time_c1 - 1:
                    if pp[i, 0] - pp[0, 0] != 0:
                        kb[i, 0] = (pp[i, 1] - pp[0, 1]) / (pp[i, 0] - pp[0, 0])
                        kb[i, 1] = pp[i, 1] - kb[i, 0] * pp[i, 0]
                    if pp[i, 0] - pp[0, 0] == 0:
                        kb[i, 0] = 1e8
                        kb[i, 1] = pp[0, 0]

            # 换热系数and环境温度
            for i in range(hang + time_d1 + time_c1):
                kb[i, 2] = h
                kb[i, 3] = tf


            # 节点数组，储存节点的x，y坐标
            arr1 = np.loadtxt(p1)
            # num1为节点个数
            num1 = len(arr1)

            realnode = num1 - time_c1


            # 初始化整体矩阵及向量
            K = np.zeros((realnode, realnode))
            m = np.zeros((realnode, realnode))
            Edge = np.zeros((realnode, realnode))
            F = np.zeros((realnode, 1))

            # 单元数组，储存每个单元内节点的编号
            arr2 = np.loadtxt(c1)
            # num2为单元个数
            num2 = len(arr2)

            # s_point获取圆心在arr1里的编号
            s_point = []
            for i in range(num1):
                for j in range(time_c1):
                    if abs(arr1[i, 0] - circle[j, 0]) <= 1e-5 and abs(arr1[i, 1] - circle[j, 1]) <= 1e-5:
                        s_point.append(i)

                if time_c1 == len(s_point):
                    break

            # 对s_point进行升序排序
            s_point.sort()

            # 删除arr1中代表圆心的编号
            arr1 = np.delete(arr1, s_point, 0)


            global x_max, y_max, x_min, y_min
            # 储存图形x和y坐标的最大值
            Max_xy = np.max(arr1, axis=0)
            x_max = Max_xy[0]
            y_max = Max_xy[1]
            # print(x_max, y_max)
            # 储存图形x和y坐标的最小值
            Min_xy = np.min(arr1, axis=0)
            x_min = Min_xy[0]
            y_min = Min_xy[1]
            # print(x_min, y_min)


            global ChangedArr
            ChangedArr = arr1


            # 修改arr2
            for i in range(time_c1):
                for j in range(num2):
                    for k in range(3):
                        if i != time_c1 - 1:
                            if s_point[i] < arr2[j, k] < s_point[i + 1]:
                                arr2[j, k] = arr2[j, k] - (i + 1)
                        if i == time_c1 - 1:
                            if arr2[j, k] > s_point[i]:
                                arr2[j, k] -= time_c1


            global ChangedArr2
            ChangedArr2 = arr2


            # line数组（记录边界线段两点的编号）
            arr3 = np.loadtxt(l1)
            # num3为线段个数
            num3 = len(arr3)

            # 修改arr3
            for i in range(time_c1):
                for j in range(num3):
                    for k in range(2):
                        if i != time_c1 - 1:
                            if s_point[i] < arr3[j, k] < s_point[i + 1]:
                                arr3[j, k] = arr3[j, k] - (i + 1)
                        if i == time_c1 - 1:
                            if arr3[j, k] > s_point[i]:
                                arr3[j, k] -= time_c1


            self.sig.emit(float(30))


            # 初始化坐标数组
            x = np.zeros(realnode)
            y = np.zeros(realnode)

            for i in range(realnode):
                x[i] = arr1[i, 0]
                y[i] = arr1[i, 1]

            x = x * 1000
            y = y * 1000

            pxx = ''.join([name2, '/', 'x.txt'])
            np.savetxt(pxx, x)
            pyy = ''.join([name2, '/', 'y.txt'])
            np.savetxt(pyy, y)


            # 开始组装
            # 由于扩散系数和传递系数都随碳浓度变化，所以每计算一个时间步就要重新组装矩阵和向量


            # 先求出各单元的面积>>>>>>>>>
            # S为面积数组
            S = np.zeros(num2)

            K_list = []

            for i in range(num2):
                x1 = arr1[int(arr2[i, 0]), 0]
                y1 = arr1[int(arr2[i, 0]), 1]
                x2 = arr1[int(arr2[i, 1]), 0]
                y2 = arr1[int(arr2[i, 1]), 1]
                x3 = arr1[int(arr2[i, 2]), 0]
                y3 = arr1[int(arr2[i, 2]), 1]

                # 三角形单元面积
                S[i] = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

                # 单元质量矩阵
                m_e = S[i] / 24 * np.array([[2, 1, 1],
                                            [1, 2, 1],
                                            [1, 1, 2]])

                # 组装质量矩阵
                m[int(arr2[i, 0]), int(arr2[i, 0])] += m_e[0, 0]
                m[int(arr2[i, 0]), int(arr2[i, 1])] += m_e[0, 1]
                m[int(arr2[i, 0]), int(arr2[i, 2])] += m_e[0, 2]
                m[int(arr2[i, 1]), int(arr2[i, 0])] += m_e[1, 0]
                m[int(arr2[i, 1]), int(arr2[i, 1])] += m_e[1, 1]
                m[int(arr2[i, 1]), int(arr2[i, 2])] += m_e[1, 2]
                m[int(arr2[i, 2]), int(arr2[i, 0])] += m_e[2, 0]
                m[int(arr2[i, 2]), int(arr2[i, 1])] += m_e[2, 1]
                m[int(arr2[i, 2]), int(arr2[i, 2])] += m_e[2, 2]

                b1 = y2 - y3
                c1 = x3 - x2

                # a2 = x3 * y1 - x1 * y3
                b2 = y3 - y1
                c2 = x1 - x3

                # a3 = x1 * y2 - x2 * y1
                b3 = y1 - y2
                c3 = x2 - x1

                K_e1 = np.array([[b1 * b1, b1 * b2, b1 * b3],
                                 [b1 * b2, b2 * b2, b2 * b3],
                                 [b1 * b3, b2 * b3, b3 * b3]])

                K_e2 = np.array([[c1 * c1, c1 * c2, c1 * c3],
                                 [c1 * c2, c2 * c2, c2 * c3],
                                 [c1 * c3, c2 * c3, c3 * c3]])

                K_list.append((K_e1 + K_e2) / (4 * S[i]))


            # 将质量矩阵转化为稀疏矩阵
            m_csr = sp.csr_matrix(m)


            # 设置传热时间和时间步长
            t_max = time_hc

            looptime = stepnum

            dt = t_max / looptime

            # 时间数组
            global t
            t = np.arange(dt, t_max + dt, dt)

            yes = 1
            internal = float(60 / looptime)

            answer = np.ones(realnode) * t0

            # 定义初始浓度
            u = np.ones((realnode, 1)) * t0
            u_csr = sp.csr_matrix(u)

            # print(time.time())
            # 遍历每个时间步
            for ti in t:
                K = np.zeros((realnode, realnode))
                Edge = np.zeros((realnode, realnode))
                F = np.zeros((realnode, 1))

                # Edge、F的组装（利用arr3）
                for i in range(num3):
                    # 线段
                    for j in range(hang + time_d1 + time_c1):
                        dif1 = 0
                        dif2 = 0
                        if kb[j, 0] != 1e8:
                            dif1 = arr1[int(arr3[i, 0]), 0] * kb[j, 0] + kb[j, 1] - arr1[int(arr3[i, 0]), 1]
                            dif2 = arr1[int(arr3[i, 1]), 0] * kb[j, 0] + kb[j, 1] - arr1[int(arr3[i, 1]), 1]

                        if kb[j, 0] == 1e8:
                            dif1 = arr1[int(arr3[i, 0]), 0] - kb[j, 1]
                            dif2 = arr1[int(arr3[i, 1]), 0] - kb[j, 1]

                        if abs(dif1) <= 1e-5 and abs(dif2) <= 1e-5:
                            # ds为微小线段的长度
                            ds = (((arr1[int(arr3[i, 0]), 1] - arr1[int(arr3[i, 1]), 1]) ** 2 +
                                   (arr1[int(arr3[i, 0]), 0] - arr1[int(arr3[i, 1]), 0]) ** 2) ** 0.5)


                            # 先计算出该单元的平均碳浓度
                            c_m1 = (u_csr[int(arr3[i, 0]), 0] + u_csr[int(arr3[i, 1]), 0]) / 2

                            #print(c_m1)
                            row2 = beta_matrix.shape[0]

                            for m in range(row2 - 1):
                                dx1 = c_m1 - beta_matrix[m, 0]
                                dx2 = beta_matrix[m + 1, 0] - c_m1
                                if dx1 >= 0 and dx2 > 0:
                                    beta = (beta_matrix[m, 1] * dx2 + beta_matrix[m + 1, 1] * dx1) / (dx1 + dx2)
                                    break

                            # print(beta)

                            Edge[int(arr3[i, 0]), int(arr3[i, 0])] += beta * ds / 3
                            Edge[int(arr3[i, 0]), int(arr3[i, 1])] += beta * ds / 6
                            Edge[int(arr3[i, 1]), int(arr3[i, 0])] += beta * ds / 6
                            Edge[int(arr3[i, 1]), int(arr3[i, 1])] += beta * ds / 3

                            F[int(arr3[i, 0]), 0] += beta * ds / 2
                            F[int(arr3[i, 1]), 0] += beta * ds / 2



                    # 圆弧
                    for k in range(time_c1):
                        dis_1 = (arr1[int(arr3[i, 0]), 0] - circle[k, 0]) ** 2 + (
                                    arr1[int(arr3[i, 0]), 1] - circle[k, 1]) ** 2
                        dif3 = dis_1 - circle[k, 2] ** 2

                        dis_2 = (arr1[int(arr3[i, 1]), 0] - circle[k, 0]) ** 2 + (
                                    arr1[int(arr3[i, 1]), 1] - circle[k, 1]) ** 2
                        dif4 = dis_2 - circle[k, 2] ** 2

                        if abs(dif3) <= 1e-5 and abs(dif4) <= 1e-5:
                            ds2 = (arr1[int(arr3[i, 0]), 0] - arr1[int(arr3[i, 1]), 0]) ** 2 + \
                                  (arr1[int(arr3[i, 0]), 1] - arr1[int(arr3[i, 1]), 1]) ** 2
                            ds = ds2 ** 0.5

                            # 先计算出该单元的平均碳浓度
                            c_m2 = (u_csr[int(arr3[i, 0]), 0] + u_csr[int(arr3[i, 1]), 0]) / 2


                            row2 = beta_matrix.shape[0]

                            for n in range(row2 - 1):
                                dx1 = c_m2 - beta_matrix[n, 0]
                                dx2 = beta_matrix[n + 1, 0] - c_m2
                                if dx1 >= 0 and dx2 > 0:
                                    beta = (beta_matrix[n, 1] * dx2 + beta_matrix[n + 1, 1] * dx1) / (dx1 + dx2)
                                    break

                            Edge[int(arr3[i, 0]), int(arr3[i, 0])] += beta * ds / 3
                            Edge[int(arr3[i, 0]), int(arr3[i, 1])] += beta * ds / 6
                            Edge[int(arr3[i, 1]), int(arr3[i, 0])] += beta * ds / 6
                            Edge[int(arr3[i, 1]), int(arr3[i, 1])] += beta * ds / 3

                            F[int(arr3[i, 0]), 0] += beta * ds / 2
                            F[int(arr3[i, 1]), 0] += beta * ds / 2

                total = 0
                total2 = 0

                row1 = Dc_matrix.shape[0]
                # 刚度矩阵的组装
                for i in range(num2):

                    hhh = time.time()

                    # 各个单元的扩散系数
                    dif_co = 0

                    # 先计算出该单元的平均碳浓度
                    c_m3 = (answer[int(arr2[i, 0])] + answer[int(arr2[i, 1])] + answer[int(arr2[i, 2])]) / 3

                    iii = time.time()
                    # 插值得到Dc
                    for j in range(row1 - 1):
                        dx1 = c_m3 - Dc_matrix[j, 0]
                        dx2 = Dc_matrix[j + 1, 0] - c_m3
                        if dx1 >= 0 and dx2 > 0:
                            dif_co = (Dc_matrix[j, 1] * dx2 + Dc_matrix[j + 1, 1] * dx1) / (dx1 + dx2)
                            break

                    jjj = time.time()
                    total += iii - hhh
                    total2 += jjj - iii
                    
                    K_e = K_list[i] * dif_co

                    K[int(arr2[i, 0]), int(arr2[i, 0])] += K_e[0, 0]
                    K[int(arr2[i, 0]), int(arr2[i, 1])] += K_e[0, 1]
                    K[int(arr2[i, 0]), int(arr2[i, 2])] += K_e[0, 2]
                    K[int(arr2[i, 1]), int(arr2[i, 0])] += K_e[1, 0]
                    K[int(arr2[i, 1]), int(arr2[i, 1])] += K_e[1, 1]
                    K[int(arr2[i, 1]), int(arr2[i, 2])] += K_e[1, 2]
                    K[int(arr2[i, 2]), int(arr2[i, 0])] += K_e[2, 0]
                    K[int(arr2[i, 2]), int(arr2[i, 1])] += K_e[2, 1]
                    K[int(arr2[i, 2]), int(arr2[i, 2])] += K_e[2, 2]


                K_csr = sp.csr_matrix(K)

                print(time.time())
                # 将numpy转化为csr格式
                Edge_csr = sp.csr_matrix(Edge)
                F_csr = sp.csr_matrix(F)

                answer = ssl.spsolve(K_csr + Edge_csr + m_csr / dt,
                                    m_csr.dot(u_csr) / dt + F_csr * canshu[yes - 1, 0])

                #u_dense = u_csr.toarray()
                # print(time.time())
                for i in range(realnode):
                    u_csr[i, 0] = answer[i]

                # 保存每一时刻的温度
                if ti != t_max:
                    temperature = ''.join([name2, '/', 'temperature', str(ti), '.txt'])
                else:
                    temperature = ''.join([name2, '/', 'temperature.txt'])

                np.savetxt(temperature, answer)

                # print(time.time())
                self.sig.emit(float(30 + internal * yes))
                yes += 1


            global n_time
            n_time = time.time()  # 结束时间


            self.sig.emit(float(100))

            print(total)
            print(total2)

class ReturnPressed(QThread):
    sentence = pyqtSignal(str)

    def run(self):
        self.sentence.emit()


# 子线程类，内含自定义的信号
class MyThread(QThread):
    result = pyqtSignal()

    def run(self):
        self.result.emit()


# Ui_quit类用以产生退出程序前展示的界面
class Ui_quit(object):

    def setupUi(self, quit):
        quit.setWindowFlags(Qt.WindowCloseButtonHint)
        quit.setObjectName("quit")
        quit.resize(int(height * 0.225), int(height * 0.09))
        quit.setWindowTitle("退出程序")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(quit)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label = QtWidgets.QLabel(quit)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.pushButton = QtWidgets.QPushButton(quit)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.pushButton_2 = QtWidgets.QPushButton(quit)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.retranslateUi(quit)
        self.pushButton.clicked.connect(self.signal)
        self.pushButton.clicked.connect(quit.close)
        #self.pushButton.clicked.connect()
        self.pushButton_2.clicked.connect(quit.close)
        QtCore.QMetaObject.connectSlotsByName(quit)

    def retranslateUi(self, quit):
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("quit", "是否保存并退出？"))
        self.pushButton.setText(_translate("quit", "Yes"))
        self.pushButton_2.setText(_translate("quit", "No"))

    def signal(self):
        global j
        j = 1


# 重写QMainWindow里的closeEvent
# 使在关闭程序前弹出窗口询问是否保存并退出
class MyMainWindow(QtWidgets.QMainWindow):


    def closeEvent(self, event):
        # 调用Ui_quit
        new = QtWidgets.QDialog()
        n = Ui_quit()
        n.setupUi(new)
        new.exec_()

        if j == 1:
            self.mt = MyThread()
            self.mt.result.connect(ui.save)

            self.mt.start()

            event.accept()
        else:
            event.ignore()


def cloud_chart(coordinate_x, coordinate_y, temp):

    #
    # # plt.figure()
    # # 节点坐标数组
    # xx = np.linspace(x_min * 1000, x_max * 1000, 100)
    # yy = np.linspace(y_min * 1000, y_max * 1000, 100)
    # # 生成二维数据坐标点
    # px, py = np.meshgrid(xx, yy)
    #
    # z = interpolate.griddata((coordinate_x, coordinate_y), temp, (px, py), method="linear")
    #
    # for f in range(100):
    #     for g in range(100):
    #         arr1 = ChangedArr
    #         num = len(arr1)
    #
    #         distance = np.zeros(num)
    #
    #         for i in range(num):
    #             distance[i] = (arr1[i, 0] - xx[f] * 0.001) ** 2 + (arr1[i, 1] - yy[g] * 0.001) ** 2
    #
    #         index = np.argmin(distance)  # 得到距离插值点最近的已知点的索引
    #         # 判断所选择的点是否在封闭图形内
    #         arr2 = ChangedArr2
    #         num_cell = len(arr2)
    #
    #         list_cell = []
    #         for i in range(num_cell):
    #             for j in range(3):
    #                 if int(arr2[i, j]) == index:
    #                     list_cell.append(i)
    #
    #         flag = 0
    #         for i in range(len(list_cell)):
    #             # 三角形各顶点的编号
    #             b1 = arr2[list_cell[i], 0]
    #             b2 = arr2[list_cell[i], 1]
    #             b3 = arr2[list_cell[i], 2]
    #
    #             x1 = arr1[int(b1), 0]
    #             y1 = arr1[int(b1), 1]
    #             x2 = arr1[int(b2), 0]
    #             y2 = arr1[int(b2), 1]
    #             x3 = arr1[int(b3), 0]
    #             y3 = arr1[int(b3), 1]
    #
    #             # 求得四个三角形的面积，再作比较，判断输入点是否在三角形内
    #             s = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)
    #             s1 = abs((xx[f] * (y2 - y3) + x2 * (y3 - yy[g]) + x3 * (yy[g] - y2)) / 2) * 0.001
    #             s2 = abs((x1 * (yy[g] - y3) + xx[f] * (y3 - y1) + x3 * (y1 - yy[g])) / 2) * 0.001
    #             s3 = abs((x1 * (y2 - yy[g]) + x2 * (yy[g] - y1) + xx[f] * (y1 - y2)) / 2) * 0.001
    #
    #             if (s - s1 - s2 - s3) * 1e9 >= -1:
    #                 flag += 1
    #
    #                 break
    #             else:
    #                 pass
    #
    #
    #         if flag == 0:
    #             z[f, g] = None
    #
    #
    # #
    # fig, ax = plt.subplots(figsize=(6, 6))  # 创建子图
    #
    # # 设置云图颜色范围以及颜色梯度划分
    # levels = np.linspace(temp.min(), temp.max(), 500)
    #
    # # 设置cmap为jet，最小值为蓝色，最大值为红色
    # # 绘制轮廓图
    # cset = ax.contourf(px, py, z, levels, cmap=plt.get_cmap('jet'))
    #
    # # 设置云图坐标范围以及坐标轴标签
    # ax.set_xlim([x_min * 1000, x_max * 1000])
    # ax.set_ylim([y_min * 1000, y_max * 1000])
    # ax.set_xlabel("X(mm)")
    # ax.set_ylabel("Y(mm)")
    #
    # # 设置色条的刻度，标签
    # cbar = fig.colorbar(cset)
    # cbar.set_label('c(wt%)')

    plt.figure(figsize=(6, 6))  # 创建子图
    size = int(34 - 0.3 * density) + 4 # size为scatter中点的大小

    plt.scatter(coordinate_x, coordinate_y, c=temp, s=size, cmap="jet")


    cbar = plt.colorbar()
    cbar.set_label('c(wt%)')

    plt.axis('equal')
    plt.title("distribution image")
    plt.xlabel("X(mm)")
    plt.ylabel("Y(mm)")


def angle(p_num, direction, degree):

    sin_theta = math.sin(degree * math.pi / 180)


    x = np.zeros(3) # x为变形后的顶点坐标

    x[2] = p_num

    # 获取顶点及顶点左右两点的坐标
    coor = np.zeros((3, 2))


    coor[1, 0] = ppp[p_num, 0]
    coor[1, 1] = ppp[p_num, 1]

    coor[0, 0] = ppp[p_num - 1, 0]
    coor[0, 1] = ppp[p_num - 1, 1]

    if p_num != ppp.shape[0] - 1:
        coor[2, 0] = ppp[p_num + 1, 0]
        coor[2, 1] = ppp[p_num + 1, 1]
    else:
        coor[2, 0] = ppp[0, 0]
        coor[2, 1] = ppp[0, 1]



    length1 = ((coor[0, 0] - coor[2, 0])**2 + (coor[0, 1] - coor[2, 1])**2)**0.5 # length1为点1、3连线的长度
    length2 = ((coor[0, 0] - coor[1, 0]) ** 2 + (coor[0, 1] - coor[1, 1]) ** 2) ** 0.5
    length3 = ((coor[1, 0] - coor[2, 0]) ** 2 + (coor[1, 1] - coor[2, 1]) ** 2) ** 0.5


    # 求出夹角度数：


    # 求出三角形各边斜率
    if coor[0, 0] == coor[1, 0]:
        k1 = None
        b1 = coor[0, 0]
    if coor[0, 0] != coor[1, 0]:
        k1 = (coor[1, 1] - coor[0, 1]) / (coor[1, 0] - coor[0, 0])
        b1 = coor[1, 1] - k1 * coor[1, 0]

    if coor[2, 0] == coor[1, 0]:
        k2 = None
        b2 = coor[1, 0]
    if coor[2, 0] != coor[1, 0]:
        k2 = (coor[1, 1] - coor[2, 1]) / (coor[1, 0] - coor[2, 0])
        b2 = coor[1, 1] - k2 * coor[1, 0]

    if coor[2, 0] == coor[0, 0]:
        k3 = None
        b3 = coor[0, 0]
    if coor[2, 0] != coor[0, 0]:
        k3 = (coor[0, 1] - coor[2, 1]) / (coor[0, 0] - coor[2, 0])
        b3 = coor[0, 1] - k3 * coor[0, 0]


    theta2 = 0 # 弧度制

    if direction == 'left':
        cos_theta2 = (length1**2 + length2**2 - length3**2) / (2 * length1 * length2)
        theta2 = math.acos(cos_theta2)

    if direction == 'right':
        cos_theta2 = (length1 ** 2 + length3 ** 2 - length2 ** 2) / (2 * length1 * length3)
        theta2 = math.acos(cos_theta2)


    theta3 = math.pi - degree * math.pi / 180 - theta2

    l = length1 * math.sin(theta3) / sin_theta


    if direction == 'left':

        if k1 is not None:
            if coor[0, 0] > coor[1, 0]:
                x[0] = coor[0, 0] - l / (1 + k1 ** 2) ** 0.5
            if coor[0, 0] < coor[1, 0]:
                x[0] = coor[0, 0] + l / (1 + k1 ** 2) ** 0.5

            if coor[0, 1] > coor[1, 1]:
                x[1] = coor[0, 1] - l * k1 / (1 + k1 ** 2) ** 0.5
            if coor[0, 1] < coor[1, 1]:
                x[1] = coor[0, 1] + l * k1 / (1 + k1 ** 2) ** 0.5

        else:
            x[0] = coor[1, 0]
            if coor[0, 1] > coor[1, 1]:
                x[1] = coor[0, 1] - l
            if coor[0, 1] < coor[1, 1]:
                x[1] = coor[0, 1] + l


    if direction == 'right':

        if k2 is not None:
            if coor[2, 0] > coor[1, 0]:
                x[0] = coor[2, 0] - l / (1 + k2**2)**0.5
            if coor[2, 0] < coor[1, 0]:
                x[0] = coor[2, 0] + l / (1 + k2 ** 2) ** 0.5

            if coor[2, 1] > coor[1, 1]:
                x[1] = coor[2, 1] - l * k2 / (1 + k2 ** 2) ** 0.5
            if coor[2, 1] < coor[1, 1]:
                x[1] = coor[2, 1] + l * k2 / (1 + k2 ** 2) ** 0.5

        else:
            x[0] = coor[1, 0]
            if coor[2, 1] > coor[1, 1]:
                x[1] = coor[2, 1] - l
            if coor[2, 1] < coor[1, 1]:
                x[1] = coor[2, 1] + l


    return x


def cut(po, num, dep1, dep2):

    # 初始化p1,p2,p3
    p1 = np.zeros(2)
    p2 = np.zeros(2)
    p3 = np.zeros(2)

    p2[0] = po[num, 0]
    p2[1] = po[num, 1]

    p1[0] = po[num - 1, 0]
    p1[1] = po[num - 1, 1]

    if num + 1 < po.shape[0]:
        p3[0] = po[num + 1, 0]
        p3[1] = po[num + 1, 1]
    if num + 1 == po.shape[0]:
        p3[0] = po[0, 0]
        p3[1] = po[0, 1]


    # 得出倒角后两点坐标以及两点连线的k,b
    cut_xy = np.zeros((2, 2))

    # p1 and p2
    if p1[0] == p2[0]:
        cut_xy[0, 0] = p2[0]
        if p1[1] > p2[1]:
            cut_xy[0, 1] = p2[1] + dep1
        if p1[1] < p2[1]:
            cut_xy[0, 1] = p2[1] - dep1

    if p1[0] != p2[0]:
        k_12 = (p2[1] - p1[1]) / (p2[0] - p1[0])

        if p1[0] < p2[0]:
            cut_xy[0, 0] = p2[0] - dep1 / (k_12**2 + 1)**0.5
            cut_xy[0, 1] = p2[1] - dep1 * k_12 / (k_12**2 + 1)**0.5
        if p1[0] > p2[0]:
            cut_xy[0, 0] = p2[0] + dep1 / (k_12 ** 2 + 1) ** 0.5
            cut_xy[0, 1] = p2[1] + dep1 * k_12 / (k_12 ** 2 + 1) ** 0.5


    # p2 and p3
    if p3[0] == p2[0]:
        cut_xy[1, 0] = p2[0]
        if p3[1] > p2[1]:
            cut_xy[1, 1] = p2[1] + dep2
        if p3[1] < p2[1]:
            cut_xy[1, 1] = p2[1] - dep2

    if p3[0] != p2[0]:
        k_32 = (p2[1] - p3[1]) / (p2[0] - p3[0])

        if p3[0] < p2[0]:
            cut_xy[1, 0] = p2[0] - dep2 / (k_32 ** 2 + 1) ** 0.5
            cut_xy[1, 1] = p2[1] - dep2 * k_32 / (k_32 ** 2 + 1) ** 0.5
        if p3[0] > p2[0]:
            cut_xy[1, 0] = p2[0] + dep2 / (k_32 ** 2 + 1) ** 0.5
            cut_xy[1, 1] = p2[1] + dep2 * k_32 / (k_32 ** 2 + 1) ** 0.5


    # 计算k,b
    #if cut_xy[1, 0] - cut_xy[0, 0] != 0:
        #cut_xy[2, 0] = (cut_xy[1, 1] - cut_xy[0, 1]) / (cut_xy[1, 0] - cut_xy[0, 0])
        #cut_xy[2, 1] = cut_xy[1, 1] - cut_xy[1, 0] * cut_xy[2, 0]
    #if cut_xy[1, 0] - cut_xy[0, 0] == 0:
        #cut_xy[2, 0] = 1e8
        #cut_xy[2, 1] = cut_xy[1, 0] # 此时b不是在y轴的截距，而是在x轴上的截距


    return cut_xy


def circle_center(po, num_p, r):

    p1 = np.zeros(2)
    p2 = np.zeros(2)
    p3 = np.zeros(2)

    p2[0] = po[num_p, 0]
    p2[1] = po[num_p, 1]

    p1[0] = po[num_p - 1, 0]
    p1[1] = po[num_p - 1, 1]

    if num_p + 1 < po.shape[0]:
        p3[0] = po[num_p + 1, 0]
        p3[1] = po[num_p + 1, 1]
    if num_p + 1 == po.shape[0]:
        p3[0] = po[0, 0]
        p3[1] = po[0, 1]

    c_xy = np.zeros((3, 2))

    p_x = p2[0]
    p_y = p2[1]

    # 有一线段平行于y轴
    if p3[0] - p2[0] == 0 or p1[0] - p2[0] == 0:
        xc = yc = x_ver1 = y_ver1 = x_ver2 = y_ver2 = 0

        if p3[0] - p2[0] == 0:
            if p1[1] != p2[1]:

                k1 = (p1[1] - p2[1]) / (p1[0] - p2[0])

                tan_h = ((1 + k1 ** 2) ** 0.5 - 1) / (-k1)
                tan_f = (1 - tan_h) / (1 + tan_h)

                if p3[1] > p2[1]:
                    xc = p2[0] - r
                    yc = p2[1] + r / tan_f
                    x_ver1 = p2[0] - r / tan_f / (1 + k1 ** 2) ** 0.5
                    y_ver1 = p2[1] - r * k1 / tan_f / (1 + k1 ** 2) ** 0.5
                    x_ver2 = p2[0]
                    y_ver2 = yc

                if p3[1] < p2[1]:
                    xc = p2[0] + r
                    yc = p2[1] - r / tan_f
                    x_ver1 = p2[0] + r / tan_f / (1 + k1 ** 2) ** 0.5
                    y_ver1 = p2[1] + r * k1 / tan_f / (1 + k1 ** 2) ** 0.5
                    x_ver2 = p2[0]
                    y_ver2 = yc

            # p1p2连线平行于x轴，p2p3连线平行于y轴
            if p1[1] == p2[1]:
                if p1[0] < p2[0]:
                    xc = p2[0] - r
                    yc = p2[1] + r
                    x_ver1 = xc
                    y_ver1 = p2[1]
                    x_ver2 = p2[0]
                    y_ver2 = yc

                if p1[0] > p2[0]:
                    xc = p2[0] + r
                    yc = p2[1] - r
                    x_ver1 = xc
                    y_ver1 = p2[1]
                    x_ver2 = p2[0]
                    y_ver2 = yc

        if p1[0] - p2[0] == 0:
            if p2[1] != p3[1]:

                k2 = (p3[1] - p2[1]) / (p3[0] - p2[0])

                tan_h = ((1 + k2 ** 2) ** 0.5 - 1) / k2
                tan_f = (1 - tan_h) / (1 + tan_h)

                if p2[1] > p1[1]:
                    xc = p2[0] - r

                    yc = p2[1] - r / tan_f

                    x_ver2 = p2[0] - r / tan_f / (1 + k2 ** 2) ** 0.5
                    y_ver2 = p2[1] - r * k2 / tan_f / (1 + k2 ** 2) ** 0.5
                    x_ver1 = p2[0]
                    y_ver1 = yc

                if p2[1] < p1[1]:
                    xc = p2[0] + r

                    yc = r / tan_f + p2[1]

                    x_ver2 = p2[0] + r / tan_f / (1 + k2**2)**0.5
                    y_ver2 = p2[1] + r * k2 / tan_f / (1 + k2**2)**0.5
                    x_ver1 = p2[0]
                    y_ver1 = yc

            # p1p2连线平行于y轴，p2p3连线平行于x轴
            if p2[1] == p3[1]:
                if p3[0] < p2[0]:
                    xc = p2[0] - r
                    yc = p2[1] - r
                    x_ver1 = p2[0]
                    y_ver1 = yc
                    x_ver2 = xc
                    y_ver2 = p2[1]

                if p3[0] > p2[0]:
                    xc = p2[0] + r
                    yc = p2[1] + r
                    x_ver1 = p2[0]
                    y_ver1 = yc
                    x_ver2 = xc
                    y_ver2 = p2[1]

        c_xy[0, 0] = xc
        c_xy[0, 1] = yc
        c_xy[1, 0] = x_ver1
        c_xy[1, 1] = y_ver1
        c_xy[2, 0] = x_ver2
        c_xy[2, 1] = y_ver2
        return c_xy

    # 没有线段平行于y轴：
    if p3[0] - p2[0] != 0 and p1[0] - p2[0] != 0:

        k1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
        k2 = (p3[1] - p2[1]) / (p3[0] - p2[0])

        b_1 = p_y - k1 * p_x
        b_2 = p_y - k2 * p_x

        # k1 + k2 = 0:
        if (k1 + k2) == 0:

            xc = yc = x_ver1 = y_ver1 = x_ver2 = y_ver2 = 0

            # 两线段关于x轴对称
            if (p1[0] - p2[0]) * (p3[0] - p2[0]) > 0:
                yc = p2[1]

                if p2[0] < p1[0] and p2[0] < p3[0]:
                    xc = p2[0] + r * (1 + k1**2)**0.5 / abs(k1)

                    y_ver1 = yc + r / (1 + k1**2)**0.5
                    y_ver2 = yc - r / (1 + k1**2)**0.5
                    x_ver1 = p2[0] + r / (1 + k1**2)**0.5 / abs(k1)
                    x_ver2 = x_ver1

                if p2[0] > p1[0] and p2[0] > p3[0]:
                    xc = p2[0] - r * (1 + k1 ** 2) ** 0.5 / abs(k1)

                    y_ver1 = yc - r / (1 + k1 ** 2) ** 0.5
                    y_ver2 = yc + r / (1 + k1 ** 2) ** 0.5
                    x_ver1 = p2[0] - r / (1 + k1 ** 2) ** 0.5 / abs(k1)
                    x_ver2 = x_ver1

            # 两线段关于y轴对称
            if (p1[1] - p2[1]) * (p3[1] - p2[1]) > 0:
                xc = p2[0]

                if p2[1] < p1[1] and p2[1] < p3[1]:
                    yc = p2[1] + (1 + k1**2)**0.5 * r

                    x_ver1 = xc - r * abs(k1) / (1 + k1**2)**0.5
                    x_ver2 = xc + r * abs(k1) / (1 + k1**2)**0.5
                    y_ver1 = p2[1] + r * k1**2 / (1 + k1**2)**0.5
                    y_ver2 = x_ver1

                if p2[1] > p1[1] and p2[1] > p3[1]:
                    yc = p2[0] - (1 + k1**2)**0.5 * r

                    x_ver1 = xc + r * abs(k1) / (1 + k1 ** 2) ** 0.5
                    x_ver2 = xc - r * abs(k1) / (1 + k1 ** 2) ** 0.5
                    y_ver1 = p2[1] - r * k1 ** 2 / (1 + k1 ** 2) ** 0.5
                    y_ver2 = x_ver1

            c_xy[0, 0] = xc
            c_xy[0, 1] = yc
            c_xy[1, 0] = x_ver1
            c_xy[1, 1] = y_ver1
            c_xy[2, 0] = x_ver2
            c_xy[2, 1] = y_ver2
            return c_xy

        # k1 + k2 != 0:
        if (k1 + k2) != 0:

            up = k1 * (k2**2 + 1) * (k1**2 + 1)**0.5 + k2 * (k1**2 + 1) * (k2**2 + 1)**0.5
            down = (k2**2 + 1) * (k1**2 + 1)**0.5 + (k1**2 + 1) * (k2**2 + 1)**0.5

            # 角平分线与p1,p3连线的交点横、纵坐标、角平分线表达式的斜率及截距数组
            xs = np.zeros((2, 4))

            # 角平分线的斜率
            k3 = up / down
            b_3 = p_y - k3 * p_x

            k4 = - 1 / k3
            b_4 = p_y - k4 * p_x

            xs1 = 0
            xs2 = 0
            ys1 = ys2 = 0

            # ks不存在时：
            if p1[0] == p3[0]:
                xs1 = xs2 = p1[0]
                ys1 = k3 * xs1 + b_3
                ys2 = k4 * xs2 + b_4

            # ks存在时：
            if p1[0] != p3[0]:
                ks = (p1[1] - p3[1]) / (p1[0] - p3[0])
                bs = p1[1] - ks * p1[0]

                if ks != k3:
                    xs1 = (b_3 - bs) / (ks - k3)
                    ys1 = k3 * xs1 + b_3
                if ks == k3:
                    xs1 = (b_4 - bs) / (ks - k4)
                    ys1 = k3 * xs1 + b_3
                    k3 = k4
                    b_3 = b_4

                if ks != k4:
                    xs2 = (b_4 - bs) / (ks - k4)
                    ys2 = k4 * xs2 + b_4
                if ks == k4:
                    xs2 = (b_3 - bs) / (ks - k3)
                    ys2 = k4 * xs2 + b_4
                    k4 = k3
                    b_4 = b_3

            xs[0, 0] = xs1
            xs[0, 1] = k3
            xs[0, 2] = b_3
            xs[0, 3] = ys1

            xs[1, 0] = xs2
            xs[1, 1] = k4
            xs[1, 2] = b_4
            xs[1, 3] = ys2

        # 如果交点横坐标在p1,p3之间(即角平分线斜率正确)
            for q in range(2):

                if (xs[q, 0] - p1[0]) * (xs[q, 0] - p3[0]) <= 0 and (xs[q, 3] - p1[1]) * (xs[q, 3] - p3[1]) <= 0:

                    # 圆心的坐标数组
                    center_l = np.zeros((2, 2))
                    center_l[0, 0] = (r * (1 + k1**2)**0.5 + b_3 - b_1) / (k1 - xs[q, 1])
                    center_l[0, 1] = xs[q, 1] * center_l[0, 0] + xs[q, 2]
                    center_l[1, 0] = (r * (1 + k1**2)**0.5 + b_1 - b_3) / (xs[q, 1] - k1)
                    center_l[1, 1] = xs[q, 1] * center_l[1, 0] + xs[q, 2]

                    # 有一条线段平行于x轴
                    if k1 == 0 or k2 == 0:

                        if k1 == 0:
                            for z in range(2):
                                x_ver1 = center_l[z, 0]
                                y_ver1 = p_y

                                k_ver2 = - 1 / k2
                                b_ver2 = center_l[z, 1] - k_ver2 * center_l[z, 0]

                                x_ver2 = (b_ver2 - b_2) / (k2 - k_ver2)
                                y_ver2 = k2 * x_ver2 + b_2

                                j1 = (x_ver1 - p1[0]) * (x_ver1 - p_x)
                                j2 = (x_ver2 - p3[0]) * (x_ver2 - p_x)

                                if j1 < 0 and j2 < 0:
                                    c_xy[0, 0] = center_l[z, 0]
                                    c_xy[0, 1] = center_l[z, 1]
                                    c_xy[1, 0] = x_ver1
                                    c_xy[1, 1] = y_ver1
                                    c_xy[2, 0] = x_ver2
                                    c_xy[2, 1] = y_ver2
                                    return c_xy

                        if k2 == 0:
                            for z in range(2):
                                x_ver2 = center_l[z, 0]
                                y_ver2 = p_y

                                k_ver1 = - 1 / k1
                                b_ver1 = center_l[z, 1] - k_ver1 * center_l[z, 0]

                                x_ver1 = (b_ver1 - b_1) / (k1 - k_ver1)
                                y_ver1 = k1 * x_ver1 + b_1

                                j1 = (x_ver1 - p1[0]) * (x_ver1 - p_x)
                                j2 = (x_ver2 - p3[0]) * (x_ver2 - p_x)

                                if j1 < 0 and j2 < 0:
                                    c_xy[0, 0] = center_l[z, 0]
                                    c_xy[0, 1] = center_l[z, 1]
                                    c_xy[1, 0] = x_ver1
                                    c_xy[1, 1] = y_ver1
                                    c_xy[2, 0] = x_ver2
                                    c_xy[2, 1] = y_ver2
                                    return c_xy

                    # 没有线段平行于x轴
                    if k1 != 0 and k2 != 0:

                        for z in range(2):

                            # 由圆心向两边做垂线的斜率
                            k_ver1 = - 1 / k1
                            k_ver2 = - 1 / k2
                            # b
                            b_ver1 = center_l[z, 1] - k_ver1 * center_l[z, 0]
                            b_ver2 = center_l[z, 1] - k_ver2 * center_l[z, 0]
                            # 垂足的坐标
                            x_ver1 = (b_ver1 - b_1) / (k1 - k_ver1)
                            y_ver1 = k1 * x_ver1 + b_1

                            x_ver2 = (b_ver2 - b_2) / (k2 - k_ver2)
                            y_ver2 = k2 * x_ver2 + b_2

                            # 判断垂足是否在线段上（未考虑r过大的情况）
                            j1 = (x_ver1 - p1[0]) * (x_ver1 - p_x)
                            j2 = (x_ver2 - p3[0]) * (x_ver2 - p_x)

                            # 两垂足都在线段上：
                            if j1 < 0 and j2 < 0:
                                c_xy[0, 0] = center_l[z, 0]
                                c_xy[0, 1] = center_l[z, 1]
                                c_xy[1, 0] = x_ver1
                                c_xy[1, 1] = y_ver1
                                c_xy[2, 0] = x_ver2
                                c_xy[2, 1] = y_ver2
                                return c_xy



class Ui_MainWindow(object):
    # 类变量
    time_c = 0
    s = []
    t = []

    time_d = 0
    u = []
    v = []
    w = []

    timeOFsave = 0

    def __init__(self):

        self.sender = send() # 实例化send进程
        self.sender.sig.connect(self.slot) # 绑定signal信号和slot槽函数


        # self.picture1_2 = MyMouseDoubleClicked()
        # self.picture1_2.my_signal.connect(self.showbiggerphoto)
        #
        # self.picture2_2 = MyMouseDoubleClicked()
        # self.picture2_2.my_signal.connect(self.showbiggerphoto)


        self.return1 = ReturnPressed()
        #self.return1.sentence.connect(self.jump)




    def showbiggerphoto(self, name):
        name0 = name[0:8]

        if name2 != 0:
            filePath = ''.join([name2, '/', name0, '.png'])

            if os.path.exists(filePath) is True:
                img = Image.open(filePath)
                img.show()


    def slot(self, num):

        self.progressBar.setProperty("value", num)

        if num == 25:
            gmsh.finalize() # 结束gmsh线程

        if num == 100:
            self.sender.terminate() # 结束sender线程


            operate_time = ''.join(["计算共耗时", str(n_time - b_time), "s"])
            self.textEdit_2.append(operate_time)
            self.textEdit_2.append(' ')


            # 绘制云图
            l2 = [name2, '/', 'picture2.png']
            location2 = ''.join(l2)


            xt = np.loadtxt(''.join([name2, '/', 'x.txt']))
            yt = np.loadtxt(''.join([name2, '/', 'y.txt']))
            ut = np.loadtxt(''.join([name2, '/', 'temperature.txt']))


            cloud_chart(xt, yt, ut)
            plt.savefig(location2, dpi=1000, bbox_inches='tight')
            plt.close()


            pixmap2 = QPixmap(location2)
            self.picture2_2.setPixmap(pixmap2)
            self.picture2_2.setScaledContents(True)


            self.progressBar.hide()
            self.textEdit_2.append("temperature distribution image is created successfully.\n")


    def ooo(self):
        try:
            self.progressBar.show()

            # 将材料性质，倒角圆角数据传入全局变量
            global t0, time_hc
            t0 = float(self.c0.text())
            time_hc = float(self.diftime.text())




            global sss, ttt, uuu, vvv, www, time_c1, time_d1
            sss = Ui_MainWindow.s
            ttt = Ui_MainWindow.t
            time_c1 = Ui_MainWindow.time_c

            uuu = Ui_MainWindow.u
            vvv = Ui_MainWindow.v
            www = Ui_MainWindow.w
            time_d1 = Ui_MainWindow.time_d


            self.sender.start() # 开始进程
            #self.threadpool.start(sender)
        except:
            pass

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1130)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(width, height))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background:rgb(0, 85, 127);color:white;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background:blue")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.horizontalLayout_6.addLayout(self.verticalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(19, 529, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_4.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_3.sizePolicy().hasHeightForWidth())
        self.lineEdit_3.setSizePolicy(sizePolicy)
        self.lineEdit_3.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_4.addWidget(self.lineEdit_3, 3, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background:blue")
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_4.addWidget(self.pushButton_2, 4, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.gridLayout_4.addWidget(self.label_17, 3, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout_4.addWidget(self.label_15, 1, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout_4.addWidget(self.label_16, 2, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_4.addWidget(self.label_14, 0, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy)
        self.lineEdit_2.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_4.addWidget(self.lineEdit_2, 2, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 4, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.add1 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add1.sizePolicy().hasHeightForWidth())
        self.add1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.add1.setFont(font)
        self.add1.setStyleSheet("background:blue")
        self.add1.setObjectName("add1")
        self.gridLayout_3.addWidget(self.add1, 3, 1, 1, 1)
        self.r_2 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.r_2.sizePolicy().hasHeightForWidth())
        self.r_2.setSizePolicy(sizePolicy)
        self.r_2.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.r_2.setObjectName("r_2")
        self.gridLayout_3.addWidget(self.r_2, 2, 1, 1, 1)
        self.num1_2 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.num1_2.sizePolicy().hasHeightForWidth())
        self.num1_2.setSizePolicy(sizePolicy)
        self.num1_2.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.num1_2.setObjectName("num1_2")
        self.gridLayout_3.addWidget(self.num1_2, 1, 1, 1, 1)
        self.r = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.r.setFont(font)
        self.r.setObjectName("r")
        self.gridLayout_3.addWidget(self.r, 2, 0, 1, 1)
        self.num1 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.num1.setFont(font)
        self.num1.setObjectName("num1")
        self.gridLayout_3.addWidget(self.num1, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.depth22 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.depth22.sizePolicy().hasHeightForWidth())
        self.depth22.setSizePolicy(sizePolicy)
        self.depth22.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.depth22.setObjectName("depth22")
        self.gridLayout_2.addWidget(self.depth22, 4, 1, 1, 1)
        self.depth12 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.depth12.sizePolicy().hasHeightForWidth())
        self.depth12.setSizePolicy(sizePolicy)
        self.depth12.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.depth12.setObjectName("depth12")
        self.gridLayout_2.addWidget(self.depth12, 3, 1, 1, 1)
        self.num2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.num2.setFont(font)
        self.num2.setObjectName("num2")
        self.gridLayout_2.addWidget(self.num2, 2, 0, 1, 1)
        self.num2_2 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.num2_2.sizePolicy().hasHeightForWidth())
        self.num2_2.setSizePolicy(sizePolicy)
        self.num2_2.setStyleSheet("background:rgb(254, 255, 242);color:black;")
        self.num2_2.setObjectName("num2_2")
        self.gridLayout_2.addWidget(self.num2_2, 2, 1, 1, 1)
        self.depth2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.depth2.setFont(font)
        self.depth2.setObjectName("depth2")
        self.gridLayout_2.addWidget(self.depth2, 4, 0, 1, 1)
        self.add2 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add2.sizePolicy().hasHeightForWidth())
        self.add2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.add2.setFont(font)
        self.add2.setStyleSheet("background:blue")
        self.add2.setObjectName("add2")
        self.gridLayout_2.addWidget(self.add2, 5, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)
        self.depth1 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.depth1.setFont(font)
        self.depth1.setObjectName("depth1")
        self.gridLayout_2.addWidget(self.depth1, 3, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_2, 2, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem2, 3, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem3, 1, 0, 1, 1)
        self.horizontalLayout_6.addLayout(self.gridLayout_5)
        spacerItem4 = QtWidgets.QSpacerItem(18, 529, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem4)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.gridLayout_6.addWidget(self.label_19, 2, 3, 1, 1)
        self.cf = QtWidgets.QLineEdit(self.centralwidget)
        self.cf.setStyleSheet("background:rgb(254, 255, 242);\n"
                              "color:rgb(0, 0, 0);")
        self.cf.setObjectName("cf")
        self.gridLayout_6.addWidget(self.cf, 4, 1, 1, 2)
        self.diftime = QtWidgets.QLineEdit(self.centralwidget)
        self.diftime.setStyleSheet("background:rgb(254, 255, 242);\n"
                                   "color:rgb(0, 0, 0);")
        self.diftime.setObjectName("diftime")
        self.gridLayout_6.addWidget(self.diftime, 1, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_6.addWidget(self.label_3, 2, 0, 1, 1)
        self.toolButton_2 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_2.setObjectName("toolButton_2")
        self.gridLayout_6.addWidget(self.toolButton_2, 3, 4, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.gridLayout_6.addWidget(self.label_21, 4, 3, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_6.addWidget(self.label_20, 3, 3, 1, 1)
        self.c0 = QtWidgets.QLineEdit(self.centralwidget)
        self.c0.setStyleSheet("background:rgb(254, 255, 242);\n"
                              "color:rgb(0, 0, 0);")
        self.c0.setObjectName("c0")
        self.gridLayout_6.addWidget(self.c0, 0, 1, 1, 2)
        self.beta = QtWidgets.QLineEdit(self.centralwidget)
        self.beta.setStyleSheet("background:rgb(254, 255, 242);\n"
                                "color:rgb(0, 0, 0);")
        self.beta.setObjectName("beta")
        self.gridLayout_6.addWidget(self.beta, 3, 1, 1, 2)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_6.addWidget(self.label_13, 1, 3, 1, 1)
        self.toolButton_3 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_3.setObjectName("toolButton_3")
        self.gridLayout_6.addWidget(self.toolButton_3, 4, 4, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 1, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_6.addWidget(self.label_11, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_6.addWidget(self.label_7, 4, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_6.addWidget(self.label_6, 3, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 0, 3, 1, 1)
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setObjectName("toolButton")
        self.gridLayout_6.addWidget(self.toolButton, 2, 4, 1, 1)
        self.Dc = QtWidgets.QLineEdit(self.centralwidget)
        self.Dc.setStyleSheet("background:rgb(254, 255, 242);\n"
                              "color:rgb(0, 0, 0);")
        self.Dc.setObjectName("Dc")
        self.gridLayout_6.addWidget(self.Dc, 2, 1, 1, 2)
        self.verticalLayout_4.addLayout(self.gridLayout_6)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem5)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.reset = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.reset.sizePolicy().hasHeightForWidth())
        self.reset.setSizePolicy(sizePolicy)
        self.reset.setMinimumSize(QtCore.QSize(100, 66))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.reset.setFont(font)
        self.reset.setStyleSheet("background:rgb(255, 0, 0);\n"
                                 "")
        self.reset.setObjectName("reset")
        self.horizontalLayout.addWidget(self.reset)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_4.addItem(spacerItem8)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        self.run = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.run.sizePolicy().hasHeightForWidth())
        self.run.setSizePolicy(sizePolicy)
        self.run.setMinimumSize(QtCore.QSize(100, 66))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.run.setFont(font)
        self.run.setStyleSheet("background:rgb(0, 170, 0)")
        self.run.setObjectName("run")
        self.horizontalLayout_2.addWidget(self.run)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem10)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem11)
        self.horizontalLayout_6.addLayout(self.verticalLayout_4)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem12)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_6)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.original_pic = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original_pic.sizePolicy().hasHeightForWidth())
        self.original_pic.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.original_pic.setFont(font)
        self.original_pic.setObjectName("original_pic")
        self.verticalLayout_5.addWidget(self.original_pic)
        # self.picture1_2 = QtWidgets.QLabel(self.centralwidget)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.picture1_2.sizePolicy().hasHeightForWidth())
        # self.picture1_2.setSizePolicy(sizePolicy)
        # self.picture1_2.setMinimumSize(QtCore.QSize(pic_size, pic_size))
        # self.picture1_2.setMaximumSize(QtCore.QSize(pic_size, pic_size))
        # self.picture1_2.setToolTip("")
        # self.picture1_2.setStyleSheet("background:rgb(254, 255, 242)")
        # self.picture1_2.setFrameShape(QtWidgets.QFrame.Box)
        # self.picture1_2.setText("")
        # self.picture1_2.setObjectName("picture1_2")
        # self.verticalLayout_5.addWidget(self.picture1_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_5)
        self.verticalLayout_6.addLayout(self.horizontalLayout_4)
        spacerItem13 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem13)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_7.addWidget(self.label_4, 0, 0, 1, 1)
        self.cloud_chart = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cloud_chart.sizePolicy().hasHeightForWidth())
        self.cloud_chart.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.cloud_chart.setFont(font)
        self.cloud_chart.setObjectName("cloud_chart")
        self.gridLayout_7.addWidget(self.cloud_chart, 0, 2, 1, 1)
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit_2.sizePolicy().hasHeightForWidth())
        self.textEdit_2.setSizePolicy(sizePolicy)
        self.textEdit_2.setMinimumSize(QtCore.QSize(0, 450))
        self.textEdit_2.setMaximumSize(QtCore.QSize(16777215, pic_size))
        self.textEdit_2.setObjectName("textEdit_2")
        self.gridLayout_7.addWidget(self.textEdit_2, 1, 0, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(150, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem14, 1, 1, 1, 1)
        self.picture2_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picture2_2.sizePolicy().hasHeightForWidth())
        self.picture2_2.setSizePolicy(sizePolicy)
        self.picture2_2.setMinimumSize(QtCore.QSize(pic_size, pic_size))
        self.picture2_2.setMaximumSize(QtCore.QSize(pic_size, pic_size))
        self.picture2_2.setToolTip("")
        self.picture2_2.setStyleSheet("background:rgb(254, 255, 242)")
        self.picture2_2.setFrameShape(QtWidgets.QFrame.Box)
        self.picture2_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.picture2_2.setLineWidth(1)
        self.picture2_2.setText("")
        self.picture2_2.setObjectName("picture2_2")
        self.gridLayout_7.addWidget(self.picture2_2, 1, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMinimumSize(QtCore.QSize(450, 0))
        self.progressBar.setMaximumSize(QtCore.QSize(450, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.progressBar.setFont(font)
        self.progressBar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.progressBar.setStyleSheet("QProgressBar{text-align: center;}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_5.addWidget(self.progressBar)
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem15)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_7.addLayout(self.verticalLayout_2)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem16)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_R = QtWidgets.QMenu(self.menubar)
        self.menu_R.setObjectName("menu_R")
        self.menu_T = QtWidgets.QMenu(self.menubar)
        self.menu_T.setObjectName("menu_T")
        self.menu_4 = QtWidgets.QMenu(self.menu_T)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menu_T)
        self.menu_5.setObjectName("menu_5")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionnew = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionnew.setFont(font)
        self.actionnew.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionnew.setObjectName("actionnew")
        self.actionopen = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionopen.setFont(font)
        self.actionopen.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionopen.setObjectName("actionopen")
        self.actionguidence = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionguidence.setFont(font)
        self.actionguidence.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionguidence.setObjectName("actionguidence")
        self.actioncontact_support = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actioncontact_support.setFont(font)
        self.actioncontact_support.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actioncontact_support.setObjectName("actioncontact_support")
        self.actionSubmit_Feedback = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionSubmit_Feedback.setFont(font)
        self.actionSubmit_Feedback.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionSubmit_Feedback.setObjectName("actionSubmit_Feedback")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionAbout.setFont(font)
        self.actionAbout.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionAbout.setObjectName("actionAbout")
        self.actionsave = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionsave.setFont(font)
        self.actionsave.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionsave.setObjectName("actionsave")
        self.actionx = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.actionx.setFont(font)
        self.actionx.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionx.setObjectName("actionx")
        self.action_R = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.action_R.setFont(font)
        self.action_R.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.action_R.setObjectName("action_R")
        # self.actionp1 = QtWidgets.QAction(MainWindow)
        # self.actionp1.setShortcutContext(QtCore.Qt.WindowShortcut)
        # self.actionp1.setObjectName("actionp1")
        self.actionp2 = QtWidgets.QAction(MainWindow)
        self.actionp2.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionp2.setObjectName("actionp2")
        self.actionCritical_Path_2 = QtWidgets.QAction(MainWindow)
        self.actionCritical_Path_2.setObjectName("actionCritical_Path_2")
        self.actionsetting = QtWidgets.QAction(MainWindow)
        self.actionsetting.setObjectName("actionsetting")
        self.actionpoint = QtWidgets.QAction(MainWindow)
        self.actionpoint.setObjectName("actionpoint")
        self.actionline = QtWidgets.QAction(MainWindow)
        self.actionline.setObjectName("actionline")
        self.menu.addAction(self.actionnew)
        self.menu.addAction(self.actionopen)
        self.menu.addAction(self.actionsave)
        self.menu.addAction(self.actionx)
        self.menu_2.addAction(self.actionguidence)
        self.menu_2.addAction(self.actioncontact_support)
        self.menu_2.addAction(self.actionSubmit_Feedback)
        self.menu_2.addAction(self.actionAbout)
        self.menu_R.addAction(self.action_R)
        # self.menu_4.addAction(self.actionp1)
        self.menu_4.addAction(self.actionp2)
        self.menu_5.addAction(self.actionpoint)
        self.menu_5.addAction(self.actionline)
        self.menu_T.addAction(self.menu_4.menuAction())
        self.menu_T.addAction(self.menu_5.menuAction())
        self.menu_3.addAction(self.actionsetting)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_R.menuAction())
        self.menubar.addAction(self.menu_T.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())


        self.figure = plt.figure()
        self.picture1_2 = FigureCanvas(self.figure)
        self.picture1_2.setMinimumSize(QtCore.QSize(pic_size, pic_size))
        self.picture1_2.setMaximumSize(QtCore.QSize(pic_size, pic_size))
        self.picture1_2.setStyleSheet("background:rgb(254, 255, 242)")
        self.verticalLayout_5.addWidget(self.picture1_2)

        self.ax = self.figure.add_subplot(111)



        self.retranslateUi(MainWindow)
        # self.pushButton.clicked.connect(self.picture1_2.clear)
        self.pushButton.clicked.connect(self.show1) # type: ignore


        self.add1.clicked.connect(self.add_circle)
        self.add1.clicked.connect(self.num1_2.clear) # type: ignore
        self.add1.clicked.connect(self.r_2.clear) # type: ignore
        self.add1.clicked.connect(self.show1)


        self.add2.clicked.connect(self.add_cut)
        self.add2.clicked.connect(self.num2_2.clear) # type: ignore
        self.add2.clicked.connect(self.depth12.clear) # type: ignore
        self.add2.clicked.connect(self.depth22.clear) # type: ignore
        self.add2.clicked.connect(self.show1)


        self.pushButton_2.clicked.connect(self.ChangeAngle)
        self.pushButton_2.clicked.connect(self.lineEdit.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_2.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_3.clear)
        self.pushButton_2.clicked.connect(self.show1)


        self.run.clicked.connect(self.ooo)  # type: ignore
        self.reset.clicked.connect(self.delete)
        self.reset.clicked.connect(self.show1)


        self.toolButton.clicked.connect(self.form_1)
        self.toolButton_2.clicked.connect(self.form_2)
        self.toolButton_3.clicked.connect(self.form_3)


        self.num1_2.returnPressed.connect(self.r_2.setFocus)  # type: ignore
        self.num2_2.returnPressed.connect(self.depth12.setFocus)  # type: ignore
        self.depth12.returnPressed.connect(self.depth22.setFocus)  # type: ignore
        self.r_2.returnPressed.connect(self.add1.click)  # type: ignore
        self.depth22.returnPressed.connect(self.add2.click)  # type: ignore

        self.lineEdit.returnPressed.connect(self.lineEdit_2.setFocus)  # type: ignore
        self.lineEdit_2.returnPressed.connect(self.lineEdit_3.setFocus)  # type: ignore
        self.lineEdit_3.returnPressed.connect(self.pushButton_2.click)  # type: ignore


        self.actionnew.triggered.connect(self.new)
        self.actionsave.triggered.connect(self.save)
        self.actionopen.triggered.connect(self.open)
        self.actionx.triggered.connect(self.exit)

        self.action_R.triggered.connect(self.ooo)

        # self.actionp1.triggered.connect(self.show_p1)
        self.actionp2.triggered.connect(self.show_p2)

        self.actionpoint.triggered.connect(self.Hchart)
        self.actionline.triggered.connect(self.Lchart)

        self.actionsetting.triggered.connect(self.setting)

        self.actionguidence.triggered.connect(self.guide)
        self.actioncontact_support.triggered.connect(self.support)
        self.actionSubmit_Feedback.triggered.connect(self.feedback)
        self.actionAbout.triggered.connect(self.about)


        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def delete(self):
        Ui_MainWindow.time_c = 0
        Ui_MainWindow.s = []
        Ui_MainWindow.t = []

        Ui_MainWindow.time_d = 0
        Ui_MainWindow.u = []
        Ui_MainWindow.v = []
        Ui_MainWindow.w = []

        global xxx, yyy, zzz, time_ch
        xxx = []
        yyy = []
        zzz = []
        time_ch = 0


    def save(self):
        Ui_MainWindow.timeOFsave += 1
        if Ui_MainWindow.timeOFsave == 1:
            if name2 != 0:
                coordinate = ''.join([name2, '/', 'coordinate.txt'])

                circle = ''.join([name2, '/', 'circle.txt'])
                cu = ''.join([name2, '/', 'cut.txt'])
                degree = ''.join([name2, '/', 'degree.txt'])

                Dc = ''.join([name2, '/', 'Dc.txt'])
                beta = ''.join([name2, '/', 'beta.txt'])
                cf = ''.join([name2, '/', 'cf.txt'])

                try:
                    with open(coordinate, "w") as f:
                        f.write(self.textEdit.toPlainText())
                except:
                    pass

                try:
                    with open(circle, "w") as f:
                        str_s = []
                        for i in range(Ui_MainWindow.time_c):
                            if i == Ui_MainWindow.time_c -1:
                                str_s.append(str(Ui_MainWindow.s[i]) + '\n')
                            else:
                                str_s.append(''.join([str(Ui_MainWindow.s[i]), ' ']))
                        f.writelines(str_s)

                        str_t = []
                        for i in range(Ui_MainWindow.time_c):
                            if i != Ui_MainWindow.time_c -1:
                                str_t.append(''.join([str(Ui_MainWindow.t[i]), ' ']))
                            else:
                                str_t.append(str(Ui_MainWindow.t[i]))
                        f.writelines(str_t)
                except:
                    pass

                try:
                    with open(cu, "w") as f:
                        str_u = []
                        for i in range(Ui_MainWindow.time_d):
                            if i == Ui_MainWindow.time_d - 1:
                                str_u.append(str(Ui_MainWindow.u[i]) + '\n')
                            else:
                                str_u.append(''.join([str(Ui_MainWindow.u[i]), ' ']))
                        f.writelines(str_u)

                        str_v = []
                        for i in range(Ui_MainWindow.time_d):
                            if i == Ui_MainWindow.time_d - 1:
                                str_v.append(str(Ui_MainWindow.v[i]) + '\n')
                            else:
                                str_v.append(''.join([str(Ui_MainWindow.v[i]), ' ']))
                        f.writelines(str_v)

                        str_w = []
                        for i in range(Ui_MainWindow.time_d):
                            if i != Ui_MainWindow.time_d - 1:
                                str_w.append(''.join([str(Ui_MainWindow.w[i]), ' ']))
                            else:
                                str_w.append(str(Ui_MainWindow.w[i]))
                        f.writelines(str_w)
                except:
                    pass

                try:
                    with open(degree, "w") as f:
                        str_x = []
                        for i in range(time_ch):
                            if i == time_ch - 1:
                                str_x.append(str(xxx[i]) + '\n')
                            else:
                                str_x.append(''.join([str(xxx[i]), ' ']))
                        f.writelines(str_x)

                        str_y = []
                        for i in range(time_ch):
                            if i == time_ch - 1:
                                str_y.append(yyy[i] + '\n')
                            else:
                                str_y.append(''.join([yyy[i], ' ']))
                        f.writelines(str_y)

                        str_z = []
                        for i in range(time_ch):
                            if i != time_ch - 1:
                                str_z.append(''.join([str(zzz[i]), ' ']))
                            else:
                                str_z.append(str(zzz[i]))
                        f.writelines(str_z)

                except:
                    pass

                try:
                    # 保存材料参数和扩散过程中的相关参数
                    np.savetxt(Dc, Dc_matrix)
                    np.savetxt(beta, beta_matrix)
                    np.savetxt(cf, cf_matrix)
                except:
                    pass
                # f.write(self.c0.text())
                # f.write(self.diftime.text())
            else:
                pass
        else:
            pass


    def form_1(self):
        ss = QtWidgets.QDialog()
        c = Ui_Form_1()
        c.setupUi(ss)
        ss.exec_()

        try:
            if len(Dc_matrix) != 0:
                self.Dc.setText("已保存")

        except:
            pass


    def form_2(self):

        ss = QtWidgets.QDialog()
        c = Ui_Form_2()
        c.setupUi(ss)
        ss.exec_()

        try:
            if len(beta_matrix) != 0:
                self.beta.setText("已保存")

        except:
            pass


    def form_3(self):
        ss = QtWidgets.QDialog()
        c = Ui_Form_3()
        c.setupUi(ss)
        # ss.mt.result.connect(c.save)
        ss.exec_()

        try:
            if len(cf_matrix) != 0:
                self.cf.setText("已保存")
        except:
            pass



    # def show_p1(self):
    #     if name2 != 0:
    #         filePath = ''.join([name2, '/', 'picture1.png'])
    #
    #         if os.path.exists(filePath) is True:
    #             img = Image.open(filePath)
    #             img.show()


    def show_p2(self):
        if name2 != 0:
            filePath = ''.join([name2, '/', 'picture2.png'])

            if os.path.exists(filePath) is True:
                img = Image.open(filePath)
                img.show()





    def guide(self):
        ss = QtWidgets.QDialog()
        c = Ui_support()
        c.setupUi(ss)
        ss.exec_()


    def support(self):
        ss = QtWidgets.QDialog()
        c = Ui_support()
        c.setupUi(ss)
        ss.exec_()


    def feedback(self):
        ss = QtWidgets.QDialog()
        c = Ui_feedback()
        c.setupUi(ss)
        ss.exec_()


    def about(self):
        ss = QtWidgets.QDialog()
        c = Ui_About()
        c.setupUi(ss)
        ss.exec_()


    def exit(self):
        sys.exit()


    # 新建文件夹函数
    def new(self):
        new = QtWidgets.QDialog()
        n = Ui_createnew()
        n.setupUi(new)
        new.exec_()

        if name3 != 0:
            self.textEdit_2.append(name3)


    def setting(self):
        sett = QtWidgets.QDialog()
        settt = Ui_set()
        settt.setupUi(sett)
        sett.exec_()


        if density != 50:
            self.textEdit_2.append("网格密度已修改为：" + str(density))
            self.textEdit_2.append(' ')

        if stepnum != 10:
            self.textEdit_2.append("时间步数已修改为：" + str(stepnum))
            self.textEdit_2.append(' ')


    # open函数
    def open(self):
        directory1 = QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径

        self.textEdit_2.append("已选取文件夹，文件夹名称为：")
        self.textEdit_2.append(directory1)
        self.textEdit_2.append("")

        global name2
        name2 = directory1


        coordinate = ''.join([name2, '/', 'coordinate.txt'])

        circle = ''.join([name2, '/', 'circle.txt'])
        cu = ''.join([name2, '/', 'cut.txt'])
        degree = ''.join([name2, '/', 'degree.txt'])

        Dc = ''.join([name2, '/', 'Dc.txt'])
        beta = ''.join([name2, '/', 'beta.txt'])
        cf = ''.join([name2, '/', 'cf.txt'])

        # 尝试读取文件夹数据并在界面显示
        try:
            with open(coordinate, "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                    self.textEdit.append(line)

            self.textEdit_2.append("已读取初始顶点数据......")
            self.textEdit_2.append("")
        except:
            pass

        # 读取角度数据
        try:
            list3 = []
            with open(degree, "r") as f:
                for line in f.readlines():
                    row = line.strip('\n')  # 去掉列表中每一个元素的换行符
                    self.textEdit_2.append(row)
                    list3.append(row)


            for i in range(3):
                data = list3[i].split(' ')
                print(data)

                global time_ch
                time_ch = len(data)

                global xxx, yyy, zzz

                if i == 0:
                    for j in range(time_ch):
                        xxx.append(int(data[j]))

                if i == 1:
                    for j in range(time_ch):
                        yyy.append(data[j])

                if i == 2:
                    for j in range(time_ch):
                        zzz.append(float(data[j]))
            self.textEdit_2.append("已读取角度数据......")
            self.textEdit_2.append("")
        except:
            pass



        # 读取倒角数据
        try:
            list2 = []
            with open(cu, "r") as f:
                for line in f.readlines():
                    row = line.strip('\n')  # 去掉列表中每一个元素的换行符
                    self.textEdit_2.append(row)
                    list2.append(row)

            for i in range(3):
                data = list2[i].split(' ')
                print(data)

                Ui_MainWindow.time_d = len(data)

                if i == 0:
                    for j in range(Ui_MainWindow.time_d):
                        Ui_MainWindow.u.append(int(data[j]))

                if i == 1:
                    for j in range(Ui_MainWindow.time_d):
                        Ui_MainWindow.v.append(float(data[j]))

                if i == 2:
                    for j in range(Ui_MainWindow.time_d):
                        Ui_MainWindow.w.append(float(data[j]))
            self.textEdit_2.append("已读取倒角数据......")
            self.textEdit_2.append("")
        except:
            pass

        # 读取圆角数据
        try:
            list1 = []
            with open(circle, "r") as f:
                for line in f.readlines():
                    row = line.strip('\n')  # 去掉列表中每一个元素的换行符
                    self.textEdit_2.append(row)
                    list1.append(row)


            for i in range(2):
                data = list1[i].split(' ')
                print(data)

                Ui_MainWindow.time_c = len(data)
                print(Ui_MainWindow.time_c)
                if i == 0:
                    for j in range(Ui_MainWindow.time_c):
                        Ui_MainWindow.s.append(int(data[j]))

                if i == 1:
                    for j in range(Ui_MainWindow.time_c):
                        Ui_MainWindow.t.append(float(data[j]))


            self.textEdit_2.append("已读取圆角数据......")
            self.textEdit_2.append("")

        except:
            pass


        self.show1()


        try:
            # 读取Dc, beta, cf数据
            global Dc_matrix
            Dc_matrix = np.loadtxt(Dc)
            global beta_matrix
            beta_matrix = np.loadtxt(beta)
            global cf_matrix
            cf_matrix = np.loadtxt(cf)
        except:
            pass


        #location = ''.join([directory1, '/', 'picture1.png'])
        #pixmap = QPixmap(location)
        #self.picture1_2.setPixmap(pixmap)
        #self.picture1_2.setScaledContents(True)


        # file_path, file_type = QFileDialog.getOpenFileName(None, 'Open File', os.getcwd())
        #
        # if file_path == '':
        #     pass
        # else:
        #     pass
            # if file_path[-3:] == 'png':
            #     self.textEdit_2.append(file_path)
            #     #self.textEdit_2.append(file_type)
            #     #img = Image.open(file_path)
            #     #img.show()
            #
            #     cc = QtWidgets.QDialog()
            #     child = Ui_OpenPicture()
            #     child.setupUi(cc)
            #     child.show(file_path)
            #     cc.exec_()
            #
            #
            #
            # if file_path[-3:] == 'txt':
            #     self.textEdit_2.append(file_path)
            #
            #     cc = QtWidgets.QDialog()
            #     child = Ui_points()
            #     child.setupUi(cc)
            #     child.show(file_path)
            #     cc.exec_()


    def Hchart(self):
        cc = QtWidgets.QDialog()
        child = Ui_HistoryChart()
        child.setupUi(cc)
        cc.exec_()


    def Lchart(self):
        cc = QtWidgets.QDialog()
        child = Ui_Line()
        child.setupUi(cc)
        cc.exec_()


    # 新建一个dialog小窗
    #def show_points(self):
        #cc = QtWidgets.QDialog()
        #child = Ui_points()
        #child.setupUi(cc)
        #child.show()
        #cc.exec_()


    # 重置函数
    def reset_all(self):
        Ui_MainWindow.s.clear()
        Ui_MainWindow.t.clear()
        Ui_MainWindow.u.clear()
        Ui_MainWindow.v.clear()
        Ui_MainWindow.w.clear()

        Ui_MainWindow.time_c = 0
        Ui_MainWindow.time_d = 0


        self.textEdit.clear()
        self.textEdit_2.clear()

        self.picture1_2.clear()
        self.picture2_2.clear()


    def add_circle(self):
        if len(self.num1_2.text()) == 0 or len(self.r_2.text()) == 0:
            pass
        else:

            Ui_MainWindow.s.append(int(self.num1_2.text()))
            Ui_MainWindow.t.append(float(self.r_2.text()))
            Ui_MainWindow.time_c += 1

            self.textEdit_2.append("圆角已添加！")
            self.textEdit_2.append("顶点编号为：")
            self.textEdit_2.append(self.num1_2.text())
            self.textEdit_2.append("圆弧半径为：")
            self.textEdit_2.append(self.r_2.text())
            self.textEdit_2.append("")


    def add_cut(self):
        if len(self.num2_2.text()) == 0 or len(self.depth12.text()) == 0 or len(self.depth22.text()) == 0:
            pass
        else:
            Ui_MainWindow.u.append(int(self.num2_2.text()))
            Ui_MainWindow.v.append(float(self.depth12.text()))
            Ui_MainWindow.w.append(float(self.depth22.text()))
            Ui_MainWindow.time_d += 1

            self.textEdit_2.append("倒角已添加！")
            self.textEdit_2.append("顶点编号为：")
            self.textEdit_2.append(self.num2_2.text())
            self.textEdit_2.append("左侧截取长度为：")
            self.textEdit_2.append(self.depth12.text())
            self.textEdit_2.append("右侧截取长度为：")
            self.textEdit_2.append(self.depth22.text())
            self.textEdit_2.append("")


    def ChangeAngle(self):
        if len(self.lineEdit.text()) == 0 or len(self.lineEdit_2.text()) == 0 or len(self.lineEdit_3.text()) == 0:
            pass
        else:

            p = int(self.lineEdit.text())
            l = self.lineEdit_2.text()
            a = float(self.lineEdit_3.text())


            global xxx, yyy, zzz, time_ch
            xxx.append(p)
            yyy.append(l)
            zzz.append(a)
            time_ch += 1

            self.textEdit_2.append("角度已修改！")
            self.textEdit_2.append("顶点编号为：")
            self.textEdit_2.append(self.lineEdit.text())

            self.textEdit_2.append(self.lineEdit_2.text())
            self.textEdit_2.append("角度大小为：")
            self.textEdit_2.append(self.lineEdit_3.text())
            self.textEdit_2.append("")


    # show1函数绘制出几何图像，包括经过圆角及倒角操作后的图像
    def show1(self):
        self.ax.cla()

        points = self.textEdit.toPlainText()
        global oc
        oc = points

        if len(points) == 0:
            pass
        else:

            list1 = points.split('\n')

            global ppp
            ppp = np.zeros((len(list1), 2))
            time = 0
            for i in list1:
                f = i.split(' ')
                ppp[time, 0] = float(f[0])
                ppp[time, 1] = float(f[1])
                time += 1


            # ChangedPoint = [] # 该列表储存angle函数返回的数组

            if time_ch != 0:
                for i in range(time_ch):
                    CP = angle(xxx[i], yyy[i], zzz[i])

                    ppp[int(CP[2]), 0] = CP[0]
                    ppp[int(CP[2]), 1] = CP[1]


            # for i in range(time_ch):
            #     CP = ChangedPoint[i]
            #     ppp[int(CP[2]), 0] = CP[0]
            #     ppp[int(CP[2]), 1] = CP[1]


            # hang为初始点个数
            hang = ppp.shape[0]

            # qqq与ppp数组完全相同
            qqq = np.zeros((hang, 2))
            for i in range(hang):
                for j in range(2):
                    qqq[i, j] = ppp[i, j]


            # 先清除之前的图像
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(6, 6))


            # 原始图像，用实线表示
            for i in range(hang - 1):
                self.ax.plot([qqq[i, 0], qqq[i + 1, 0]], [qqq[i, 1], qqq[i + 1, 1]], color='black')

            self.ax.plot([qqq[-1, 0], qqq[0, 0]], [qqq[-1, 1], qqq[0, 1]], color='black')

            for index, item in enumerate(zip(qqq[:, 0], qqq[:, 1])):
                self.ax.text(item[0], item[1], index, fontsize=13, color='blue')


            # 修改ppp数组：


            magic = []
            for i in range(hang):
                a = np.zeros((2, 1))
                a[0, 0] = ppp[i, 0]
                a[1, 0] = ppp[i, 1]

                magic.append(a)


            # 使用circle_center和cut函数修改magic的元素
            for i in range(Ui_MainWindow.time_c):
                magic[Ui_MainWindow.s[i]] = circle_center(ppp, Ui_MainWindow.s[i], Ui_MainWindow.t[i])

            for i in range(Ui_MainWindow.time_d):
                magic[Ui_MainWindow.u[i]] = cut(ppp, Ui_MainWindow.u[i], Ui_MainWindow.v[i], Ui_MainWindow.w[i])


            global spe
            spe = magic


            # 初始化坐标
            x_points = np.zeros(hang + Ui_MainWindow.time_d + Ui_MainWindow.time_c)
            y_points = np.zeros(hang + Ui_MainWindow.time_d + Ui_MainWindow.time_c)


            m = 0
            x = []
            for i in range(hang):
                if magic[i].shape[0] == 2 and magic[i].shape[1] == 1:
                    x_points[i + m] = magic[i][0, 0]
                    y_points[i + m] = magic[i][1, 0]
                if magic[i].shape[0] == 2 and magic[i].shape[1] == 2:
                    x_points[i + m] = magic[i][0, 0]
                    x_points[i + m + 1] = magic[i][1, 0]
                    y_points[i + m] = magic[i][0, 1]
                    y_points[i + m + 1] = magic[i][1, 1]
                    m += 1
                if magic[i].shape[0] == 3:
                    x_points[i + m] = magic[i][1, 0]
                    x_points[i + m + 1] = magic[i][2, 0]
                    y_points[i + m] = magic[i][1, 1]
                    y_points[i + m + 1] = magic[i][2, 1]
                    x.append(i + m)
                    m += 1

            global X, Y
            X = x_points
            Y = y_points


            # 圆角
            if Ui_MainWindow.time_c != 0:
                for i in range(Ui_MainWindow.time_c):
                    dg = np.zeros((3, 2))
                    # s[i]为编号，t[i]为半径
                    dg = circle_center(qqq, Ui_MainWindow.s[i], Ui_MainWindow.t[i])

                    # 使用Arc函数画圆弧

                    # t1 = theta1
                    # t2 = theta2
                    t1 = 0
                    t2 = 0

                    if dg[0, 0] - dg[1, 0] != 0:
                        k1 = (dg[0, 1] - dg[1, 1]) / (dg[0, 0] - dg[1, 0])
                        if dg[0, 0] < dg[1, 0]:
                            t1 = math.atan(k1)
                        if dg[0, 0] > dg[1, 0]:
                            t1 = math.atan(k1) + math.pi

                        t1 = t1 * 180 / math.pi
                    if dg[0, 0] - dg[1, 0] == 0:
                        if dg[0, 1] > dg[1, 1]:
                            t1 = 270
                        else:
                            t1 = 90

                    if dg[0, 0] - dg[2, 0] != 0:
                        k2 = (dg[0, 1] - dg[2, 1]) / (dg[0, 0] - dg[2, 0])
                        if dg[0, 0] < dg[2, 0]:
                            t2 = math.atan(k2)
                        if dg[0, 0] > dg[2, 0]:
                            t2 = math.atan(k2) + math.pi

                        t2 = t2 * 180 / math.pi
                    if dg[0, 0] - dg[2, 0] == 0:
                        if dg[0, 1] > dg[2, 1]:
                            t2 = 270
                        else:
                            t2 = 90


                    if t2 - t1 > 180 or 0 < t1 - t2 < 180:
                        tm = t1
                        t1 = t2
                        t2 = tm


                    a = Arc((dg[0, 0], dg[0, 1]), Ui_MainWindow.t[i]*2, Ui_MainWindow.t[i]*2, theta1=t1, theta2=t2, color='r', lw=2)

                    global arc
                    arc.append(a)

                    a.set(ls='--')
                    self.ax.add_patch(a)


            # 处理后图像，用红色虚线表示
            if Ui_MainWindow.time_d != 0 or Ui_MainWindow.time_c != 0:
                for i in range(hang + Ui_MainWindow.time_d + Ui_MainWindow.time_c - 1):
                    sign = 1
                    for j in range(len(x)):
                        if i == x[j]:
                            sign = 0
                            break
                    if sign == 1:
                        self.ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color='red', linestyle='--', lw=2)

                self.ax.plot([x_points[-1], x_points[0]], [y_points[-1], y_points[0]], color='red', linestyle='--', lw=2)


            # plt.axis('equal')
            # plt.tick_params(labelsize=13)
            # plt.title("geometric image")
            # plt.xlabel("X(mm)")
            # plt.ylabel("Y(mm)")

            self.ax.axis('equal')
            self.ax.tick_params(labelsize=13)
            self.ax.set_title("geometric image")
            # self.ax.title("geometric image")
            self.ax.set_xlabel("X(mm)")
            self.ax.set_ylabel("Y(mm)")
            # self.ax.xlabel("X(mm)")
            # self.ax.ylabel("Y(mm)")

            # self.ax.cla()
            #
            # self.ax.set_title('Carbon concentration variation curve')
            # self.ax.set_xlabel('time(s)')
            # self.ax.set_ylabel('c(wt%)')


            # 更新Matplotlib图形
            self.picture1_2.draw()

            # l1 = [name2, '/', 'picture1.png']
            # location1 = ''.join(l1)
            #
            # plt.savefig(location1, dpi=1000, bbox_inches='tight', pad_inches=0.5)
            # plt.close()
            #
            #
            # pixmap = QPixmap(location1)
            # self.picture1_2.setPixmap(pixmap)
            # self.picture1_2.setScaledContents(True)

            self.textEdit_2.append("geometric image is created successfully.\n")


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "二维有限元渗碳模拟软件"))
        self.label_8.setText(_translate("MainWindow", "逆时针输入顶点坐标："))
        self.pushButton.setText(_translate("MainWindow", "完成"))
        self.pushButton_2.setText(_translate("MainWindow", "确认"))
        self.label_17.setText(_translate("MainWindow", "角度："))
        self.label_15.setText(_translate("MainWindow", "顶点编号："))
        self.label_16.setText(_translate("MainWindow", "left/right："))
        self.label_14.setText(_translate("MainWindow", "修改角度："))
        self.add1.setText(_translate("MainWindow", "确认"))
        self.r.setText(_translate("MainWindow", "半径："))
        self.num1.setText(_translate("MainWindow", "顶点编号："))
        self.label_2.setText(_translate("MainWindow", "添加圆角："))
        self.num2.setText(_translate("MainWindow", "顶点编号："))
        self.depth2.setText(_translate("MainWindow", "右侧截取长度："))
        self.add2.setText(_translate("MainWindow", "确认"))
        self.label_5.setText(_translate("MainWindow", "添加倒角："))
        self.depth1.setText(_translate("MainWindow", "左侧截取长度："))
        self.label.setText(_translate("MainWindow", "参数设定："))
        self.label_19.setText(_translate("MainWindow", "m^2/s"))
        self.label_3.setText(_translate("MainWindow", "扩散系数："))
        self.toolButton_2.setText(_translate("MainWindow", "..."))
        self.label_21.setText(_translate("MainWindow", "wt%"))
        self.label_20.setText(_translate("MainWindow", "m/s"))
        self.label_13.setText(_translate("MainWindow", "s"))
        self.toolButton_3.setText(_translate("MainWindow", "..."))
        self.label_10.setText(_translate("MainWindow", "渗碳时间："))
        self.label_11.setText(_translate("MainWindow", "初始碳势："))
        self.label_7.setText(_translate("MainWindow", "气氛碳势："))
        self.label_6.setText(_translate("MainWindow", "传递系数："))
        self.label_12.setText(_translate("MainWindow", "wt%"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.reset.setText(_translate("MainWindow", "重置"))
        self.run.setText(_translate("MainWindow", "运行"))
        self.label_4.setText(_translate("MainWindow", "交互窗口："))
        self.textEdit_2.setStyleSheet(_translate("MainWindow", "background:rgb(254, 255, 242);color:black;"))
        self.original_pic.setText(_translate("MainWindow", "几何图像："))
        self.cloud_chart.setText(_translate("MainWindow", "碳浓度分布图像："))
        self.menu.setTitle(_translate("MainWindow", "文件(F)"))
        self.menu_2.setTitle(_translate("MainWindow", "帮助(H)"))
        self.menu_R.setTitle(_translate("MainWindow", "运行(R)"))
        self.menu_T.setTitle(_translate("MainWindow", "工具(T)"))
        self.menu_4.setTitle(_translate("MainWindow", "查看原图"))
        self.menu_5.setTitle(_translate("MainWindow", "碳势变化曲线"))
        self.menu_3.setTitle(_translate("MainWindow", "设置(S)"))
        self.actionnew.setText(_translate("MainWindow", "新建(N)"))
        self.actionopen.setText(_translate("MainWindow", "打开(O)"))
        self.actionguidence.setText(_translate("MainWindow", "入门指南(G)"))
        self.actioncontact_support.setText(_translate("MainWindow", "联系支持(C)"))
        self.actionSubmit_Feedback.setText(_translate("MainWindow", "提交反馈(F)"))
        self.actionAbout.setText(_translate("MainWindow", "关于(A)"))
        self.actionsave.setText(_translate("MainWindow", "保存(S)"))
        self.actionx.setText(_translate("MainWindow", "退出(E)"))
        self.action_R.setText(_translate("MainWindow", "运行(R)"))
        # self.actionp1.setText(_translate("MainWindow", "几何图像"))
        self.actionp2.setText(_translate("MainWindow", "碳势分布图像"))
        self.actionCritical_Path_2.setText(_translate("MainWindow", "Critical Path"))
        self.actionsetting.setText(_translate("MainWindow", "设置(S)"))
        self.actionpoint.setText(_translate("MainWindow", "point"))
        self.actionline.setText(_translate("MainWindow", "line"))


class Ui_points(object):
    def setupUi(self, points):
        points.setObjectName("points")
        points.resize(800, 450)
        self.horizontalLayout = QtWidgets.QHBoxLayout(points)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.textEdit = QtWidgets.QTextEdit(points)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)

        self.retranslateUi(points)
        QtCore.QMetaObject.connectSlotsByName(points)

    def retranslateUi(self, points):
        _translate = QtCore.QCoreApplication.translate
        points.setWindowTitle(_translate("points", "查看文本文件"))

    def show(self, name):

        f = open(name, 'r')
        content = f.read()
        self.textEdit.append(content)
        f.close()


class Ui_support(object):
    def setupUi(self, support):
        support.setObjectName("support")
        support.resize(int(height * 0.75), int(height * 0.42))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        support.setFont(font)
        self.horizontalLayout = QtWidgets.QHBoxLayout(support)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(support)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.horizontalLayout.addWidget(self.plainTextEdit)

        self.retranslateUi(support)
        QtCore.QMetaObject.connectSlotsByName(support)

    def retranslateUi(self, support):
        _translate = QtCore.QCoreApplication.translate
        support.setWindowTitle(_translate("support", "联系支持"))
        self.plainTextEdit.setPlainText(_translate("support", "如果需要联系并获得支持，请发送邮件至：\n"
        "3566526738@qq.com\n"
        "或\n"
        "bigyellowno.1@sjtu.edu.com\n"
        "或\n"
        "bigyellowno.1@gmail.com\n"
        "\n"
        "\n"
        "If you want to contact me and gain support, please send e-mails to:\n"
        "3566526738@qq.com\n"
        "or\n"
        "bigyellowno.1@sjtu.edu.com\n"
        "or\n"
        "bigyellowno.1@gmail.com"))


class Ui_feedback(object):
    def setupUi(self, feedback):
        feedback.setObjectName("feedback")
        feedback.resize(int(height * 0.75), int(height * 0.42))
        self.horizontalLayout = QtWidgets.QHBoxLayout(feedback)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(feedback)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.horizontalLayout.addWidget(self.plainTextEdit)

        self.retranslateUi(feedback)
        QtCore.QMetaObject.connectSlotsByName(feedback)

    def retranslateUi(self, feedback):
        _translate = QtCore.QCoreApplication.translate
        feedback.setWindowTitle(_translate("feedback", "提交反馈"))
        self.plainTextEdit.setPlainText(_translate("feedback", "如果需要提交反馈，请发送邮件至：\n"
        "3566526738@qq.com\n"
        "或\n"
        "bigyellowno.1@sjtu.edu.com\n"
        "或\n"
        "bigyellowno.1@gmail.com\n"
        "\n"
        "\n"
        "If you want to submit feedbacks, please send e-mails to:\n"
        "3566526738@qq.com\n"
        "or\n"
        "bigyellowno.1@sjtu.edu.com\n"
        "or\n"
        "bigyellowno.1@gmail.com"))


# createnew类
class Ui_createnew(object):
    def setupUi(self, createnew):
        createnew.setWindowFlags(Qt.WindowCloseButtonHint)
        createnew.setObjectName("createnew")
        createnew.resize(int(height * 0.26), int(height / 10))
        createnew.setLayoutDirection(QtCore.Qt.LeftToRight)
        createnew.setAutoFillBackground(False)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(createnew)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(createnew)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(createnew)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(createnew)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.retranslateUi(createnew)
        self.pushButton.clicked.connect(self.confirm)
        self.pushButton.clicked.connect(createnew.close)  # type: ignore
        self.lineEdit.returnPressed.connect(self.confirm)
        self.lineEdit.returnPressed.connect(createnew.close)  # type: ignore

        QtCore.QMetaObject.connectSlotsByName(createnew)


    def retranslateUi(self, createnew):
        _translate = QtCore.QCoreApplication.translate
        createnew.setWindowTitle(_translate("createnew", "新建文件夹"))
        self.label.setText(_translate("createnew", "名称："))
        self.pushButton.setText(_translate("createnew", "确定"))


    def confirm(self):

        global name3

        if yesORno == 0:
            name3 = "新建失败，请先登录！"

        else:
            name = self.lineEdit.text() # str

            global cwd
            cwd = os.getcwd()  # 获取当前文件运行的目录




            if len(name) != 0:
                list = [cwd, '/', name]

                global name2
                name2 = ''.join(list)

                if os.path.exists(name2) is False:
                    os.mkdir(name2)

                    list2 = ["A new folder named ", "'", name, "'", " has been created.\n"]

                    name3 = ''.join(list2)
                else:
                    name3 = "Error: The folder named " + "'" + name + "'" + " has existed.\n"

            if len(name) == 0:
                name3 = "Error: The folder name cannot be empty!\n"


# set类
class Ui_set(object):

    def setupUi(self, set):
        set.setObjectName("set")
        set.resize(408, 111)
        self.horizontalLayout = QtWidgets.QHBoxLayout(set)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(set)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 2)
        self.label = QtWidgets.QLabel(set)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(set)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(set)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(set)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 1, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(set)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(set)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 2, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(set)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 2, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(set)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 2, 3, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)


        self.retranslateUi(set)
        self.pushButton.clicked.connect(self.update)
        self.pushButton_2.clicked.connect(self.update2)
        QtCore.QMetaObject.connectSlotsByName(set)


    def retranslateUi(self, set):
        _translate = QtCore.QCoreApplication.translate
        set.setWindowTitle(_translate("set", "设置"))
        self.label_3.setText(_translate("set", "重新设置网格密度和时间步数:"))
        self.label.setText(_translate("set", "网格密度："))
        self.label_4.setText(_translate("set", "(10-100)"))
        self.pushButton.setText(_translate("set", "OK"))
        self.label_2.setText(_translate("set", "时间步数："))
        self.label_5.setText(_translate("set", "(10-1000)"))
        self.pushButton_2.setText(_translate("set", "OK"))

    def update(self):
        # num为网格密度
        # step0为时间步数
        num = int(self.lineEdit.text())

        global density
        density = num


    def update2(self):
        step0 = int(self.lineEdit_2.text())

        global stepnum
        stepnum = step0


class Ui_About(object):
    def setupUi(self, About):
        About.setObjectName("About")
        About.resize(int(height * 0.628), int(height * 0.25))
        self.horizontalLayout = QtWidgets.QHBoxLayout(About)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(About)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.horizontalLayout.addWidget(self.plainTextEdit)

        self.retranslateUi(About)
        QtCore.QMetaObject.connectSlotsByName(About)

    def retranslateUi(self, About):
        _translate = QtCore.QCoreApplication.translate
        About.setWindowTitle(_translate("About", "关于"))
        self.plainTextEdit.setPlainText(_translate("About", "该软件基于有限元方法，用于模拟二维扩散过程。\n"
                                                            "\n"
                                                            "使用python编写，主要用到的外部库为matplotlib、gmsh、numpy、PYQT5和scipy等。\n"
                                                            "\n"
                                                            "当前版本为：version2.5"))


# 时间历程图
class Ui_HistoryChart(object):
    magic = []

    def setupUi(self, HistoryChart):
        HistoryChart.setObjectName("HistoryChart")
        HistoryChart.resize(int(width * 0.54), int(height * 0.54))
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(HistoryChart)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(HistoryChart)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(HistoryChart)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(HistoryChart)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.label_4 = QtWidgets.QLabel(HistoryChart)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(HistoryChart)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(HistoryChart)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout.addWidget(self.lineEdit_2)
        self.label_5 = QtWidgets.QLabel(HistoryChart)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton = QtWidgets.QPushButton(HistoryChart)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.pushButton_2 = QtWidgets.QPushButton(HistoryChart)
        self.pushButton_2.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        # self.photo2 = QtWidgets.QLabel(HistoryChart)
        # self.photo2.setMinimumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        # self.photo2.setMaximumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        # self.photo2.setFrameShape(QtWidgets.QFrame.Box)
        # self.photo2.setText("")
        # self.photo2.setObjectName("photo2")
        # self.horizontalLayout_5.addWidget(self.photo2)
        self.figure = plt.figure()
        self.photo2 = FigureCanvas(self.figure)
        self.photo2.setMinimumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        self.photo2.setMaximumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        self.horizontalLayout_5.addWidget(self.photo2)

        self.ax = self.figure.add_subplot(111)

        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_7.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self.pushButton_3 = QtWidgets.QPushButton(HistoryChart)
        self.pushButton_3.setMinimumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_6.addWidget(self.pushButton_3)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem7)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem8)
        # self.photo = QtWidgets.QLabel(HistoryChart)
        # self.photo.setMinimumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        # self.photo.setMaximumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        # self.photo.setFrameShape(QtWidgets.QFrame.Box)
        # self.photo.setFrameShadow(QtWidgets.QFrame.Plain)
        # self.photo.setText("")
        # self.photo.setObjectName("photo")
        # self.verticalLayout_4.addWidget(self.photo)
        self.figure2 = plt.figure()
        self.photo = FigureCanvas(self.figure2)
        self.photo.setMinimumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        self.photo.setMaximumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        self.verticalLayout_4.addWidget(self.photo)

        self.ax2 = self.figure2.add_subplot(111)

        self.horizontalLayout_7.addLayout(self.verticalLayout_4)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)

        self.retranslateUi(HistoryChart)
        self.pushButton.clicked.connect(self.paint)
        # self.pushButton.clicked.connect(self.paint0)


        self.pushButton_2.clicked.connect(self.lineEdit.clear)  # type: ignore
        self.pushButton_2.clicked.connect(self.lineEdit_2.clear)  # type: ignore
        # self.pushButton_2.clicked.connect(self.photo.clear)  # type: ignore
        # self.pushButton_2.clicked.connect(self.photo2.clear) # type: ignore
        self.pushButton_3.clicked.connect(self.savetxt)
        QtCore.QMetaObject.connectSlotsByName(HistoryChart)


        # hang为初始点个数
        hang = ppp.shape[0]

        # qqq与ppp数组完全相同
        qqq = np.zeros((hang, 2))
        for i in range(hang):
            for j in range(2):
                qqq[i, j] = ppp[i, j]

        # 初始化fig AND ax
        # fig, ax = plt.subplots(figsize=(6, 6))

        # 原始图像，用实线表示
        if time_c1 == 0 and time_d1 == 0:
            for i in range(hang - 1):
                self.ax.plot([qqq[i, 0], qqq[i + 1, 0]], [qqq[i, 1], qqq[i + 1, 1]], color='black')

            self.ax.plot([qqq[-1, 0], qqq[0, 0]], [qqq[-1, 1], qqq[0, 1]], color='black')

        # 修改ppp数组：
        magic = []

        magic = spe

        Ui_HistoryChart.magic = magic

        # 初始化坐标
        x_points = np.zeros(hang + time_d1 + time_c1)
        y_points = np.zeros(hang + time_d1 + time_c1)

        m = 0
        x = []
        for i in range(hang):
            if magic[i].shape[0] == 2 and magic[i].shape[1] == 1:
                x_points[i + m] = magic[i][0, 0]
                y_points[i + m] = magic[i][1, 0]
            if magic[i].shape[0] == 2 and magic[i].shape[1] == 2:
                x_points[i + m] = magic[i][0, 0]
                x_points[i + m + 1] = magic[i][1, 0]
                y_points[i + m] = magic[i][0, 1]
                y_points[i + m + 1] = magic[i][1, 1]
                m += 1
            if magic[i].shape[0] == 3:
                x_points[i + m] = magic[i][1, 0]
                x_points[i + m + 1] = magic[i][2, 0]
                y_points[i + m] = magic[i][1, 1]
                y_points[i + m + 1] = magic[i][2, 1]
                x.append(i + m)
                m += 1

        # 圆角
        if time_c1 != 0:
            for i in range(time_c1):
                dg = np.zeros((3, 2))
                # s[i]为编号，t[i]为半径
                dg = circle_center(qqq, sss[i], ttt[i])

                # 使用Arc函数画圆弧

                # t1 = theta1
                # t2 = theta2
                t1 = 0
                t2 = 0

                if dg[0, 0] - dg[1, 0] != 0:
                    k1 = (dg[0, 1] - dg[1, 1]) / (dg[0, 0] - dg[1, 0])
                    if dg[0, 0] < dg[1, 0]:
                        t1 = math.atan(k1)
                    if dg[0, 0] > dg[1, 0]:
                        t1 = math.atan(k1) + math.pi

                    t1 = t1 * 180 / math.pi
                if dg[0, 0] - dg[1, 0] == 0:
                    if dg[0, 1] > dg[1, 1]:
                        t1 = 270
                    else:
                        t1 = 90

                if dg[0, 0] - dg[2, 0] != 0:
                    k2 = (dg[0, 1] - dg[2, 1]) / (dg[0, 0] - dg[2, 0])
                    if dg[0, 0] < dg[2, 0]:
                        t2 = math.atan(k2)
                    if dg[0, 0] > dg[2, 0]:
                        t2 = math.atan(k2) + math.pi

                    t2 = t2 * 180 / math.pi
                if dg[0, 0] - dg[2, 0] == 0:
                    if dg[0, 1] > dg[2, 1]:
                        t2 = 270
                    else:
                        t2 = 90

                if t2 - t1 > 180 or 0 < t1 - t2 < 180:
                    tm = t1
                    t1 = t2
                    t2 = tm

                a = Arc((dg[0, 0], dg[0, 1]), ttt[i] * 2, ttt[i] * 2, theta1=t1, theta2=t2,
                        color='black', lw=2)

                self.ax.add_patch(a)

        # 处理后图像，用红色虚线表示
        if time_d1 != 0 or time_c1 != 0:
            for i in range(hang + time_d1 + time_c1 - 1):
                sign = 1
                for j in range(len(x)):
                    if i == x[j]:
                        sign = 0
                        break
                if sign == 1:
                    self.ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color='black', lw=2)

            self.ax.plot([x_points[-1], x_points[0]], [y_points[-1], y_points[0]], color='black', lw=2)

        # # 读取lineEdit中的坐标值
        # given_x = float(self.lineEdit.text())
        # given_y = float(self.lineEdit_2.text())
        #
        #
        # ax.scatter(given_x, given_y)

        # plt.tick_params(labelsize=13)

        self.ax.axis('equal')
        self.ax.set_title('geometric image')
        self.ax.set_xlabel('X(mm)')
        self.ax.set_ylabel('Y(mm)')

        self.photo2.draw()


    def retranslateUi(self, HistoryChart):
        _translate = QtCore.QCoreApplication.translate
        HistoryChart.setWindowTitle(_translate("HistoryChart", "碳浓度变化曲线"))
        self.label.setText(_translate("HistoryChart", "input x, y："))
        self.label_2.setText(_translate("HistoryChart", "x:"))
        self.label_4.setText(_translate("HistoryChart", "mm"))
        self.label_3.setText(_translate("HistoryChart", "y:"))
        self.label_5.setText(_translate("HistoryChart", "mm"))
        self.pushButton.setText(_translate("HistoryChart", "run"))
        self.pushButton_2.setText(_translate("HistoryChart", "clear"))
        self.pushButton_3.setText(_translate("HistoryChart", "导出数据"))

    # def paint0(self):
    #     # hang为初始点个数
    #     hang = ppp.shape[0]
    #
    #     # qqq与ppp数组完全相同
    #     qqq = np.zeros((hang, 2))
    #     for i in range(hang):
    #         for j in range(2):
    #             qqq[i, j] = ppp[i, j]
    #
    #
    #     # 初始化fig AND ax
    #     # fig, ax = plt.subplots(figsize=(6, 6))
    #
    #     # 原始图像，用实线表示
    #     if time_c1 == 0 and time_d1 == 0:
    #         for i in range(hang - 1):
    #             self.ax.plot([qqq[i, 0], qqq[i + 1, 0]], [qqq[i, 1], qqq[i + 1, 1]], color='black')
    #
    #         self.ax.plot([qqq[-1, 0], qqq[0, 0]], [qqq[-1, 1], qqq[0, 1]], color='black')
    #
    #     # 修改ppp数组：
    #     magic = []
    #
    #     magic = spe
    #
    #     Ui_HistoryChart.magic = magic
    #
    #     # 初始化坐标
    #     x_points = np.zeros(hang + time_d1 + time_c1)
    #     y_points = np.zeros(hang + time_d1 + time_c1)
    #
    #     m = 0
    #     x = []
    #     for i in range(hang):
    #         if magic[i].shape[0] == 2 and magic[i].shape[1] == 1:
    #             x_points[i + m] = magic[i][0, 0]
    #             y_points[i + m] = magic[i][1, 0]
    #         if magic[i].shape[0] == 2 and magic[i].shape[1] == 2:
    #             x_points[i + m] = magic[i][0, 0]
    #             x_points[i + m + 1] = magic[i][1, 0]
    #             y_points[i + m] = magic[i][0, 1]
    #             y_points[i + m + 1] = magic[i][1, 1]
    #             m += 1
    #         if magic[i].shape[0] == 3:
    #             x_points[i + m] = magic[i][1, 0]
    #             x_points[i + m + 1] = magic[i][2, 0]
    #             y_points[i + m] = magic[i][1, 1]
    #             y_points[i + m + 1] = magic[i][2, 1]
    #             x.append(i + m)
    #             m += 1
    #
    #     # 圆角
    #     if time_c1 != 0:
    #         for i in range(time_c1):
    #             dg = np.zeros((3, 2))
    #             # s[i]为编号，t[i]为半径
    #             dg = circle_center(qqq, sss[i], ttt[i])
    #
    #             # 使用Arc函数画圆弧
    #
    #             # t1 = theta1
    #             # t2 = theta2
    #             t1 = 0
    #             t2 = 0
    #
    #             if dg[0, 0] - dg[1, 0] != 0:
    #                 k1 = (dg[0, 1] - dg[1, 1]) / (dg[0, 0] - dg[1, 0])
    #                 if dg[0, 0] < dg[1, 0]:
    #                     t1 = math.atan(k1)
    #                 if dg[0, 0] > dg[1, 0]:
    #                     t1 = math.atan(k1) + math.pi
    #
    #                 t1 = t1 * 180 / math.pi
    #             if dg[0, 0] - dg[1, 0] == 0:
    #                 if dg[0, 1] > dg[1, 1]:
    #                     t1 = 270
    #                 else:
    #                     t1 = 90
    #
    #             if dg[0, 0] - dg[2, 0] != 0:
    #                 k2 = (dg[0, 1] - dg[2, 1]) / (dg[0, 0] - dg[2, 0])
    #                 if dg[0, 0] < dg[2, 0]:
    #                     t2 = math.atan(k2)
    #                 if dg[0, 0] > dg[2, 0]:
    #                     t2 = math.atan(k2) + math.pi
    #
    #                 t2 = t2 * 180 / math.pi
    #             if dg[0, 0] - dg[2, 0] == 0:
    #                 if dg[0, 1] > dg[2, 1]:
    #                     t2 = 270
    #                 else:
    #                     t2 = 90
    #
    #             if t2 - t1 > 180 or 0 < t1 - t2 < 180:
    #                 tm = t1
    #                 t1 = t2
    #                 t2 = tm
    #
    #             a = Arc((dg[0, 0], dg[0, 1]), ttt[i] * 2, ttt[i] * 2, theta1=t1, theta2=t2,
    #                     color='black', lw=2)
    #
    #             self.ax.add_patch(a)
    #
    #     # 处理后图像，用红色虚线表示
    #     if time_d1 != 0 or time_c1 != 0:
    #         for i in range(hang + time_d1 + time_c1 - 1):
    #             sign = 1
    #             for j in range(len(x)):
    #                 if i == x[j]:
    #                     sign = 0
    #                     break
    #             if sign == 1:
    #                 self.ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color='black', lw=2)
    #
    #         self.ax.plot([x_points[-1], x_points[0]], [y_points[-1], y_points[0]], color='black', lw=2)
    #
    #
    #     # # 读取lineEdit中的坐标值
    #     # given_x = float(self.lineEdit.text())
    #     # given_y = float(self.lineEdit_2.text())
    #     #
    #     #
    #     # ax.scatter(given_x, given_y)
    #
    #
    #
    #     # plt.tick_params(labelsize=13)
    #
    #     self.ax.axis('equal')
    #     self.ax.set_title('geometric image')
    #     self.ax.set_xlabel('X(mm)')
    #     self.ax.set_ylabel('Y(mm)')
    #
    #     self.photo2.draw()
    #     # 将图片存入address
    #     # address = ''.join([name2, '/', 'picture7.png'])
    #     # plt.savefig(address, dpi=1000, bbox_inches='tight')
    #     # plt.close()
    #     #
    #     # # 在窗口中显示图片
    #     # pixmap = QPixmap(address)
    #     # self.photo2.setPixmap(pixmap)
    #     # self.photo2.setScaledContents(True)
    #

    def savetxt(self):
        name, ok = QInputDialog.getText(MainWindow, "新建txt文档", "请输入文件名称：", QtWidgets.QLineEdit.Normal)
        if ok:  # 判断是否单击的ok按钮
            file_name = name  # 获取输入对话框中的字符串，显示在文本框中

        try:
            length = len(self.time)
            curve = np.zeros((length, 2))

            for i in range(length):
                for j in range(2):
                    if j == 0:
                        curve[i, j] = self.time[i]
                    if j == 1:
                        curve[i, j] = self.tpt[i]

            np.savetxt(''.join([name2, '/', str(file_name), '.txt']), curve)

            QMessageBox.information(None, "information", "数据已成功导出", QMessageBox.Ok)
        except:
            QMessageBox.warning(None, "警告", "数据未成功导出", QMessageBox.Ok)

    def paint(self):
        self.ax2.cla()
        # 读取x, y坐标值
        if len(self.lineEdit.text()) != 0:
            x = float(self.lineEdit.text())
            x = x / 1000
        if len(self.lineEdit_2.text()) != 0:
            y = float(self.lineEdit_2.text())
            y = y / 1000

        # 读取lineEdit中的坐标值
        given_x = float(self.lineEdit.text())
        given_y = float(self.lineEdit_2.text())

        self.ax.scatter(given_x, given_y)
        self.photo2.draw()

        print(x, y)
        # 利用形函数插值

        arr1 = ChangedArr
        num = len(arr1)


        distance = np.zeros(num)

        for i in range(num):
            distance[i] = (arr1[i, 0] - x)**2 + (arr1[i, 1] - y)**2

        index = np.argmin(distance) # 得到距离插值点最近的已知点的索引


        # 判断所选择的点是否在封闭图形内
        arr2 = ChangedArr2
        num_cell = len(arr2)

        list_cell = []
        for i in range(num_cell):
            for j in range(3):
                if int(arr2[i, j]) == index:
                    list_cell.append(i)

        print(list_cell)

        func1 = 0
        func2 = 0
        func3 = 0

        label1 = 0
        label2 = 0
        label3 = 0

        flag = 0
        for i in range(len(list_cell)):
            # 三角形各顶点的编号
            b1 = arr2[list_cell[i], 0]
            b2 = arr2[list_cell[i], 1]
            b3 = arr2[list_cell[i], 2]

            x1 = arr1[int(b1), 0]
            y1 = arr1[int(b1), 1]
            x2 = arr1[int(b2), 0]
            y2 = arr1[int(b2), 1]
            x3 = arr1[int(b3), 0]
            y3 = arr1[int(b3), 1]


            # 求得四个三角形的面积，再作比较，判断输入点是否在三角形内
            s = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)
            s1 = abs((x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)) / 2)
            s2 = abs((x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y)) / 2)
            s3 = abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2)

            if (s - s1 - s2 - s3) * 1e9 >= -1:
                print(x1, y1, x2, y2, x3, y3)
                print(s, s1, s2, s3)
                flag += 1

                a1 = x2 * y3 - x3 * y2
                beta1 = y2 - y3
                c1 = x3 - x2

                a2 = x3 * y1 - x1 * y3
                beta2 = y3 - y1
                c2 = x1 - x3

                a3 = x1 * y2 - x2 * y1
                beta3 = y1 - y2
                c3 = x2 - x1

                func1 = (a1 + beta1 * x + c1 * y) / (2 * s)
                func2 = (a2 + beta2 * x + c2 * y) / (2 * s)
                func3 = (a3 + beta3 * x + c3 * y) / (2 * s)
                label1 = int(b1)
                label2 = int(b2)
                label3 = int(b3)

                break
            else:
                # print(s - s1 - s2 - s3)
                pass
        print(flag)

        if flag == 0:

            # 后两项分别为按钮(以|隔开，共有7种按钮类型，见示例后)、默认按钮(省略则默认为第一个按钮)
            QMessageBox.warning(None, "警告", "所取点不在图形内！", QMessageBox.Ok)

        if flag != 0:

            degree = []

            t_max = np.max(t)
            for ti in t:
                if ti != t_max:
                    temperature = ''.join([name2, '/', 'temperature', str(ti), '.txt'])
                    tem = np.loadtxt(temperature)

                    c_want = func1 * tem[label1] + func2 * tem[label2] + func3 * tem[label3]
                    degree.append(c_want)
                else:
                    temperature = ''.join([name2, '/', 'temperature.txt'])
                    tem = np.loadtxt(temperature)

                    c_want = func1 * tem[label1] + func2 * tem[label2] + func3 * tem[label3]
                    degree.append(c_want)


            time = np.insert(t, 0, 0)
            tpt = np.array(degree)
            tpt = np.insert(tpt, 0, t0)

            self.time = time


            # 对碳浓度保留4位小数
            for i in range(tpt.shape[0]):
                tpt[i] = int(tpt[i] * 1000) / 1000

            self.tpt = tpt
            # plt.figure(figsize=(8, 6))
            self.ax2.plot(time, tpt)

            self.ax2.set_title("concentration variation curve")
            self.ax2.set_xlabel("t(s)")
            self.ax2.set_ylabel("c(wt%)")
            if t0 != tpt.max():
                self.ax2.set_ylim(t0, tpt.max())
            # plt.axis([0, time_hc ,t0, tpt.max()])

            self.photo.draw()


class Ui_Line(object):
    magic = []

    def setupUi(self, Line):
        Line.setObjectName("Line")
        Line.resize(int(width * 0.54), int(height * 0.54))
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(Line)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Line)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(Line)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label_4 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.lineEditx1 = QtWidgets.QLineEdit(Line)
        self.lineEditx1.setMinimumSize(QtCore.QSize(120, 0))
        self.lineEditx1.setMaximumSize(QtCore.QSize(120, 16777215))
        self.lineEditx1.setObjectName("lineEditx1")
        self.horizontalLayout.addWidget(self.lineEditx1)
        self.label_8 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_5 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.lineEdity1 = QtWidgets.QLineEdit(Line)
        self.lineEdity1.setMinimumSize(QtCore.QSize(120, 0))
        self.lineEdity1.setMaximumSize(QtCore.QSize(120, 16777215))
        self.lineEdity1.setObjectName("lineEdity1")
        self.horizontalLayout.addWidget(self.lineEdity1)
        self.label_9 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(Line)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.label_6 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        self.lineEditx2 = QtWidgets.QLineEdit(Line)
        self.lineEditx2.setMinimumSize(QtCore.QSize(120, 0))
        self.lineEditx2.setMaximumSize(QtCore.QSize(120, 16777215))
        self.lineEditx2.setObjectName("lineEditx2")
        self.horizontalLayout_2.addWidget(self.lineEditx2)
        self.label_10 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_2.addWidget(self.label_10)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.label_7 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.lineEdity2 = QtWidgets.QLineEdit(Line)
        self.lineEdity2.setMinimumSize(QtCore.QSize(120, 0))
        self.lineEdity2.setMaximumSize(QtCore.QSize(120, 16777215))
        self.lineEdity2.setObjectName("lineEdity2")
        self.horizontalLayout_2.addWidget(self.lineEdity2)
        self.label_11 = QtWidgets.QLabel(Line)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_2.addWidget(self.label_11)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.pushButton = QtWidgets.QPushButton(Line)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem7)
        self.pushButton_2 = QtWidgets.QPushButton(Line)
        self.pushButton_2.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem8)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem9)
        # self.photo2 = QtWidgets.QLabel(Line)
        # self.photo2.setMinimumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        # self.photo2.setMaximumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        # self.photo2.setFrameShape(QtWidgets.QFrame.Box)
        # self.photo2.setText("")
        # self.photo2.setObjectName("photo2")
        # self.horizontalLayout_6.addWidget(self.photo2)
        self.figure = plt.figure()
        self.photo2 = FigureCanvas(self.figure)
        self.photo2.setMinimumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        self.photo2.setMaximumSize(QtCore.QSize(int(height * 0.37), int(height * 0.37)))
        self.horizontalLayout_6.addWidget(self.photo2)

        self.ax = self.figure.add_subplot(111)

        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem10)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem11)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem12)
        self.pushButton_3 = QtWidgets.QPushButton(Line)
        self.pushButton_3.setMinimumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_7.addWidget(self.pushButton_3)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem13)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        spacerItem14 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem14)
        # self.photo = QtWidgets.QLabel(Line)
        # self.photo.setMinimumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        # self.photo.setMaximumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        # self.photo.setFrameShape(QtWidgets.QFrame.Box)
        # self.photo.setText("")
        # self.photo.setObjectName("photo")
        # self.verticalLayout_3.addWidget(self.photo)
        self.figure2 = plt.figure()
        self.photo = FigureCanvas(self.figure2)
        self.photo.setMinimumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        self.photo.setMaximumSize(QtCore.QSize(int(height * 0.56), int(height * 0.42)))
        self.verticalLayout_3.addWidget(self.photo)

        self.ax2 = self.figure2.add_subplot(111)

        self.horizontalLayout_5.addLayout(self.verticalLayout_3)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_5)

        self.retranslateUi(Line)
        self.pushButton_2.clicked.connect(self.lineEditx1.clear) # type: ignore
        self.pushButton_2.clicked.connect(self.lineEdity1.clear) # type: ignore
        self.pushButton_2.clicked.connect(self.lineEditx2.clear) # type: ignore
        self.pushButton_2.clicked.connect(self.lineEdity2.clear) # type: ignore

        self.pushButton_3.clicked.connect(self.savetxt)
        # self.pushButton_2.clicked.connect(self.photo.clear) # type: ignore
        # self.pushButton_2.clicked.connect(self.photo2.clear) # type: ignore


        # self.pushButton.clicked.connect(self.paint0)
        self.pushButton.clicked.connect(self.paint)

        # hang为初始点个数
        hang = ppp.shape[0]

        # qqq与ppp数组完全相同
        qqq = np.zeros((hang, 2))
        for i in range(hang):
            for j in range(2):
                qqq[i, j] = ppp[i, j]

        # 先清除之前的图像
        # plt.clf()
        # 初始化fig AND ax
        # fig, ax = plt.subplots(figsize=(6, 6))

        # 原始图像，用实线表示
        if time_c1 == 0 and time_d1 == 0:
            for i in range(hang - 1):
                self.ax.plot([qqq[i, 0], qqq[i + 1, 0]], [qqq[i, 1], qqq[i + 1, 1]], color='black')

            self.ax.plot([qqq[-1, 0], qqq[0, 0]], [qqq[-1, 1], qqq[0, 1]], color='black')

        # 修改ppp数组：
        magic = []

        magic = spe

        Ui_Line.magic = magic

        # 初始化坐标
        x_points = np.zeros(hang + time_d1 + time_c1)
        y_points = np.zeros(hang + time_d1 + time_c1)

        m = 0
        x = []
        for i in range(hang):
            if magic[i].shape[0] == 2 and magic[i].shape[1] == 1:
                x_points[i + m] = magic[i][0, 0]
                y_points[i + m] = magic[i][1, 0]
            if magic[i].shape[0] == 2 and magic[i].shape[1] == 2:
                x_points[i + m] = magic[i][0, 0]
                x_points[i + m + 1] = magic[i][1, 0]
                y_points[i + m] = magic[i][0, 1]
                y_points[i + m + 1] = magic[i][1, 1]
                m += 1
            if magic[i].shape[0] == 3:
                x_points[i + m] = magic[i][1, 0]
                x_points[i + m + 1] = magic[i][2, 0]
                y_points[i + m] = magic[i][1, 1]
                y_points[i + m + 1] = magic[i][2, 1]
                x.append(i + m)
                m += 1

        # 圆角
        if time_c1 != 0:
            for i in range(time_c1):
                dg = np.zeros((3, 2))
                # s[i]为编号，t[i]为半径
                dg = circle_center(qqq, sss[i], ttt[i])

                # 使用Arc函数画圆弧

                # t1 = theta1
                # t2 = theta2
                t1 = 0
                t2 = 0

                if dg[0, 0] - dg[1, 0] != 0:
                    k1 = (dg[0, 1] - dg[1, 1]) / (dg[0, 0] - dg[1, 0])
                    if dg[0, 0] < dg[1, 0]:
                        t1 = math.atan(k1)
                    if dg[0, 0] > dg[1, 0]:
                        t1 = math.atan(k1) + math.pi

                    t1 = t1 * 180 / math.pi
                if dg[0, 0] - dg[1, 0] == 0:
                    if dg[0, 1] > dg[1, 1]:
                        t1 = 270
                    else:
                        t1 = 90

                if dg[0, 0] - dg[2, 0] != 0:
                    k2 = (dg[0, 1] - dg[2, 1]) / (dg[0, 0] - dg[2, 0])
                    if dg[0, 0] < dg[2, 0]:
                        t2 = math.atan(k2)
                    if dg[0, 0] > dg[2, 0]:
                        t2 = math.atan(k2) + math.pi

                    t2 = t2 * 180 / math.pi
                if dg[0, 0] - dg[2, 0] == 0:
                    if dg[0, 1] > dg[2, 1]:
                        t2 = 270
                    else:
                        t2 = 90

                if t2 - t1 > 180 or 0 < t1 - t2 < 180:
                    tm = t1
                    t1 = t2
                    t2 = tm

                a = Arc((dg[0, 0], dg[0, 1]), ttt[i] * 2, ttt[i] * 2, theta1=t1, theta2=t2,
                        color='black', lw=2)

                self.ax.add_patch(a)

        # 处理后图像，用红色虚线表示
        if time_d1 != 0 or time_c1 != 0:
            for i in range(hang + time_d1 + time_c1 - 1):
                sign = 1
                for j in range(len(x)):
                    if i == x[j]:
                        sign = 0
                        break
                if sign == 1:
                    self.ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color='black', lw=2)

            self.ax.plot([x_points[-1], x_points[0]], [y_points[-1], y_points[0]], color='black', lw=2)



        self.ax.axis('equal')
        self.ax.tick_params(labelsize=13)
        self.ax.set_title("geometric image")
        self.ax.set_xlabel("X(mm)")
        self.ax.set_ylabel("Y(mm)")

        self.photo2.draw()


        QtCore.QMetaObject.connectSlotsByName(Line)

    def retranslateUi(self, Line):
        _translate = QtCore.QCoreApplication.translate
        Line.setWindowTitle(_translate("Line", "碳浓度变化曲线"))
        self.label.setText(_translate("Line", "输入线段起点和终点坐标："))
        self.label_2.setText(_translate("Line", "起点："))
        self.label_4.setText(_translate("Line", "x1:"))
        self.label_8.setText(_translate("Line", "mm"))
        self.label_5.setText(_translate("Line", "y1:"))
        self.label_9.setText(_translate("Line", "mm"))
        self.label_3.setText(_translate("Line", "终点："))
        self.label_6.setText(_translate("Line", "x2:"))
        self.label_10.setText(_translate("Line", "mm"))
        self.label_7.setText(_translate("Line", "y2:"))
        self.label_11.setText(_translate("Line", "mm"))
        self.pushButton.setText(_translate("Line", "run"))
        self.pushButton_2.setText(_translate("Line", "clear"))
        self.pushButton_3.setText(_translate("Line", "导出数据"))

    def savetxt(self):
        name, ok = QInputDialog.getText(MainWindow, "新建txt文档", "请输入文件名称：", QtWidgets.QLineEdit.Normal)
        if ok:  # 判断是否单击的ok按钮
            file_name = name  # 获取输入对话框中的字符串，显示在文本框中


        try:
            l = len(self.length)
            curve = np.zeros((l, 2))

            for i in range(l):
                for j in range(2):
                    if j == 0:
                        curve[i, j] = self.length[i]
                    if j == 1:
                        curve[i, j] = self.tpt[i]

            np.savetxt(''.join([name2, '/', str(file_name), '.txt']), curve)

            QMessageBox.information(None, "information", "数据已成功导出", QMessageBox.Ok)
        except:
            QMessageBox.warning(None, "警告", "数据未成功导出", QMessageBox.Ok)


    def paint(self):
        # 画出线段
        # 读取lineEdit中的坐标值
        x11 = float(self.lineEditx1.text())
        y11 = float(self.lineEdity1.text())
        x22 = float(self.lineEditx2.text())
        y22 = float(self.lineEdity2.text())

        self.ax.plot([x11, x22], [y11, y22], lw=3)
        self.photo2.draw()

        self.ax2.cla()
        # 读取lineEdit中的坐标值
        x11 = x11 / 1000
        y11 = y11 / 1000
        x22 = x22 / 1000
        y22 = y22 / 1000

        num = 51  # 在一条线段上取21个点，划分成20个小格
        array = np.zeros((num, 2))  # array用以储存原始点的坐标

        array[0, 0] = x11
        array[0, 1] = y11
        array[-1, 0] = x22
        array[-1, 1] = y22

        # 线段不垂直于x轴
        if x11 != x22:

            k = (y22 - y11) / (x22 - x11)
            b = y11 - k * x11

            dx = (x22 - x11) / (num - 1)

            for i in range(num - 2):
                array[i + 1, 0] = x11 + dx * (i + 1)
                array[i + 1, 1] = array[i + 1, 0] * k + b
        # 线段垂直于x轴
        if x11 == x22:
            dy = (y22 - y11) / (num - 1)

            for i in range(num - 2):
                array[i + 1, 0] = x11
                array[i + 1, 1] = y11 + dy * (i + 1)


        length = np.zeros(num)
        length[0] = 0
        for i in range(1, num):
            if x11 != x22:
                length[i] = (1 + k ** 2) ** 0.5 * abs(array[i, 0] - array[0, 0])
            else:
                length[i] = abs(array[i, 1] - array[0, 1])

        length = length * 1000

        arr = np.zeros(num)  # arr储存离原始点最近的点的编号


        # 最近邻插值
        arr1 = ChangedArr
        num_p = len(arr1)


        for i in range(num):
            distance = np.zeros(num_p - time_c1)
            for j in range(num_p - time_c1):
                distance[j] = (arr1[j, 0] - array[i, 0]) ** 2 + (arr1[j, 1] - array[i, 1]) ** 2

            index = np.argmin(distance)  # 得到距离插值点最近的已知点的索引
            arr[i] = index


        # 判断所选择的点是否在封闭图形内
        arr2 = ChangedArr2
        num_cell = len(arr2)

        flag = np.zeros(num)

        degree = []

        for k in range(num):
            list_cell = []
            for i in range(num_cell):
                for j in range(3):
                    if int(arr2[i, j]) == arr[k]:
                        list_cell.append(i)

            print(list_cell)

            x = array[k, 0]
            y = array[k, 1]

            func1 = 0
            func2 = 0
            func3 = 0

            label1 = 0
            label2 = 0
            label3 = 0
            for i in range(len(list_cell)):
                # 三角形各顶点的编号
                b1 = arr2[list_cell[i], 0]
                b2 = arr2[list_cell[i], 1]
                b3 = arr2[list_cell[i], 2]

                x1 = arr1[int(b1), 0]
                y1 = arr1[int(b1), 1]
                x2 = arr1[int(b2), 0]
                y2 = arr1[int(b2), 1]
                x3 = arr1[int(b3), 0]
                y3 = arr1[int(b3), 1]

                # 求得四个三角形的面积，再作比较，判断输入点是否在三角形内
                s = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)
                s1 = abs((x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)) / 2)
                s2 = abs((x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y)) / 2)
                s3 = abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2)

                if (s - s1 - s2 - s3) * 1e9 >= -1:
                    print(x1, y1, x2, y2, x3, y3)
                    print(s, s1, s2, s3)
                    flag[k] += 1

                    a1 = x2 * y3 - x3 * y2
                    beta1 = y2 - y3
                    c1 = x3 - x2

                    a2 = x3 * y1 - x1 * y3
                    beta2 = y3 - y1
                    c2 = x1 - x3

                    a3 = x1 * y2 - x2 * y1
                    beta3 = y1 - y2
                    c3 = x2 - x1

                    func1 = (a1 + beta1 * x + c1 * y) / (2 * s)
                    func2 = (a2 + beta2 * x + c2 * y) / (2 * s)
                    func3 = (a3 + beta3 * x + c3 * y) / (2 * s)
                    label1 = int(b1)
                    label2 = int(b2)
                    label3 = int(b3)

                    temperature = ''.join([name2, '/', 'temperature', '.txt'])
                    tem = np.loadtxt(temperature)

                    c_want = func1 * tem[label1] + func2 * tem[label2] + func3 * tem[label3]
                    degree.append(c_want)

                    break
                else:
                    pass

            # print(flag)

        sum = 1
        for z in range(num):
            sum = sum * flag[z]

        if sum == 0:
            # 后两项分别为按钮(以|隔开，共有7种按钮类型，见示例后)、默认按钮(省略则默认为第一个按钮)
            QMessageBox.warning(None, "警告", "线段上有点不在图形内！", QMessageBox.Ok)

        if sum != 0:

            # plt.figure(figsize=(8, 6))
            self.ax2.set_title("concentration variation curve")
            self.ax2.set_xlabel("x(mm)")
            self.ax2.set_ylabel("c(wt%)")


            array = array * 1000

            # temperature = ''.join([name2, '/', 'temperature', '.txt'])
            # tem = np.loadtxt(temperature)

            # degree = []
            # for i in range(num):
            #     degree.append(tem[int(arr[i])])

            tpt = np.array(degree)


            # 对碳浓度保留4位小数
            for i in range(tpt.shape[0]):
                tpt[i] = int(tpt[i] * 1000) / 1000

            self.tpt = tpt
            self.length = length

            self.ax2.plot(length, tpt)
            if t0 != tpt.max():
                self.ax2.set_ylim(t0, tpt.max())

            self.photo.draw()

class Ui_OpenPicture(object):
    def setupUi(self, OpenPicture):
        OpenPicture.setObjectName("OpenPicture")
        OpenPicture.resize(622, 622)
        self.horizontalLayout = QtWidgets.QHBoxLayout(OpenPicture)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(OpenPicture)
        self.label.setMinimumSize(QtCore.QSize(600, 600))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)

        self.retranslateUi(OpenPicture)
        QtCore.QMetaObject.connectSlotsByName(OpenPicture)

    def retranslateUi(self, OpenPicture):
        _translate = QtCore.QCoreApplication.translate
        OpenPicture.setWindowTitle(_translate("OpenPicture", "查看图片"))

    def show(self, filename):
        pixmap = QPixmap(filename)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)


class Ui_Login(object):
    def setupUi(self, Login):
        Login.setObjectName("Login")
        Login.resize(400, 377)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(Login)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Login)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(Login)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.label_2 = QtWidgets.QLabel(Login)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.textEdit = QtWidgets.QTextEdit(Login)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(Login)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Login)

        self.pushButton.clicked.connect(self.compare)
        self.pushButton.clicked.connect(Login.close)

        try:
            with open(''.join([os.getcwd(), '/', 'code.txt']), "r") as f:  # 打开文件
                data = f.read()  # 读取文件

            if data == result2:
                self.textEdit.setText(data)

        except:
            pass

        QtCore.QMetaObject.connectSlotsByName(Login)

    def retranslateUi(self, Login):
        _translate = QtCore.QCoreApplication.translate
        Login.setWindowTitle(_translate("Login", "登录"))
        self.label.setText(_translate("Login", "请将下列字符复制并发送邮件至：bigyellowno.1@sjtu.edu.cn"))
        self.label_2.setText(_translate("Login", "在下方输入回复的密钥："))
        self.pushButton.setText(_translate("Login", "OK"))

        self.lineEdit.setText(result)


    def compare(self):
        text = self.textEdit.toPlainText()
        if text == result2:

            with open(''.join([os.getcwd(), '/', 'code.txt']), "w") as file:
                file.write(text)

            global yesORno
            yesORno = 1
        else:

            sys.exit()


class Ui_Form_1(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1353, 607)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        Form.setFont(font)
        Form.setStyleSheet("")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout.addWidget(self.pushButton_6)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.tableWidget = QtWidgets.QTableWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(450, 450))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.tableWidget.setFont(font)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(QtCore.Qt.SolidLine)
        self.tableWidget.setWordWrap(True)
        self.tableWidget.setRowCount(1000)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(120)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(40)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.horizontalLayout_3.addWidget(self.tableWidget)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        # self.widget = QtWidgets.QWidget(Form)
        # self.widget.setMinimumSize(QtCore.QSize(750, 500))
        # self.widget.setStyleSheet("")
        # self.widget.setObjectName("widget")
        # self.gridLayout.addWidget(self.widget, 1, 1, 1, 1)
        self.horizontalLayout_4.addLayout(self.gridLayout)


        # 初始化一个定时器
        self.timer = QTimer()
        # showTime()方法
        self.timer.timeout.connect(self.save)
        self.timer.start(100)


        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(QtCore.QSize(750, 500))
        self.gridLayout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        self.pushButton_4.clicked.connect(self.save)
        self.pushButton_6.clicked.connect(self.draw)


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def save(self):

        global Dc_matrix

        num = 0
        while self.tableWidget.item(num, 0) is not None:
            num += 1

        try:
            Dc_matrix = np.zeros((num, 2))
            for i in range(2):
                for j in range(num):
                    Dc_matrix[j, i] = float(self.tableWidget.item(j, i).text())
        except:
            pass


    def draw(self):

        #print(self.tableWidget.columnCount(),self.tableWidget.rowCount())

        # i为数据个数
        num = 0
        while self.tableWidget.item(num, 0) is not None:
            num += 1

        try:
            t = []
            for i in range(num):
                t.append(float(self.tableWidget.item(i, 0).text()))


            D = []
            for j in range(num):
                D.append(float(self.tableWidget.item(j, 1).text()))


            # 创建一个Matplotlib子图
            self.ax.cla()

            self.ax.set_title('Diffusion coefficient variation curve')
            self.ax.set_xlabel('c(wt%)')
            self.ax.set_ylabel('Dc(m^2/s)')


            # 绘制一个简单的图形
            self.ax.plot(t, D)

            # 更新Matplotlib图形
            self.canvas.draw()

        except:
            pass


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "参数设定"))
        self.pushButton_4.setText(_translate("Form", "保存"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Form", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("Form", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("Form", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("Form", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("Form", "5"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "碳势（%）"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "扩散系数（m^2/s）"))
        self.pushButton_6.setText(_translate("Form", "显示图像"))


        if Dc_matrix.shape[0] == 1:
            pass
        if Dc_matrix.shape[0] != 1:
            # try:
            print(Dc_matrix)
            # self.tableWidget.setRowCount(cf_matrix.shape[0])
            # self.tableWidget.setColumnCount(cf_matrix.shape[1])

            for i in range(Dc_matrix.shape[0]):

                for j in range(Dc_matrix.shape[1]):

                    newItem = QtWidgets.QTableWidgetItem(str(Dc_matrix[i, j]))
                    newItem.setTextAlignment(Qt.AlignHCenter)
                    self.tableWidget.setItem(i, j, newItem)

class Ui_Form_2(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1353, 607)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        Form.setFont(font)
        Form.setStyleSheet("")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout.addWidget(self.pushButton_6)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.tableWidget = QtWidgets.QTableWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(450, 450))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.tableWidget.setFont(font)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(QtCore.Qt.SolidLine)
        self.tableWidget.setWordWrap(True)
        self.tableWidget.setRowCount(1000)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(120)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(40)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.horizontalLayout_3.addWidget(self.tableWidget)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        # self.widget = QtWidgets.QWidget(Form)
        # self.widget.setMinimumSize(QtCore.QSize(750, 500))
        # self.widget.setStyleSheet("")
        # self.widget.setObjectName("widget")
        # self.gridLayout.addWidget(self.widget, 1, 1, 1, 1)
        self.horizontalLayout_4.addLayout(self.gridLayout)


        # 初始化一个定时器
        self.timer = QTimer()
        # showTime()方法
        self.timer.timeout.connect(self.save)
        self.timer.start(100)


        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(QtCore.QSize(750, 500))
        self.gridLayout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        self.pushButton_4.clicked.connect(self.save)
        self.pushButton_6.clicked.connect(self.draw)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def save(self):

        global beta_matrix

        num = 0
        while self.tableWidget.item(num, 0) is not None:
            num += 1

        try:
            beta_matrix = np.zeros((num, 2))
            for i in range(2):
                for j in range(num):
                    beta_matrix[j, i] = float(self.tableWidget.item(j, i).text())
        except:
            pass


    def draw(self):

        # print(self.tableWidget.columnCount(),self.tableWidget.rowCount())

        # i为数据个数
        num = 0
        while self.tableWidget.item(num, 0) is not None:
            num += 1

        try:
            t = []
            for i in range(num):
                t.append(float(self.tableWidget.item(i, 0).text()))

            D = []
            for j in range(num):
                D.append(float(self.tableWidget.item(j, 1).text()))

            # 创建一个Matplotlib子图
            self.ax.cla()

            self.ax.set_title('Transfer coefficient variation curve')
            self.ax.set_xlabel('c(wt%)')
            self.ax.set_ylabel('β(m/s)')

            # 绘制一个简单的图形
            self.ax.plot(t, D)

            # 更新Matplotlib图形
            self.canvas.draw()

        except:
            pass

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "参数设定"))
        self.pushButton_4.setText(_translate("Form", "保存"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Form", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("Form", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("Form", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("Form", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("Form", "5"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "碳势（%）"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "传递系数（m/s）"))
        self.pushButton_6.setText(_translate("Form", "显示图像"))


        if beta_matrix.shape[0] == 1:
            pass
        if beta_matrix.shape[0] != 1:
            # try:
            print(beta_matrix)
            # self.tableWidget.setRowCount(cf_matrix.shape[0])
            # self.tableWidget.setColumnCount(cf_matrix.shape[1])

            for i in range(beta_matrix.shape[0]):

                for j in range(beta_matrix.shape[1]):

                    newItem = QtWidgets.QTableWidgetItem(str(beta_matrix[i, j]))
                    newItem.setTextAlignment(Qt.AlignHCenter)
                    self.tableWidget.setItem(i, j, newItem)


class Ui_Form_3(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1353, 607)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        Form.setFont(font)
        Form.setStyleSheet("")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout.addWidget(self.pushButton_6)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.tableWidget = QtWidgets.QTableWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(450, 450))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.tableWidget.setFont(font)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(QtCore.Qt.SolidLine)
        self.tableWidget.setWordWrap(True)
        self.tableWidget.setRowCount(1000)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(120)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(40)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.horizontalLayout_3.addWidget(self.tableWidget)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        # self.widget = QtWidgets.QWidget(Form)
        # self.widget.setMinimumSize(QtCore.QSize(750, 500))
        # self.widget.setStyleSheet("")
        # self.widget.setObjectName("widget")
        # self.gridLayout.addWidget(self.widget, 1, 1, 1, 1)
        self.horizontalLayout_4.addLayout(self.gridLayout)

        # 初始化一个定时器
        self.timer = QTimer()
        # showTime()方法
        self.timer.timeout.connect(self.save)
        self.timer.start(100)


        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(QtCore.QSize(750, 500))
        self.gridLayout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        self.pushButton_4.clicked.connect(self.save)
        self.pushButton_6.clicked.connect(self.draw)



        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def save(self):

        global cf_matrix

        num = 0
        while self.tableWidget.item(num, 0) is not None:
            num += 1

        try:
            cf_matrix = np.zeros((num, 2))
            for i in range(2):
                for j in range(num):
                    cf_matrix[j, i] = float(self.tableWidget.item(j, i).text())
        except:
            pass


    def draw(self):

        # print(self.tableWidget.columnCount(),self.tableWidget.rowCount())

        # i为数据个数
        num = 0
        while self.tableWidget.item(num, 0) is not None:
            num += 1

        try:
            t = []
            for i in range(num):
                t.append(float(self.tableWidget.item(i, 0).text()))

            D = []
            for j in range(num):
                D.append(float(self.tableWidget.item(j, 1).text()))

            # 创建一个Matplotlib子图
            self.ax.cla()

            self.ax.set_title('Carbon concentration variation curve')
            self.ax.set_xlabel('time(s)')
            self.ax.set_ylabel('c(wt%)')

            # 绘制一个简单的图形
            self.ax.plot(t, D)

            # 更新Matplotlib图形
            self.canvas.draw()

        except:
            pass


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "参数设定"))
        self.pushButton_4.setText(_translate("Form", "保存"))
        item = self.tableWidget.verticalHeaderItem(0)
        # time.sleep(0.1)
        item.setText(_translate("Form", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("Form", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("Form", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("Form", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("Form", "5"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "时间（s）"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "气氛碳势（wt%）"))
        self.pushButton_6.setText(_translate("Form", "显示图像"))

        if cf_matrix.shape[0] == 1:
            pass
        if cf_matrix.shape[0] != 1:
            # try:
            print(cf_matrix)
            # self.tableWidget.setRowCount(cf_matrix.shape[0])
            # self.tableWidget.setColumnCount(cf_matrix.shape[1])

            for i in range(cf_matrix.shape[0]):

                for j in range(cf_matrix.shape[1]):

                    newItem = QtWidgets.QTableWidgetItem(str(cf_matrix[i, j]))
                    newItem.setTextAlignment(Qt.AlignHCenter)
                    self.tableWidget.setItem(i, j, newItem)



if __name__ == '__main__':

    ss = QtWidgets.QDialog()
    login = Ui_Login()
    login.setupUi(ss)
    ss.exec_()


    app = QtWidgets.QApplication(sys.argv)

    MainWindow = MyMainWindow()  # 创建窗体对象
    ui = Ui_MainWindow()                  # 创建pyqt设计的窗体对象
    ui.setupUi(MainWindow)                # 调用pyqt窗体的方法对窗体对象进行初始化设置
    MainWindow.show()                     # 显示窗体

    sys.exit(app.exec_())                 # 程序关闭时退出进程
