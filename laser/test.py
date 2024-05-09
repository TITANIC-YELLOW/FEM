import numpy as np
import matplotlib.pyplot as plt

import read_dat

# cell, coordinate = read_dat.getdata('42crmo_v2.dat')
#
# np.savetxt('cell.txt', cell)
# np.savetxt('coor.txt', coordinate)

coor = np.loadtxt('coor.txt')
x, y, z = coor[:, 0], coor[:, 1], coor[:, 2]
print(np.min(x), np.max(x))
print(np.min(y), np.max(y))
print(np.min(z), np.max(z))

lst = []
for i in range(len(coor)):
    if abs(coor[i, 1]) < 1e-5:
        lst.append(i)

print(len(lst))

surface = np.zeros((len(lst), 4))
for j in range(len(lst)):
    surface[j,0] = lst[j]
    surface[j, 1:4] = coor[lst[j]]

np.savetxt('surface.txt', surface)

# # 设置图表和坐标轴
# fig = plt.figure()  # 创建一个图
# ax = fig.add_subplot(111, projection='3d')
#
# # 创建散点图
# scatter = ax.scatter3D(x, y, z, c = np.ones(len(x)), cmap='jet')
#
# # 设置坐标轴标签
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
#
# plt.show()
