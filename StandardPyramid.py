import matplotlib.pyplot as plt

fig = plt.figure()  # 创建一个图
ax = fig.add_subplot(111, projection='3d')
X = [-1, -1, 1, 1, 0]
Y = [-1, 1, -1, 1, 0]
Z = [0, 0, 0, 0, 1]
ax.scatter3D(X, Y, Z)
ax.plot_trisurf(X, Y, Z, color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Standard Pyramid')
plt.show()
