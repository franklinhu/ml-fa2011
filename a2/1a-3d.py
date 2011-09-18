import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter([1, -1], [1, -1], [1, 1], c="r", marker="^")
ax.scatter([1, -1], [-1, 1], [-1, -1], c="b", marker="o")

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x1x2')

plt.show()

