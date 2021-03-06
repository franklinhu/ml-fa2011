import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1, 2, 2])
y1 = np.array([1, 2, 0])

x2 = np.array([0, 1, 0])
y2 = np.array([0, 0, 1])

plt.plot(x1, y1, "r+", x2, y2, 'bo')
plt.axis([-0.5, 2.5, -0.5, 2.5])
plt.show()
