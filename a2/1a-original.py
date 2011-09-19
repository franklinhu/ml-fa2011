import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1, -1])
y1 = np.array([1, -1])

x2 = np.array([-1, 1])
y2 = np.array([1, -1])

plt.fill_between([0,1.5],1.5,color="#dddddd")
plt.fill_between([0,-1.5],-1.5,color="#dddddd")

plt.plot(x1, y1, "rx", x2, y2, 'bo')
plt.axis([-1.5, 1.5, -1.5, 1.5])

plt.xlabel('x1')
plt.ylabel('x2')

plt.show()
