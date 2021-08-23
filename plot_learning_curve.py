import numpy as np
import matplotlib.pyplot as plt

x = np.array([300, 600, 900, 1200, 1500, 1800, 2100, 2400])
y = np.array([0.0080, 0.0091, 0.0104, 0.0072, 0.0060, 0.0030, 0.0032, 0.0029])

#plt.figure(figsize=(10,8))
plt.plot(x, y)
plt.title("Bounding Box Regression")
plt.xlabel("Number of Images")
plt.ylabel("Mean Squared Error")

plt.show()


j = np.array([300, 600, 900, 1200, 1500, 1800, 2100, 2400])
k = np.array([0.94, 0.91, 0.89, 0.93, 0.95, 0.95, 0.97, 0.96])

#plt.figure(figsize=(10,8))
plt.plot(j, k)
plt.title("Model Classification")
plt.xlabel("Number of Images")
plt.ylabel("Accuracy")

plt.show()