import numpy as np

W = np.array([(0.01, -0.05, 0.1, 0.05), (0.7, 0.2, 0.05, 0.16), (0.0, -0.45, -0.2, 0.03)])  # Weights
xi = np.array([-15, 22, -44, 56])  # Input
b = np.array([0.0, 0.2, -0.3])  # Bias

delta = 1

Wxi = np.dot(W, xi)

y = np.sum((Wxi, b), axis=0)

ey = np.exp(y)
p = ey / np.sum(ey)

final = -np.log(p[p.size - 1])

print(final)

# final = 1.0401905694301092
