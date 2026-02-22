import json
import numpy as np

# A minimal crossed layout model: y ~ 1 + (1|A) + (1|B)
# 4 observations, 2 levels of A, 2 levels of B
# A = [0, 0, 1, 1]
# B = [0, 1, 0, 1]

y = np.array([2.5, 3.1, 2.1, 4.0])
X = np.ones((4, 1))

# Z_A (2 columns)
Z_A = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1]
])

# Z_B (2 columns)
Z_B = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
])

# Z is horizontal concatenation
Z = np.hstack([Z_A, Z_B])
Zt = Z.T

# Mock theta: [theta_A, theta_B]
theta = np.array([0.5, 0.8])

data = {
    "model": "y ~ 1 + (1 | A) + (1 | B)",
    "inputs": {
        "X": X.tolist(),
        "Zt": Zt.tolist(),
        "y": y.tolist()
    },
    "outputs": {
        "theta": theta.tolist()
    }
}

with open("tests/data/mock_crossed.json", "w") as f:
    json.dump(data, f, indent=2)

print("Mock crossed dataset generated.")
