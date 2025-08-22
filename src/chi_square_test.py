import numpy as np
from scipy.stats import chi2_contingency

data = np.array([
    [146, 18],  # Direct
    [147, 17],  # Manual
    [146, 18]   # Auto
])

chi2, p, dof, expected = chi2_contingency(data)

print("Chi-square:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", p)
print("Expected frequencies:\n", expected)