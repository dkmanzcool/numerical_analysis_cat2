import numpy as np

def lagrange_polynomial(x, y):
    n = len(x)
    
    def L(k, x):
        """Compute the k-th Lagrange polynomial at x."""
        L = np.ones_like(x)
        for j in range(n):
            if j != k:
                L *= (x - x[j]) / (x[k] - x[j])
        return L
    
    def P(x):
        """Compute the Lagrange interpolation polynomial at x."""
        return sum(y[k] * L(k, x) for k in range(n))
    
    # Compute coefficients by evaluating P at a set of points
    x_eval = np.linspace(min(x), max(x), 100)
    y_eval = P(x_eval)
    
    # Fit a polynomial to these points to get the coefficients
    coeffs = np.polyfit(x_eval, y_eval, n-1)
    
    return coeffs[::-1]  # Reverse to get ascending order of powers

# Given data points
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

# Compute the coefficients
coefficients = lagrange_polynomial(x, y)

print("Coefficients of the Lagrange polynomial (in ascending order of powers):")
for i, coeff in enumerate(coefficients):
    print(f"x^{i}: {coeff}")

def polynomial(x, coeffs):
    return sum(coeff * x**i for i, coeff in enumerate(coeffs))

for xi, yi in zip(x, y):
    print(f"f({xi}) = {polynomial(xi, coefficients):.6f}, Original y = {yi}")