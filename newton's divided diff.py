import numpy as np

def newton_divided_difference(x, y):
    """
    Implement Newton's Divided Difference method for polynomial interpolation.
    
    Parameters:
    x: list or array of x coordinates
    y: list or array of y coordinates
    
    Returns:
    coefficients: list of coefficients for the interpolating polynomial
    """
    n = len(x)
    coefficients = [y[0]]  # The first coefficient is always y[0]
    
    # Create a matrix to store divided differences
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i+1][j-1] - divided_diff[i][j-1]) / (x[i+j] - x[i])
    
    # The coefficients are the first row of the divided difference matrix
    coefficients = [divided_diff[0][j] for j in range(n)]
    
    return coefficients

def newton_polynomial(coeffs, x_data, x):
    """
    Evaluate the Newton polynomial at point x.
    
    Parameters:
    coeffs: coefficients from newton_divided_difference
    x_data: original x points
    x: point at which to evaluate the polynomial
    
    Returns:
    y: value of the polynomial at x
    """
    n = len(x_data) - 1
    p = coeffs[n]
    for k in range(1, n + 1):
        p = coeffs[n-k] + (x - x_data[n-k])*p
    return p

# Given data points
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

# Compute the coefficients
coeffs = newton_divided_difference(x, y)

print("Coefficients of the Newton polynomial:")
for i, coeff in enumerate(coeffs):
    print(f"a{i}: {coeff}")

# Verify the interpolation
print("\nVerifying interpolation:")
for xi, yi in zip(x, y):
    y_interp = newton_polynomial(coeffs, x, xi)
    print(f"f({xi}) = {y_interp:.6f}, Original y = {yi}")

# Interpolate at a new point
x_new = 2.5
y_new = newton_polynomial(coeffs, x, x_new)
print(f"\nInterpolated value at x = {x_new}: {y_new:.6f}")