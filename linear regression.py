import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data (replace with your own data)
x = np.array([10,13,15,17,19])
y = np.array([2, 4, 5, 4, 5])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(x.reshape(-1, 1), y)  # Reshape x for compatibility

# Print the coefficients (slope and intercept)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Make predictions
predicted_y = model.predict(np.array([6]))  # Predict for a new data point
print("Predicted value for x=6:", predicted_y[0])