import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\rainfall.csv")

# Check the first few rows of the dataset
print("Dataset Head:")
print(df.head())

# Select only numeric columns for imputation
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Handle missing values by imputing with the mean for numeric columns
imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Check the data after imputing missing values
print("\nData after imputing missing values:")
print(df.head())

# Feature selection - choose relevant features for prediction
# Using the monthly rainfall data as features and the annual rainfall as the target
X = df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]  # Features (Monthly rainfall)
y = df['ANNUAL']  # Target variable (Annual rainfall)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Plot the actual vs predicted rainfall values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Rainfall')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.show()

# Predict rainfall for new data (example)
new_data = pd.DataFrame({
    'JAN': [50.0], 'FEB': [60.0], 'MAR': [70.0], 'APR': [80.0], 'MAY': [90.0], 
    'JUN': [100.0], 'JUL': [110.0], 'AUG': [120.0], 'SEP': [130.0], 'OCT': [140.0],
    'NOV': [150.0], 'DEC': [160.0]
})
predicted_rainfall = model.predict(new_data)
print(f'\nPredicted Annual Rainfall: {predicted_rainfall[0]} mm')
