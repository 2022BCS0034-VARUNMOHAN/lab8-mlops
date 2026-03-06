import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("data/housing.csv")

# Example: assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

print("Evaluating model...")
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Model training completed")
print(f"Mean Squared Error: {mse}")