import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("data/housing.csv")

# Remove missing values
df = df.dropna()

print("Encoding categorical column...")

# Convert ocean_proximity to numeric
df = pd.get_dummies(df, columns=["ocean_proximity"])

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("Evaluating model...")

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Training completed")
print("Mean Squared Error:", mse)