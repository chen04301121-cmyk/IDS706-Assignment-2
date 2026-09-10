import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Import the dataset
gold = pd.read_csv('gold_data_2015_25.csv')

# Inspect the dataset
print("First five rows:")
print(gold.head()) # quick overview

print("\nData info:")
gold.info()

print("\nSummary statistics:")
print(gold.describe()) # understand data types and summary statistics

print("\nMissing values:")
print(gold.isnull().sum()) # check missing values

print("\nNumber of duplicate rows:")
print(gold.duplicated().sum()) # check for duplicates

print("\nData types:")
print(gold.dtypes) # check data types of each column

# Basic Filtering and Grouping

# Convert Date from text to datetime
gold["Date"] = pd.to_datetime(gold["Date"], errors="coerce")

# Create a Year column for grouping
gold["Year"] = gold["Date"].dt.year

# Calculate the overall average GLD price
average_gld = gold["GLD"].mean()

# Filter rows where GLD is above its overall average
above_average = gold[gold["GLD"] > average_gld]

print("\nOverall average GLD price:")
print(round(average_gld, 2))

print("\nFirst five rows where GLD is above average:")
print(above_average.head())

print("\nNumber of rows where GLD is above average:")
print(len(above_average))

# Group the data by year and calculate summary statistics
yearly_summary = gold.groupby("Year")["GLD"].agg(
    average_price="mean",
    minimum_price="min",
    maximum_price="max",
    observation_count="count"
)

print("\nYearly GLD summary:")
print(yearly_summary.round(2))


# Visualization

# Sort observations by date before plotting
gold = gold.sort_values("Date")

plt.figure(figsize=(12, 6))

plt.plot(
    gold["Date"],
    gold["GLD"],
    color="goldenrod",
    linewidth=1.5
)

plt.title("GLD Price Trend from 2015 to 2025")
plt.xlabel("Date")
plt.ylabel("GLD Price")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the plot as an image
plt.savefig("gld_price_trend.png", dpi=300)

# Display the plot
plt.show()


# Machine Learning: Linear Regression

# Arrange observations in chronological order
gold = gold.sort_values("Date").reset_index(drop=True)

# Select the input features and output variable
features = ["SPX", "USO", "SLV", "EUR/USD"]
target = "GLD"

X = gold[features]
y = gold[target]

# Use the first 80% for training and the last 20% for testing
split_index = int(len(gold) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining observations:")
print(len(X_train))

print("\nTesting observations:")
print(len(X_test))

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate estimates for the test period
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("\nLinear Regression Results:")
print("Mean Absolute Error:", round(mae, 2))
print("R-squared:", round(r_squared, 3))

# Display model coefficients
coefficients = pd.DataFrame({
    "Variable": features,
    "Coefficient": model.coef_
})

print("\nModel coefficients:")
print(coefficients.round(3))

print("\nModel intercept:")
print(round(model.intercept_, 3))


# Compare actual and estimated GLD prices

test_dates = gold["Date"].iloc[split_index:]

plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    y_test,
    label="Actual GLD",
    color="goldenrod"
)

plt.plot(
    test_dates,
    y_pred,
    label="Estimated GLD",
    color="navy",
    linestyle="--"
)

plt.title("Actual vs. Estimated GLD Prices")
plt.xlabel("Date")
plt.ylabel("GLD Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("actual_vs_estimated_gld.png", dpi=300)
plt.show()