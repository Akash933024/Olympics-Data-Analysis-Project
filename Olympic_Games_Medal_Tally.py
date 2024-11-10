import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
medal_tally = pd.read_csv("C:\\Users\\khila\\OneDrive\\Documents\\Business Intelligence II\\Project\\Olympic_Games_Medal_Tally\\Olympic_Games_Medal_Tally.csv")

# Feature Engineering: Add a feature for previous medals won by each country
medal_tally['previous_medals'] = medal_tally.groupby('country')['total'].shift(1).fillna(0)

# Features and Target
features = ['year', 'gold', 'silver', 'bronze', 'previous_medals']
X = medal_tally[features]
y = medal_tally['total']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Save the model and predictions
medal_tally['predicted_medals'] = model.predict(X_scaled)
medal_tally.to_csv('medal_tally_predictions_rf.csv', index=False)
