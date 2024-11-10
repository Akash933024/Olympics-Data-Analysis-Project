import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
results = pd.read_csv("C:\\Users\\khila\\OneDrive\\Documents\\Business Intelligence II\\Project\\Olympic_Results.csv", encoding='ISO-8859-1')

# Preprocessing
results.dropna(inplace=True)

# Feature Selection
X = results[['edition_id', 'sport']]
y = results['result_detail']  # Assuming this column has categorical outcomes

# Encoding categorical variables
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the predictions
results['predicted_outcome'] = model.predict(pd.get_dummies(X))
results.to_csv('event_outcome_predictions.csv', index=False)