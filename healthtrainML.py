import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('C:/Users/LENOVO/Downloads/insurance.csv')

# Define features and target
X = df.drop(columns='charges')
y = df['charges']

# Define categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

# Known categories for categorical features
sex_categories = ['male', 'female']
smoker_categories = ['yes', 'no']
region_categories = ['northeast', 'northwest', 'southeast', 'southwest']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(categories=[sex_categories, smoker_categories, region_categories]), categorical_features)
    ])

# Define the Gradient Boosting Regressor
gbr = GradientBoostingRegressor()

# Create a pipeline that first transforms the data and then fits the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Name: Gradient Boosting")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model
with open('health_gbr.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

