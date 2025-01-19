import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess data
df = pd.read_csv(r'C:\Users\Aditya\Desktop\MACHINE_LEARNING\MLenv\dataset\life_expectancy.csv')

# Handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

unique_countries = df['Country'].unique()

# Print all unique country names
# print("Unique countries present in the dataset:")
# for country in unique_countries:
#     print(country)


# Impute numeric columns with mean
num_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

# Encode categorical variables
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
   


# Separate features and target
target = df['Life expectancy']
features = df.drop(['Life expectancy'], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Apply PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.8)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train the model
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Predictions
y_pred = model.predict(X_test_pca)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Export the model using joblib
joblib.dump(model, 'linear_regression_model.pkl')

# Output evaluation metrics
print(f"Linear Regression - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# # print(features.head())
# unique_countries = df['Country'].unique()

# # Print all unique country names
# print("Unique countries present in the dataset:")
# for country in unique_countries:
#     print(country)

