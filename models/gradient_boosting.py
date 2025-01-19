import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess data
df = pd.read_csv(r'C:\Users\Aditya\Desktop\MACHINE_LEARNING\MLenv\dataset\life_expectancy.csv')

# Handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

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
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Predictions
y_pred = model.predict(X_test_pca)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Export the model using joblib
joblib.dump(model, 'gradient_boosting_model.pkl')

# Output evaluation metrics
print(f"Gradient Boosting - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# n_components_retained = X_train_pca.shape[1]
# print(f"Number of features (principal components) after PCA: {n_components_retained}")

# # Step 2: Get the feature contributions to the retained principal components
# pca_components = pd.DataFrame(
#     pca.components_, 
#     columns=X_train.columns, 
#     index=[f'PC{i+1}' for i in range(pca.n_components_)]
# )

# # Calculate the overall contribution of each feature across all retained components
# feature_contributions = np.sum(np.abs(pca_components), axis=0)

# # Sort features by total contribution
# sorted_features = pd.Series(feature_contributions, index=X_train.columns).sort_values(ascending=False)

# # Get the top features contributing to PCA
# selected_features = sorted_features.head(n_components_retained).index.tolist()

# # Print the list of features selected by PCA
# print(f"Features selected by PCA (top {n_components_retained}):")
# print(selected_features)

# # Get the data types of the selected features
# selected_features_df = df[selected_features]  # Filter the DataFrame with the selected features
# feature_data_types = selected_features_df.dtypes  # Get data types

# # Print the data types of all selected features
# print("Data types of selected features:")
# print(feature_data_types)


