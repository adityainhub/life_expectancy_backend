import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load and preprocess data
df = pd.read_csv(r'C:\Users\Aditya\Desktop\MACHINE_LEARNING\MLenv\dataset\nilu.csv')

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

# Apply scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.8)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# export the scaler and PCA
joblib.dump(scaler,'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

print("Exported to 'scaler.pkl' and 'pca.pkl'.")

