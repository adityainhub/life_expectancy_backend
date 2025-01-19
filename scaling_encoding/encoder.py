from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd


df=pd.read_csv(r'C:\Users\Aditya\Desktop\MACHINE_LEARNING\MLenv\dataset\nilu.csv')

# Initialize LabelEncoders
country_encoder = LabelEncoder()
gender_encoder = LabelEncoder()


# Fit encoders on respective columns
df['country'] = country_encoder.fit_transform(df['Country'])
df['gender'] = gender_encoder.fit_transform(df['Gender'])

# Save the encoders to .pkl files
joblib.dump(country_encoder, 'country_encoder.pkl')
joblib.dump(gender_encoder, 'gender_encoder.pkl')

print("Encoders for 'country' and 'gender' have been exported to 'country_encoder.pkl' and 'gender_encoder.pkl'.")
