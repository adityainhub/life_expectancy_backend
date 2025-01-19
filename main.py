from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and encoders
try:
    linear_model = joblib.load(r'.\models\linear_regression_model.pkl')
    gradient_model = joblib.load(r'.\models\gradient_boosting_model.pkl')
    random_forest_model = joblib.load(r'.\models\random_forest_model.pkl')
    ensemble_model = joblib.load(r'.\models\ensemble_model.pkl')
    
    # Load preprocessing objects
    scaler = joblib.load(r'.\scaling_encoding\scaler.pkl')
    pca = joblib.load(r'.\scaling_encoding\pca.pkl')
    country_encoder = joblib.load(r'.\scaling_encoding\country_encoder.pkl')
    gender_encoder = joblib.load(r'.\scaling_encoding\gender_encoder.pkl')
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# List of valid countries (should match your training data)
COUNTRIES = [
    "Afghanistan", "Africa Eastern and Southern", "Africa Western and Central", "Albania",
    "Algeria", "Angola", "Antigua and Barbuda", "Arab World", "Argentina", "Armenia",
    "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas, The", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam",
    "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada",
    "Caribbean small states", "Cayman Islands", "Central African Republic", "Central Europe and the Baltics",
    "Chad", "Channel Islands", "Chile", "China", "Colombia", "Comoros", "Congo, Dem. Rep.",
    "Congo, Rep.", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Curacao", "Cyprus",
    "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Early-demographic dividend", "East Asia & Pacific", "East Asia & Pacific (IDA & IBRD countries)",
    "East Asia & Pacific (excluding high income)", "Ecuador", "Egypt, Arab Rep.", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Euro area",
    "Europe & Central Asia", "Europe & Central Asia (IDA & IBRD countries)",
    "Europe & Central Asia (excluding high income)", "European Union", "Faroe Islands", "Fiji",
    "Finland", "Fragile and conflict affected situations", "France", "French Polynesia", "Gabon",
    "Gambia, The", "Georgia", "Germany", "Ghana", "Greece", "Greenland", "Grenada", "Guam",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Heavily indebted poor countries (HIPC)",
    "High income", "Honduras", "Hong Kong SAR, China", "Hungary", "IBRD only", "IDA & IBRD total",
    "IDA blend", "IDA only", "IDA total", "Iceland", "India", "Indonesia", "Iran, Islamic Rep.",
    "Iraq", "Ireland", "Isle of Man", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kiribati", "Korea, Dem. People's Rep.", "Korea, Rep.", "Kosovo", "Kuwait",
    "Kyrgyz Republic", "Lao PDR", "Late-demographic dividend", "Latin America & Caribbean",
    "Latin America & Caribbean (excluding high income)", "Latin America & the Caribbean (IDA & IBRD countries)",
    "Latvia", "Least developed countries: UN classification", "Lebanon", "Lesotho", "Liberia",
    "Libya", "Liechtenstein", "Lithuania", "Low & middle income", "Low income", "Lower middle income",
    "Luxembourg", "Macao SAR, China", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali",
    "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia, Fed. Sts.",
    "Middle East & North Africa", "Middle East & North Africa (IDA & IBRD countries)",
    "Middle East & North Africa (excluding high income)", "Middle income", "Moldova", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands",
    "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North America",
    "North Macedonia", "Norway", "OECD members", "Oman", "Other small states", "Pacific island small states",
    "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Post-demographic dividend", "Pre-demographic dividend", "Puerto Rico", "Qatar",
    "Romania", "Russian Federation", "Rwanda", "Samoa", "San Marino", "Sao Tome and Principe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore",
    "Sint Maarten (Dutch part)", "Slovak Republic", "Slovenia", "Small states", "Solomon Islands",
    "Somalia", "South Africa", "South Asia", "South Asia (IDA & IBRD)", "South Sudan", "Spain",
    "Sri Lanka", "St. Kitts and Nevis", "St. Lucia", "St. Martin (French part)",
    "St. Vincent and the Grenadines", "Sub-Saharan Africa", "Sub-Saharan Africa (IDA & IBRD countries)",
    "Sub-Saharan Africa (excluding high income)", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syrian Arab Republic", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Upper middle income", "Uruguay",
    "Uzbekistan", "Vanuatu", "Venezuela, RB", "Vietnam", "Virgin Islands (U.S.)", "West Bank and Gaza",
    "World", "Yemen, Rep.", "Zambia", "Zimbabwe"
]


GENDERS = ["Male", "Female"]

class PredictionInput(BaseModel):
    year: str
    country: str
    gender: str
    tuberculosisTreatment: str
    hospitalBeds: str
    urbanPopulation: str
    ruralPopulation: str
    gdp: str
    model: str

def encode_country(Country: str) -> int:
    try:
        if Country not in country_encoder.classes_:
            raise ValueError(f"Country '{Country}' not found in training data")
        encoded_value = country_encoder.transform([Country])[0]
        logger.info(f"Encoded country '{Country}' to {encoded_value}")
        return encoded_value
    except Exception as e:
        logger.error(f"Error encoding country: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid country name: {Country}")

def encode_gender(Gender: str) -> int:
    try:
        if Gender.lower() not in [g.lower() for g in gender_encoder.classes_]:
            raise ValueError(f"Gender must be one of: {list(gender_encoder.classes_)}")
        encoded_value = gender_encoder.transform([Gender])[0]
        logger.info(f"Encoded gender '{Gender}' to {encoded_value}")
        return encoded_value
    except Exception as e:
        logger.error(f"Error encoding gender: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid gender. Must be one of: {list(gender_encoder.classes_)}")

def create_full_feature_array(input_data: PredictionInput):
    """Create a full feature array with dummy values for non-input features."""
    try:
        # Initialize array with zeros for all 21 features
        full_features = np.zeros(21)
        
        # Encode categorical variables
        country_code = encode_country(input_data.country)
        gender_code = encode_gender(input_data.gender)
        
        logger.info(f"Encoded values - Country: {country_code}, Gender: {gender_code}")
        
        # Map input values to their correct positions (0-based indexing)
        feature_mapping = {
            'country': 0,           # Country
            'year': 1,             # Year
            'gender': 2,           # Gender
            'gdp': 5,              # GDP
            'hospitalBeds': 14,    # Hospital Beds
            'tuberculosisTreatment': 16,  # Tuberculosis Treatment
            'urbanPopulation': 17,  # Urban Population
            'ruralPopulation': 18   # Rural Population
        }
        
        # Fill in the values at their correct positions
        full_features[feature_mapping['country']] = country_code
        full_features[feature_mapping['year']] = float(input_data.year)
        full_features[feature_mapping['gender']] = gender_code
        full_features[feature_mapping['gdp']] = float(input_data.gdp)
        full_features[feature_mapping['hospitalBeds']] = float(input_data.hospitalBeds)
        full_features[feature_mapping['tuberculosisTreatment']] = float(input_data.tuberculosisTreatment)
        full_features[feature_mapping['urbanPopulation']] = float(input_data.urbanPopulation)
        full_features[feature_mapping['ruralPopulation']] = float(input_data.ruralPopulation)
        
        logger.info(f"Created feature array: {full_features}")
        return full_features.reshape(1, -1)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")

def preprocess_input(input_data: PredictionInput):
    try:
        # Create full feature array with dummy values
        features = create_full_feature_array(input_data)
        logger.info(f"Created feature array shape: {features.shape}")
        
        # Scale all features using the same scaler as during training
        features_scaled = scaler.transform(features)
        logger.info(f"Scaled features shape: {features_scaled.shape}")
        logger.info(f"Scaled features: {features_scaled}")
        
        # Apply PCA transformation using the same PCA as during training
        features_pca = pca.transform(features_scaled)
        logger.info(f"PCA transformed features shape: {features_pca.shape}")
        logger.info(f"PCA features: {features_pca}")
        
        return features_pca
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Preprocess the input data
        features = preprocess_input(input_data)
        
        # Select the appropriate model
        model_map = {
            'linear': linear_model,
            'gradient': gradient_model,
            'random_forest': random_forest_model,
            'ensemble': ensemble_model
        }
        
        selected_model = model_map.get(input_data.model)
        if not selected_model:
            raise HTTPException(status_code=400, detail="Invalid model selection")
        
        logger.info(f"Using model: {input_data.model}")
        
        try:
            # Make prediction
            prediction = selected_model.predict(features)[0]
            logger.info(f"Prediction successful: {prediction}")
            return {"prediction": float(prediction)}
        except Exception as model_error:
            logger.error(f"Model prediction error: {str(model_error)}")
            logger.error(f"Model type: {type(selected_model)}")
            logger.error(f"Features shape: {features.shape}")
            raise HTTPException(
                status_code=500, 
                detail=f"Model prediction failed: {str(model_error)}"
            )
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/countries")
async def get_countries():
    return {"countries": COUNTRIES}