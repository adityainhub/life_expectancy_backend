# Life Expectancy Predictor API

This is a FastAPI application that serves a machine learning model to predict life expectancy based on various demographic and health indicators.

## Prerequisites

- Python 3.12 or compatible
- pip (Python package installer)

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-name>/backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure your model files are in the correct location:
   - Place your trained model files in the `models` directory
   - Required files:
     - `models/life_expectancy_model.pkl`
     - `models/country_encoder.pkl`
     - `models/gender_encoder.pkl`

## Running the Application

Start the server using uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload`: Enable auto-reload on code changes (development only)
- `--host 0.0.0.0`: Make server accessible from other machines on the network
- `--port 8000`: Run server on port 8000

The API will be available at:
- Local: http://localhost:8000
- Network: http://your-ip-address:8000

## API Documentation

Once the server is running, you can access:
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## API Endpoints

- `GET /countries`: Get list of available countries
- `POST /predict`: Get life expectancy prediction
  - Required JSON payload:
    ```json
    {
      "country": "string",
      "year": "number",
      "gender": "string",
      "gdp": "number",
      "hospitalBeds": "number",
      "tuberculosisTreatment": "number",
      "urbanPopulation": "number",
      "ruralPopulation": "number"
    }
    ```

## Production Deployment

For production:
1. Remove the `--reload` flag
2. Set up a proper process manager (e.g., systemd, supervisor)
3. Use a production ASGI server like Gunicorn with uvicorn workers
4. Set up proper security measures (HTTPS, rate limiting, etc.)

## Error Handling

The API includes proper error handling for:
- Invalid country names
- Invalid gender values
- Missing or invalid input data