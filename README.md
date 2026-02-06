# Productivity Prediction API

The **Productivity Prediction API** is a machine learning–powered REST API built with **FastAPI** that provides real-time productivity predictions.

It uses a **pre-trained Logistic Regression model** to evaluate numerical input features representing productivity factors and determines whether the input corresponds to a productive or non-productive state. The model is loaded at runtime, enabling fast and efficient inference without retraining.

FastAPI exposes the model through a clean REST interface, offering high performance, automatic input validation and interactive API documentation. The API returns predictions in JSON format, making it easy to integrate with web applications, mobile apps or other backend services.

The project is designed to be lightweight, scalable and easy to deploy using Docker.

---

## Tech Stack

### Python
- Used to implement the machine learning logic and API endpoints

### FastAPI
- Used to build the REST API
- Provides high performance, automatic request validation, and interactive API documentation
- Handles incoming HTTP requests and returns prediction results as JSON

### NumPy
- Used to load and process the trained model data stored in `model_data.npz`
- Handles numerical operations required for prediction

### Uvicorn
- ASGI server used to run the FastAPI application
- Provides fast and efficient request handling

### Docker
- Used to containerize the application
- Ensures consistent execution across different environments
- Makes deployment easy on any system that supports Docker

---

## Project Structure

```

├── app.py                 # Main FastAPI application and prediction logic
├── model_data.npz         # Pre-trained machine learning model data
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Docker Compose setup
├── .gitignore
└── LICENSE

````

## How the System Works

1. A client sends input data to the API as a JSON request.
2. FastAPI receives and validates the input.
3. The trained machine learning model is loaded from `model_data.npz`.
4. NumPy processes the input features.
5. The model predicts productivity status.
6. The API returns the prediction as a JSON response.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/karunarapolu/productivity-prediction-api.git
cd productivity-prediction-api
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Run Locally

```bash
uvicorn app:app --reload
```

Application will run at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Docker Setup

### Build and Run with Docker

```bash
docker build -t productivity-api .
docker run -p 8000:8000 productivity-api
```

### Using Docker Compose

```bash
docker-compose up
```

---

## API Endpoint

### POST `/predict`

Accepts input features in JSON format.

Example request:

```json
{
  "feature1": 6,
  "feature2": 4,
  "feature3": 9
}
```

Example response:

```json
{
  "prediction": "productive"
}
```

*(Input fields depend on the trained model used.)*

---

## Machine Learning Model

* The model is pre-trained and stored in `model_data.npz`
* Loaded at runtime for fast inference
* Can be replaced with a newly trained model without changing API logic

---

## Use Cases

* Productivity analysis tools
* Personal performance tracking
* Learning project for ML + API integration
* Backend service for productivity dashboards

---

## License

This project is licensed under the **MIT License**.
