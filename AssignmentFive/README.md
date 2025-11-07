# Assignment 5 - Model Monitoring

This project implements a two-container MLOps system for monitoring a sentiment analysis model using **FastAPI**, **Streamlit**, and **Docker**.

---

# System Architecture

- **FastAPI Prediction Service**  
  Serves sentiment predictions at `/predict`.  
  Each request is logged to `/logs/prediction_logs.json`.

- **Streamlit Monitoring Dashboard**  
  Reads the shared `/logs/prediction_logs.json` to visualize:
  - Data Drift (sentence length distributions)
  - Target Drift (predicted vs true label distributions)
  - Model Accuracy and Precision
  - An alert if accuracy drops below 80%

Both containers share a Docker volume (`monitor_logs`) and communicate over a Docker network** (`monitor_net`).

---

## Project Structure

---

## Running the Project

### 1. Build and Run Containers
```bash
# Build Docker images and create network/volume
make build

# Run the API and dashboard containers
make run


#Stop containers:
make clean





#curl Exmaple:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The movie was amazing!", "true_sentiment": "positive"}'

#Expected Response
#All of the logs are written to /logs/prediction_logs.json


#To evaluate model performance:
python evaluate.py


#All Steps (Repeat of the above):

# Build images and create network/volume
make build

# Run containers
make run

# Test API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The movie was amazing!", "true_sentiment": "positive"}'

# Open the Streamlit dashboard in browser
# http://localhost:8501

# Run evaluation script
python evaluate.py

# Clean up everything
make clean
