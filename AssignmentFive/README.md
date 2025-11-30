Assignment 5 – Model Monitoring

This project sets up a two-container system for serving a sentiment analysis model and monitoring its performance in real time. One container runs a FastAPI prediction service, and the second container runs a Streamlit dashboard that reads the prediction logs and shows drift, accuracy, and alerts.

System Architecture

The setup has two separate services:

1. FastAPI Prediction Service

Runs the /predict endpoint.

Every request gets logged to a shared file:
logs/prediction_logs.json

Each log entry includes:

timestamp

request_text

predicted_sentiment

true_sentiment (supplied in the request)

2. Streamlit Monitoring Dashboard

Reads the same prediction_logs.json file from the shared Docker volume.

Shows:

Data drift (sentence length distribution)

Target drift (predicted sentiment distribution)

Accuracy + precision from user feedback

Shows an alert banner if accuracy drops below 80%.

Both containers share:

A Docker network (so they can communicate)

A Docker volume (so they can access the same logs)

Project Structure
.
├── api/
│   ├── main.py
│   ├── sentiment_model.pkl
│   └── Dockerfile
├── monitoring/
│   ├── dashboard.py
│   ├── IMDB Dataset.csv
│   └── Dockerfile
├── evaluate.py
├── test.json
└── Makefile

How to Run Everything
1. Build the images
make build

2. Run both containers

This creates the network + volume and starts the API and dashboard:

make run

3. Test the API (example curl request)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The movie was amazing!", "true_sentiment": "positive"}'

4. Open the monitoring dashboard

Go to:
http://localhost:8501

This will show drift plots, prediction stats, and the alert if accuracy is too low.

5. Evaluate the API using the script

The evaluate.py script sends all items from test.json to the API and prints accuracy.

Run it with:

python evaluate.py

Cleaning Up

To stop containers and remove the network/volume:

make clean

Summary

This project shows how to package a simple MLOps workflow using Docker.
The FastAPI service handles predictions and logs everything, while the Streamlit dashboard monitors model behavior using those logs. The Makefile handles all Docker commands so the whole system can be run with just a few simple commands.
