import requests
import json

API_URL = "http://localhost:8000/predict"

with open("test_data.json", "r") as f:
    data = json.load(f)

correct = 0
for item in data:
    resp = requests.post(API_URL, json={"text": item["text"], "true_sentiment": item["true_label"]})
    pred = resp.json().get("predicted_sentiment")
    if pred == item["true_label"]:
        correct += 1

accuracy = correct / len(data)
print(f"Model accuracy on test data: {accuracy*100:.2f}%")
