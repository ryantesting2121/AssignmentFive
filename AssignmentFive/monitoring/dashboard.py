import streamlit as st
import pandas as pd
import json
import os
import altair as alt

LOG_FILE = "/logs/prediction_logs.json"
DATA_FILE = "/app/IMDB_Dataset.csv"

st.title("📈 Model Monitoring Dashboard")

if not os.path.exists(LOG_FILE):
    st.warning("No logs yet. Make some predictions first.")
    st.stop()

# Read logs
with open(LOG_FILE, "r") as f:
    logs = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(logs)
if df.empty:
    st.warning("Log file is empty.")
    st.stop()

# Data Drift: sentence lengths
dataset = pd.read_csv(DATA_FILE)
dataset["length"] = dataset["review"].apply(lambda x: len(str(x).split()))
df["length"] = df["request_text"].apply(lambda x: len(str(x).split()))

st.subheader("Data Drift: Sentence Length Distribution")
chart = alt.Chart(df).transform_density(
    "length", as_=["length", "density"]
).mark_area(opacity=0.5, color="red").encode(
    x="length:Q", y="density:Q"
)

chart2 = alt.Chart(dataset).transform_density(
    "length", as_=["length", "density"]
).mark_area(opacity=0.5, color="blue").encode(
    x="length:Q", y="density:Q"
)

st.altair_chart(chart + chart2, use_container_width=True)
st.caption("🔵 Blue = Training data | 🔴 Red = Live data")

# Target Drift
st.subheader("Target Drift: Predicted vs True Sentiments")
pred_dist = df["predicted_sentiment"].value_counts().reset_index()
pred_dist.columns = ["sentiment", "count"]

true_dist = df["true_sentiment"].value_counts().reset_index()
true_dist.columns = ["sentiment", "count"]

col1, col2 = st.columns(2)
col1.bar_chart(pred_dist.set_index("sentiment"))
col2.bar_chart(true_dist.set_index("sentiment"))

# Model accuracy and precision
correct = (df["predicted_sentiment"] == df["true_sentiment"]).sum()
accuracy = correct / len(df)
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

from sklearn.metrics import precision_score
precision = precision_score(df["true_sentiment"], df["predicted_sentiment"], average='macro')
st.metric("Model Precision", f"{precision*100:.2f}%")

# Alert if accuracy below threshold
if accuracy < 0.8:
    st.error("⚠️ ALERT: Model accuracy has dropped below 80%!")

st.success("Dashboard loaded successfully ✅")



