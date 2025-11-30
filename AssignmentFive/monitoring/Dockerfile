FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir streamlit pandas numpy matplotlib seaborn

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]
