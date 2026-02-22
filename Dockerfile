FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirement.txt .

RUN pip install --no-cache-dir -r requirement.txt

# Copy rest of app
COPY . .

# Create model directory
RUN mkdir -p ml_pipeline/models

# Streamlit port
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false"]
