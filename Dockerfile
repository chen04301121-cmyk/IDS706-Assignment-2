FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY gold_analysis.py overview.py ./
COPY gold_data_2015_25.csv ./

CMD ["python", "overview.py", "--output-dir", "/app/outputs"]