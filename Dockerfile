FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tinygpt.py data.txt ./
CMD ["python","tinygpt.py","train","--steps","1000"]
