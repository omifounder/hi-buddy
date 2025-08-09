FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 libgl1 git && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8765
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]
