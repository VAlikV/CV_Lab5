# Используем официальный образ PyTorch (CPU)
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Устанавливаем рабочую директорию
WORKDIR /app
# WORKDIR /app/output

ENV TYPE=MiDaS_small

# Копируем файлы проекта в контейнер
COPY requirements.txt .
COPY Lab5_N3.py .
# COPY horse.mp4 .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Определяем команду запуска контейнера
CMD ["python", "Lab5_N3.py", "MiDaS_small", "video.mp4"]

# sudo docker run --rm -v $(pwd)/horse.mp4:/app/video.mp4 -v $(pwd)/output:/app/output video-processor