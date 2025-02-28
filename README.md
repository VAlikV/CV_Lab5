## Для билда:
``` bash
sudo docker build -t video-processor .
```

## Для запуска:
``` bash
sudo docker run --rm -v $(pwd)/<FILE_NAME>.mp4:/app/video.mp4 -v $(pwd)/output:/app/output video-processor
```

## Пример:
``` bash
sudo docker run --rm -v $(pwd)/VID.mp4:/app/video.mp4 -v $(pwd)/output:/app/output video-processor
```

## Результат
Будет создана папка output в которой будут сохранены видео с картой глубин
