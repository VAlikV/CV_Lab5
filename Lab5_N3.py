import os
import cv2
import torch
import numpy as np
import sys

def main(model_type = "MiDaS_small", filename = 'video.mp4'):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_types = ["MiDaS_small", "DPT_Large", "DPT_Hybrid"]

    if model_type not in model_types:
        print("*** ОШИБКА НАЗВАНИЯ МОДЕЛИ ***")
        return -1

    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    midas.to(DEVICE)
    midas.eval()

    # # Load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    
    if not os.path.isfile(filename):
        print("*** ОШИБКА ИМЕНИ ФАЙЛА ***")
        return -1

    margin_width = 50

    print("*** НАЧАЛО РАБОТЫ ***")

    raw_video = cv2.VideoCapture(filename)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    output_width = frame_width * 2 + margin_width

    filename = os.path.basename(filename)
    output_path = os.path.join('output/depth_' + filename)
    output_path_2 = os.path.join('output/only_depth_' + filename)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
    out_2 = cv2.VideoWriter(output_path_2, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        pre_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        frame = transform(pre_frame).to(DEVICE)

        with torch.no_grad():
            prediction = midas(frame)

            prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=pre_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8)

        combined_frame = cv2.hconcat([raw_frame, split_region, depth_map])
        # combined_frame = cv2.hconcat([depth_map])

        out.write(combined_frame)
        out_2.write(depth_map)

    raw_video.release()
    out.release()
    out_2.release()
    
    print("*** ГОТОВО ***")

if __name__== "__main__":
    main(str(sys.argv[1]), str(sys.argv[2]))