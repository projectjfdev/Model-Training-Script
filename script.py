import cv2
import os

video_path = os.path.join("assets", "1.mp4")
output_folder = "dataset/images/train"
os.makedirs(output_folder, exist_ok=True)

# Buscar el número más alto ya usado en los archivos
existing_files = [f for f in os.listdir(output_folder) if f.endswith(".jpg")]
existing_indices = [
    int(f.split("_")[1].split(".")[0]) for f in existing_files if "_" in f
]
start_index = max(existing_indices) + 1 if existing_indices else 0

cap = cv2.VideoCapture(video_path)

frame_rate = 0.1  # 1 frame por segundo
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps * frame_rate)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        filename = os.path.join(
            output_folder, f"frame_{start_index + saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Se guardaron {saved_count} nuevas imágenes en '{output_folder}'.")
