import cv2
import os
import csv
from collections import defaultdict

# CONFIGURAR
ruta_mot = "annotations_mot.txt"  # archivo MOT exportado por CVAT
# carpeta con imágenes (para saber ancho/alto)
ruta_imagenes = "dataset/images"
salida_labels = "dataset/labels"
gt_txt_path = "dataset/gt.txt"

os.makedirs(salida_labels, exist_ok=True)

# Cargar ancho y alto de las imágenes para normalizar


def get_image_size(image_folder):
    sizes = {}
    for img_name in os.listdir(image_folder):
        if not img_name.endswith((".jpg", ".png")):
            continue
        path = os.path.join(image_folder, img_name)
        img = cv2.imread(path)
        if img is None:
            continue
        sizes[img_name] = (img.shape[1], img.shape[0])  # ancho, alto
    return sizes


sizes = get_image_size(ruta_imagenes)

# Organizar anotaciones por frame
frames = defaultdict(list)
with open(ruta_mot, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # row: frame, id, bb_left, bb_top, bb_width, bb_height, class, visibility
        frame_num = int(row[0])
        track_id = int(row[1])
        bb_left = float(row[2])
        bb_top = float(row[3])
        bb_w = float(row[4])
        bb_h = float(row[5])
        cls = row[6]  # suponemos 'vaca' o similar

        frames[frame_num].append({
            'id': track_id,
            'bbox': (bb_left, bb_top, bb_w, bb_h),
            'class': cls
        })

# Procesar y guardar archivos YOLO y gt.txt
with open(gt_txt_path, "w") as gt_file:
    for frame_num, objs in frames.items():
        # Nombre imagen (asumiendo frame_num coincide con nombre: frame_0001.jpg)
        img_name = f"frame_{frame_num - 1:04d}.jpg"  # ajustar según tu naming
        if img_name not in sizes:
            print(
                f"Imagen {img_name} no encontrada, se salta frame {frame_num}")
            continue
        w_img, h_img = sizes[img_name]

        label_path = os.path.join(
            salida_labels, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for obj in objs:
                x, y, bw, bh = obj['bbox']

                # Convertir bbox a formato YOLO (x_center, y_center, w, h) normalizado
                x_c = (x + bw / 2) / w_img
                y_c = (y + bh / 2) / h_img
                w_norm = bw / w_img
                h_norm = bh / h_img

                class_id = 0  # si solo vacas, usar 0
                f.write(
                    f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                # Escribir línea para gt.txt [frame, id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1]
                gt_file.write(
                    f"{frame_num} {obj['id']} {int(x)} {int(y)} {int(bw)} {int(bh)} -1 -1 -1 -1\n")

print("Conversión completada.")
