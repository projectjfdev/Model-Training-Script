# from ultralytics import YOLO


# def entrenar_yolo():
#     # Elegí la versión según tu PC: yolov8n.pt (nano) o yolov8s.pt (small)
#     modelo = YOLO('/models/best.pt')

#     # Entrenamiento
#     modelo.train(
#         data='dataset/dataset.yaml',
#         epochs=100,
#         imgsz=640,
#         batch=8,  # probá 4 o 8 según tu RAM
#         device='cuda'  # o 'cpu' si no tenés GPU
#     )

#     # Guardar modelo entrenado
#     modelo.export(format='onnx')  # opcional
#     print("Entrenamiento completo.")


# if __name__ == "__main__":
#     entrenar_yolo()


from ultralytics import YOLO

# Cargar tu modelo entrenado
model = YOLO("models/best.pt")

# Procesar un video completo
model.predict(
    source="assets/final-1.mp4",  # Cambiá por la ruta a tu video
    conf=0.25,
    imgsz=640,
    device="cpu",  # ← CORREGIDO: usá CPU
    save=True,      # Guarda el video procesado
    show=True       # Muestra en pantalla mientras corre
)
