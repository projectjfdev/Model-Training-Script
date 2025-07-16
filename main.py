from ultralytics import YOLO

def entrenar_yolo():
    # Elegí la versión según tu PC: yolov8n.pt (nano) o yolov8s.pt (small)
    modelo = YOLO('yolov8n.pt')

    # Entrenamiento
    modelo.train(
        data='dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=8,  # probá 4 o 8 según tu RAM
        device='cuda'  # o 'cpu' si no tenés GPU
    )

    # Guardar modelo entrenado
    modelo.export(format='onnx')  # opcional
    print("Entrenamiento completo.")


if __name__ == "__main__":
    entrenar_yolo()
