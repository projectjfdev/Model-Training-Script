import os


def crear_txt_vacios(ruta_imagenes, ruta_labels):
    os.makedirs(ruta_labels, exist_ok=True)

    archivos_imagenes = [f for f in os.listdir(
        ruta_imagenes) if f.lower().endswith(('.jpg', '.png'))]

    for img in archivos_imagenes:
        nombre_txt = os.path.splitext(img)[0] + '.txt'
        ruta_txt = os.path.join(ruta_labels, nombre_txt)
        if not os.path.exists(ruta_txt):
            with open(ruta_txt, 'w') as f:
                pass  # archivo vacío
            print(f"Creado: {ruta_txt}")
        else:
            print(f"Ya existe: {ruta_txt}")


if __name__ == "__main__":
    carpeta_imagenes = "dataset/images/train"
    carpeta_labels = "dataset/labels/train"

    crear_txt_vacios(carpeta_imagenes, carpeta_labels)
