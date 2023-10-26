import os
import cv2
import numpy as np

# Definir las rutas de los conjuntos de datos de entrenamiento y prueba
train_dir = "./Dataset mango/train"
test_dir = "./Dataset mango/test1"

# Crear los directorios de entrenamiento y prueba
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Definir el tamaño al que redimensionar las imágenes (por ejemplo, 128x128)
image_size = (128, 128)

# Cargar las imágenes y convertirlas a RGB
images = []
for i in range(1, 51):
    img_path = os.path.join("./Dataset mango", f"Figura_{i}.jpg")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convierte a RGB
            img_resized = cv2.resize(img_rgb, image_size)  # Redimensiona la imagen
            images.append(img_resized)
        else:
            print(f"Error al cargar la imagen {img_path}")
    else:
        print(f"La imagen {img_path} no existe.")

# Etiquetar las imágenes del conjunto de entrenamiento
train_images = images[:40]  # Las primeras 40 imágenes para entrenamiento
train_labels = []
for i in range(40):
    if i % 2 == 0:
        train_labels.append("verde")
    else:
        train_labels.append("maduro")

# Etiquetar las imágenes del conjunto de prueba
test_images = images[40:]  # Las últimas 10 imágenes para prueba
test_labels = []
for i in range(10):
    if i % 2 == 0:
        test_labels.append("verde")
    else:
        test_labels.append("maduro")

# Guardar los datos de entrenamiento y prueba
np.save(os.path.join(train_dir, "images.npy"), train_images)
np.save(os.path.join(train_dir, "labels.npy"), train_labels)
np.save(os.path.join(test_dir, "images.npy"), test_images)
np.save(os.path.join(test_dir, "labels.npy"), test_labels)
