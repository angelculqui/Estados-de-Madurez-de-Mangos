import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Definir las rutas de los conjuntos de datos de entrenamiento y prueba
train_dir = "./Dataset mango/train"
test_dir = "./Dataset mango/test1"

# Cargar los datos de entrenamiento y prueba
train_images = np.load(os.path.join(train_dir, "images.npy"))
train_labels = np.load(os.path.join(train_dir, "labels.npy"))
test_images = np.load(os.path.join(test_dir, "images.npy"))
test_labels = np.load(os.path.join(test_dir, "labels.npy"))

# Codificar las etiquetas (verde y maduro) a valores numéricos (0 y 1)
label_encoder = LabelEncoder()
train_labels_numeric = label_encoder.fit_transform(train_labels)
test_labels_numeric = label_encoder.transform(test_labels)

# Normalizar las imágenes
train_images = train_images / 255.0
test_images = test_images / 255.0

# Definir el modelo de la CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels_numeric, epochs=10, batch_size=16, validation_data=(test_images, test_labels_numeric))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_images, test_labels_numeric)
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')
