import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
import numpy as np
import os

# Cargar MobileNetV2 con el modelo base, sin incluir la parte superior
base_model = MobileNetV2(weights='imagenet', include_top=False)
print(base_model.summary())

# Crear un nuevo modelo que tome la entrada de MobileNetV2 y entregue la salida de la capa de Global Average Pooling
output = base_model.layers[-1].output  # La última capa de base_model debería ser una capa de pooling
output = tf.keras.layers.GlobalAveragePooling2D()(output)  # Aplica GlobalAveragePooling2D
feature_model = tf.keras.Model(inputs=base_model.input, outputs=output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_model.predict(img_array)
    return features.flatten()

# Recorrer todas las imágenes y extraer sus características
image_features = {}
for folder in os.listdir('downloaded_images'):
    folder_path = os.path.join('downloaded_images', folder)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        features = extract_features(img_path)
        image_features[img_path] = features

# Guardar las características extraídas
np.save('image_features.npy', image_features)
