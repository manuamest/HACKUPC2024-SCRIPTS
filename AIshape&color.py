# find_similar_images.py
import numpy as np
from scipy.spatial import distance
from feature_extraction import extract_features
import matplotlib.pyplot as plt
from PIL import Image

# Carga las características de las imágenes previamente almacenadas
image_features = np.load('image_features.npy', allow_pickle=True).item()

def find_similar_images(new_img_path, top_n=7):  # Solicitamos 7 para obtener 6 después de descartar el duplicado
    new_features = extract_features(new_img_path)
    distances = {img_path: distance.euclidean(new_features, features) for img_path, features in image_features.items()}
    sorted_imgs = sorted(distances, key=distances.get)[:top_n]
    return sorted_imgs[1:]  # Descarta el primero ya que es probable que sea la imagen de entrada

def display_images(image_paths):
    plt.figure(figsize=(15, 10))
    # Mostrar la imagen de entrada
    plt.subplot(1, len(image_paths) + 1, 1)
    img = Image.open(image_paths[0])
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    # Mostrar imágenes similares
    for i, img_path in enumerate(image_paths[1:], start=1):
        plt.subplot(1, len(image_paths) + 1, i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Similar {i}")
        plt.axis('off')

    plt.show()

# Ruta a la nueva imagen para encontrar imágenes similares
new_img_path = 'images/inputs/CHANCLA.jpg'
similar_images = [new_img_path] + find_similar_images(new_img_path)
print("Imágenes similares encontradas:", similar_images[1:])  # Muestra desde el segundo en adelante

# Mostrar la imagen de entrada y las imágenes similares
display_images(similar_images)
