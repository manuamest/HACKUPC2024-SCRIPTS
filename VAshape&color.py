import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
import time
from functools import lru_cache

def process_image(img_path, size=(256, 256)):
    """Procesar la imagen para obtener la imagen de contorno y la imagen con máscara aplicada."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)  # Redimensionar imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return img, masked_img

@lru_cache(maxsize=None)  # Cache para almacenar imágenes procesadas
def process_image_cached(img_path, size=(256, 256)):
    return process_image(img_path, size)

def compare_color_histograms(base_img, other_img, bins=32):
    """Comparar histogramas de color con menos bins para mejorar el rendimiento."""
    hist_range = [0, 256]
    base_hist = cv2.calcHist([base_img], [0, 1, 2], None, [bins, bins, bins], hist_range * 3)
    other_hist = cv2.calcHist([other_img], [0, 1, 2], None, [bins, bins, bins], hist_range * 3)
    base_hist = cv2.normalize(base_hist, base_hist).flatten()
    other_hist = cv2.normalize(other_hist, other_hist).flatten()
    score = cv2.compareHist(base_hist, other_hist, cv2.HISTCMP_CORREL)
    return score

def compare_images(base_img, other_img_path):
    """Comparar base_img con otra imagen y devolver el score total sumando la similitud de forma y color."""
    base_image, base_masked_img = base_img
    other_image, other_masked_img = process_image_cached(other_img_path)
    form_score = ssim(cv2.cvtColor(base_masked_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(other_masked_img, cv2.COLOR_BGR2GRAY))
    color_score = compare_color_histograms(base_masked_img, other_masked_img)
    total_score = form_score + 0.2 * color_score  # Ponderación de la puntuación de color
    return total_score

def main(directory, base_image_path, score_umbral=0.9):
    base_image, base_masked_img = process_image_cached(base_image_path)

    # Diccionario para guardar los scores
    scores = {}

    # Procesar cada imagen en el directorio
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") and filename != os.path.basename(base_image_path):
            other_image_path = os.path.join(directory, filename)
            total_score = compare_images((base_image, base_masked_img), other_image_path)
            if total_score != 1.20 and total_score >= score_umbral:
                scores[filename] = total_score

    # Ordenar las imágenes por score de similitud
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Mostrar la imagen base y las hasta 20 más similares
    num_images = min(20, len(sorted_scores)) + 1
    plt.figure(figsize=(15, 10))
    plt.subplot(5, 5, 1)
    plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')

    for i, (img, score) in enumerate(sorted_scores[:num_images-1], start=2):
        other_image, _ = process_image_cached(os.path.join(directory, img))
        plt.subplot(5, 5, i)
        plt.imshow(cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Sim Total: {score:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = 'images/images2'  # Update to your images directory
    base_image_path = 'images/inputs/prueba.jpg'  # Update to your base image

    score_umbral = 0.8

    start_time = time.time()
    main(directory=directory, base_image_path=base_image_path, score_umbral=score_umbral)
    end_time = time.time()  # Finalizar el cronómetro
    elapsed_time = end_time - start_time  # Calcular el tiempo transcurrido
    print(f"Tiempo de ejecución: {elapsed_time:.2f} segundos")
