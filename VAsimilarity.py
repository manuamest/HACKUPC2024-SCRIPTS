import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import json
import time
from functools import lru_cache
import re

def process_image(img_path, size=(256, 256)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen desde {img_path}")

    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No se encontraron contornos en {img_path}. Usando imagen completa para el procesamiento.")
        mask = np.ones_like(gray) * 255  # Usar una máscara que no altera la imagen.
    else:
        cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return img, masked_img


@lru_cache(maxsize=None)
def process_image_cached(img_path, size=(256, 256)):
    return process_image(img_path, size)

def compare_color_histograms(base_img, other_img, bins=32):
    hist_range = [0, 256]
    base_hist = cv2.calcHist([base_img], [0, 1, 2], None, [bins, bins, bins], hist_range * 3)
    other_hist = cv2.calcHist([other_img], [0, 1, 2], None, [bins, bins, bins], hist_range * 3)
    base_hist = cv2.normalize(base_hist, base_hist).flatten()
    other_hist = cv2.normalize(other_hist, other_hist).flatten()
    score = cv2.compareHist(base_hist, other_hist, cv2.HISTCMP_CORREL)
    return score

def compare_images(base_img, other_img_path):
    base_image, base_masked_img = base_img
    other_image, other_masked_img = process_image_cached(other_img_path)
    form_score = ssim(cv2.cvtColor(base_masked_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(other_masked_img, cv2.COLOR_BGR2GRAY))
    color_score = compare_color_histograms(base_masked_img, other_masked_img)
    total_score = form_score + 0.2 * color_score
    return total_score

def parse_image_details(filename):
    # Ajuste de la expresión regular para que coincida con el formato: 2021_W_1_16360610040_3_1_1.jpg
    pattern = r'(\d{4})_([A-Z])_(\d)_\d+_\d+_\d+_\d+.jpg'
    match = re.search(pattern, filename)
    if match:
        year, season, product_type = match.groups()
        product_types = {
            '0': 'Clothes',
            '1': 'Shoes',
            '2': 'Perfumery',
            '3': 'Sport',
            '4': 'Home'
        }
        return {
            'year': year,
            'season': 'Summer' if season == 'V' else 'Winter',
            'product_type': product_types.get(product_type, 'Unknown')
        }
    else:
        return {}  # Devuelve un diccionario vacío si no hay coincidencia



def main(directory, base_image_path, score_umbral=0.9):
    base_image, base_masked_img = process_image_cached(base_image_path)

    scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") and filename != os.path.basename(base_image_path):
            other_image_path = os.path.join(directory, filename)
            total_score = compare_images((base_image, base_masked_img), other_image_path)
            if total_score != 1.20 and total_score >= score_umbral:
                image_details = parse_image_details(filename)
                image_details.update({
                    'filename': filename,
                    'similarity_score': total_score
                })
                scores.append(image_details)

    # Ordenar las imágenes por score de similitud
    sorted_scores = sorted(scores, key=lambda item: item['similarity_score'], reverse=True)

    # Limitar a los primeros 6 resultados
    top_scores = sorted_scores[:6]  # Obtener solo los primeros 6 elementos

    # Saving results to a JSON file
    with open('JSONS/similarity_scores.json', 'w') as f:
        json.dump(top_scores, f, indent=4)  # Guardar solo los primeros 6

if __name__ == "__main__":
    directory = 'downloaded_images_2'
    base_image_path = 'images/inputs/PCHAN.jpg'
    score_umbral = 0.9
    start_time = time.time()
    main(directory=directory, base_image_path=base_image_path, score_umbral=score_umbral)
    end_time = time.time()
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")


