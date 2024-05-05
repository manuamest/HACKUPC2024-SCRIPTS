import numpy as np
from scipy.spatial import distance
from feature_extraction import extract_features
import json

# Carga las características de las imágenes previamente almacenadas
image_features = np.load('image_features.npy', allow_pickle=True).item()

def find_similar_images(new_img_path, top_n=6):
    new_features = extract_features(new_img_path)
    distances = {img_path: distance.euclidean(new_features, features) for img_path, features in image_features.items()}
    sorted_imgs = sorted(distances.items(), key=lambda x: x[1])[:top_n]  # incluir scores
    return sorted_imgs

def generate_json(image_data):
    results = []
    for img_path, score in image_data:
        result = {
            "year": None,
            "season": None,
            "product_type": None,
            "filename": img_path.split('/')[-1],
            "similarity_score": score
        }
        results.append(result)
    return json.dumps(results, indent=4)

# Ruta a la nueva imagen para encontrar imágenes similares
new_img_path = 'images/inputs/CHANCLA.jpg'
similar_images = find_similar_images(new_img_path)
json_output = generate_json(similar_images)
print(json_output)

# Guardar el resultado en un archivo JSON
with open('JSONS/similar_images.json', 'w') as f:
    f.write(json_output)
