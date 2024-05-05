import requests
import os
import pandas as pd
import re

# Configuración inicial
csv_path = 'images/inditex.csv'  # Ajusta la ruta según necesidad
base_dir = 'downloaded_images'
os.makedirs(base_dir, exist_ok=True)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.google.com/'
}

# Función para descargar y guardar una imagen
def download_image(url, category):
    try:
        # Extraer el nombre de la imagen desde el URL
        file_name = url.split('/')[-1].split('?')[0]
        file_path = os.path.join(base_dir, category, file_name)
        
        # Crear el directorio de categoría si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Solicitar y guardar la imagen
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Descargada {file_path}')
        else:
            print(f"Failed to download {url} with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Categorías basadas en el número de categoría
categories = {
    '0': 'Clothes',
    '1': 'Shoes',
    '2': 'Perfumery',
    '3': 'Sport',
    '4': 'Home'
}

try:
    # Leer hasta 100 líneas del archivo CSV
    df = pd.read_csv(csv_path, nrows=100)

    # Procesar cada fila del DataFrame
    for index, row in df.iterrows():
        for key in ['IMAGE_VERSION_1', 'IMAGE_VERSION_2', 'IMAGE_VERSION_3']:
            urls = str(row[key]).split(',')
            for single_url in urls:
                cleaned_url = single_url.strip().strip('"')  # Elimina espacios y comillas extras
                if cleaned_url:  # Verifica que la URL no esté vacía
                    category_code = re.search(r'/photos///\d+/V/(\d)/', cleaned_url)
                    if category_code:
                        category = categories.get(category_code.group(1), 'Otros')
                    else:
                        category = 'Otros'
                    download_image(cleaned_url, category)

except Exception as e:
    print(f"Error processing CSV: {e}")
