import requests
import os
import pandas as pd
import time
import re


# Configuración inicial
csv_path = 'inditex.csv'
images_dir = 'downloaded_images_2'
os.makedirs(images_dir, exist_ok=True)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.google.com/'
}

# Función para descargar y guardar una imagen
def download_image(url, file_path):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Descargada {file_path}')
        else:
            print(f"Failed to download {url} with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

try:
    # Leer hasta 200 líneas del archivo CSV
    df = pd.read_csv(csv_path, delimiter=',', nrows=1000)

    df.dropna(inplace=True)
    
    urls = df['IMAGE_VERSION_3'].to_list()
    urls_clean = list(filter(None, urls))


    for clean_url in urls_clean:

        c_u = str(clean_url)
        meta = re.search(r"(?<=\/\/\/)(\d+\/\w\/\d\/\d)(?=\/)", c_u).group(0)
        pre_name = meta.replace("/","_")
        post_name = c_u.split('/')[-1].split('?')[0] # Nombre
        image_name = pre_name + post_name        
        file_path = os.path.join(images_dir, image_name)
        download_image(c_u, file_path)




except Exception as e:
    print(f"Error reading CSV: {e}")
