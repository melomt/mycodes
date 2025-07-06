import fitz  # PyMuPDF
import pandas as pd
import re
import os
import shutil
import cv2
import numpy as np

# Caminho do arquivo PDF
pdf_path = '/home/matheusjm/Documentos/codigospython/arquivosvale/PATROL/Ensaio_2_vargem_grande/PAT-RT-LAB-2358.20-001-Rev.00 (1).pdf'
# Caminho da pasta com os modelos de fotos
model_folder = '/home/matheusjm/Documentos/codigospython/arquivosvale/PATROL/Ensaio_1_itabira/modelos'

# Caminho da pasta de destino para as imagens classificadas
output_folder = '/home/matheusjm/Documentos/codigospython/arquivosvale/PATROL/Ensaio_2_vargem_grande/classificadas'

# Abre o arquivo PDF
doc = fitz.open(pdf_path)

# Lista para armazenar os dados extraídos
image_data = []

# Expressão regular para encontrar o ID no formato "ID: ITA-FJ00008 - AM002"
id_pattern = re.compile(r"(?:Amostra|ID):\s*(.+)", re.IGNORECASE)

# Função para comparar duas imagens usando o método de histograma
def compare_images(img1, img2):
    # Converte as imagens para escala de cinza
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcula o histograma das imagens
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

    # Compara os histogramas usando a correlação
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

# Carrega os modelos de fotos
models = {}
for model_name in os.listdir(model_folder):
    model_path = os.path.join(model_folder, model_name)
    model_image = cv2.imread(model_path)
    if model_image is not None:
        models[model_name] = model_image

# Itera por todas as páginas do PDF
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()

    # Procura pelo ID na 
    # página
    id_match = id_pattern.search(text)
    if id_match:
        id_value = id_match.group(1)  # Captura o valor do ID

        # Extrai todas as imagens da página
        image_list = page.get_images(full=True)

        # Itera sobre as imagens encontradas
        for img_index, img in enumerate(image_list):
            xref = img[0]  # O índice de referência da imagem
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Salva a imagem temporariamente
            temp_image_path = f"temp_image_page_{page_num + 1}_img_{img_index + 1}.png"
            with open(temp_image_path, "wb") as image_file:
                image_file.write(image_bytes)

            # Carrega a imagem extraída
            extracted_image = cv2.imread(temp_image_path)

            # Compara a imagem extraída com os modelos
            best_match = None
            best_similarity = -1
            for model_name, model_image in models.items():
                similarity = compare_images(extracted_image, model_image)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = model_name

            # Define o nome do arquivo classificado
            classified_image_name = f"classified_{best_match}_page_{page_num + 1}_img_{img_index + 1}.png"
            classified_image_path = os.path.join(output_folder, classified_image_name)

            # Move a imagem para a pasta de destino com o nome classificado
            shutil.move(temp_image_path, classified_image_path)

            # Adiciona os dados extraídos como uma linha na lista
            image_data.append({
                "ID": id_value,
                "Image Filename": classified_image_name,
                "Page Number": page_num + 1,
                "Image Index": img_index + 1,
                "Model Match": best_match,
                "Similarity Score": best_similarity
            })

# Cria um DataFrame com os dados extraídos
df_images = pd.DataFrame(image_data)

# Salva o DataFrame em um arquivo CSV
df_images.to_csv(os.path.join(output_folder, "extracted_images.csv"), index=False)

print("Imagens extraídas, classificadas e salvas com sucesso!")
print(df_images)