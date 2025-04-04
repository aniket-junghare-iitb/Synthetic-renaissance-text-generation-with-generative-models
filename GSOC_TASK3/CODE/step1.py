import os
from pdf2image import convert_from_path
from PIL import Image

PDF_DIR = "/home/aniketj/GSOC_TASK3/PDFs/"  # Directory containing all the PDFs
IMAGE_DIR = "/home/aniketj/GSOC_TASK3/IMAGES/"  # Output directory to store the images
os.makedirs(IMAGE_DIR, exist_ok=True)

#Convert PDF pages to images
def pdf_to_images(pdf_path, output_folder, dpi=200):
    images = convert_from_path(pdf_path, dpi=dpi, fmt="jpeg") 
    image_paths = []
    for i, img in enumerate(images):
        img = img.convert("RGB")  
        img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1}.jpg")
        img.save(img_path, "JPEG", quality=85)
        image_paths.append(img_path)
    
    return image_paths


for pdf in os.listdir(PDF_DIR):
    if pdf.endswith(".pdf"):
        pdf_to_images(os.path.join(PDF_DIR, pdf), IMAGE_DIR, dpi=200)

print("PDF to Image Conversion Done")
