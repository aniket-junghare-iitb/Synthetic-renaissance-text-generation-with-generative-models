import os
from pdf2image import convert_from_path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import docx
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torchmetrics.image.kid import KernelInceptionDistance
import cv2
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy.fft import fft2, fftshift

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

# Load BERT model and tokenizer for text embeddings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

# Function to extract text from docx files
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = " ".join([p.text for p in doc.paragraphs])
    return text

# Function to generate text embeddings
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  

# Dataset class incorporating both images and text embeddings
class TextImageDataset(Dataset):
    def __init__(self, img_dir, txt_dir, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.txt_paths = {os.path.splitext(f)[0]: os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.docx')}
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        doc_name = os.path.splitext(os.path.basename(img_path))[0]
        text_embedding = torch.zeros((1, 768))  
        if doc_name in self.txt_paths:
            text = extract_text_from_docx(self.txt_paths[doc_name])
            text_embedding = get_text_embedding(text)
        
        return image, text_embedding.squeeze(0)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = TextImageDataset(img_dir="/home/aniketj/GSOC_TASK3/IMAGES", txt_dir="/home/aniketj/GSOC_TASK3/TRANSCRIPTIONS", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Dataset loaded with images and text embeddings")

# Generator model
class Generator(nn.Module):
    def __init__(self, input_channels=3, text_dim=768, output_channels=3, num_residual_blocks=6):
        super(Generator, self).__init__()
        self.text_fc = nn.Linear(text_dim, 256 * 256)  # Embed text into image size
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + 1, 64, kernel_size=7, stride=1, padding=3),  # +1 for text map
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, text):
        text_map = self.text_fc(text).view(-1, 1, 256, 256)  # Reshape to match image
        x = torch.cat([x, text_map], dim=1)  # Concatenate text representation with image
        x = self.initial(x)
        x = self.residual_blocks(x)
        return self.final(x)
    







# Residual block for generator
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
    




# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    




# Function to generate text-based images
def generate_text_image(model, test_image_path, img_name):
    model.eval()
    image = Image.open(test_image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_image = model(image, torch.zeros(1, 768).to(device))

    save_image(fake_image, img_name)
    print(f"Saved: {img_name}")



# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

GEN_IMAGE_DIR = "/home/aniketj/GSOC_TASK3/GENERATED_IMAGES/"  # Output directory for images
os.makedirs(GEN_IMAGE_DIR, exist_ok=True)



# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    for i, (real_text_images, text_embeddings) in enumerate(dataloader):
        real_text_images, text_embeddings = real_text_images.to(device), text_embeddings.to(device)

        # Generate fake images
        fake_text_images = G(real_text_images, text_embeddings)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion_GAN(D(real_text_images), torch.ones_like(D(real_text_images)))
        fake_loss = criterion_GAN(D(fake_text_images.detach()), torch.zeros_like(D(fake_text_images)))
        D_loss = (real_loss + fake_loss) / 2
        D_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        G_loss = criterion_GAN(D(fake_text_images), torch.ones_like(D(fake_text_images))) + \
                 criterion_cycle(fake_text_images, real_text_images) * 10
        G_loss.backward()
        optimizer_G.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")

    save_image(fake_text_images, f"/home/aniketj/GSOC_TASK3/GENERATED_IMAGES/fake_img_{epoch+1}.png")








# Generate text-based image
generate_text_image(G, "/home/aniketj/GSOC_TASK3/TEST_IMAGES/9.jpg","generated_renaissance_image9.png") # generated image is stored in JUPYTER NOTEBOOK folder




# Save trained models
torch.save(G.state_dict(), "/home/aniketj/GSOC_TASK3/generator_renaissance.pth")
torch.save(D.state_dict(), "/home/aniketj/GSOC_TASK3/discriminator_renaissance.pth")
print("Models saved successfully!")


##############################################################################################################################################

# Evaluation metrics

def calculate_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Could not load image: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not load image: {img2_path}")

    # Resize images if they have different shapes
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    score, _ = ssim(img1, img2, full=True)
    return score


# PSNR
def calculate_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))




# Function to load and preprocess an image for LPIPS
def load_image(image_path):
    img = cv2.imread(image_path)  # Load image (BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (256, 256))  # Resize to 256x256
    img = np.transpose(img, (2, 0, 1)) / 255.0 * 2 - 1  # Normalize to [-1,1]
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    return img




# Function to generate images
def generate_images(input_dir, output_dir):
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_name in tqdm(image_files, desc="Generating Images"):
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_image = G(image, torch.zeros(1, 768).to(device))

        save_path = os.path.join(output_dir, f"{img_name}")
        save_image(generated_image, save_path)

    print(f"Generated images saved in: {output_dir}")



# Load the trained generator model
G = Generator().to(device)
G.load_state_dict(torch.load("/home/aniketj/GSOC_TASK3/generator_renaissance.pth", map_location=device))
G.eval()  # Set model to evaluation mode





# Load LPIPS model (AlexNet backbone)
loss_fn = lpips.LPIPS(net='alex') 



# Load fake and real images
real_image = load_image("/home/aniketj/GSOC_TASK3/TEST_IMAGES/99.jpg")  # Replace with your real image path
fake_image = load_image("/home/aniketj/GSOC_TASK3/CODE/generated_renaissance_image.png")  # Replace with your fake (generated) image path



# Define directories
test_input_dir = "/home/aniketj/GSOC_TASK3/TEST_IMAGES"  # Directory containing input images
test_output_dir = "/home/aniketj/GSOC_TASK3/GENERATED_TEST_IMAGES"  # Directory to save generated images
os.makedirs(test_output_dir, exist_ok=True)  


generate_images(test_input_dir, test_output_dir)



ssim_score = calculate_ssim("/home/aniketj/GSOC_TASK3/CODE/generated_renaissance_image.png" , "/home/aniketj/GSOC_TASK3/TEST_IMAGES/99.jpg")
print("SSIM Score:", ssim_score)

psnr_score = calculate_psnr("/home/aniketj/GSOC_TASK3/TEST_IMAGES/99.jpg", "/home/aniketj/GSOC_TASK3/CODE/generated_renaissance_image.png")
print("PSNR Score:" ,psnr_score)

lpips_score = loss_fn(real_image, fake_image)
print("LPIPS Score:", lpips_score.item())
