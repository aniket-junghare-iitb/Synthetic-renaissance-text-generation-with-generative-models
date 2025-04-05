# 🕰️ Synthetic Renaissance Text Generation with GANs

## **Abstract**

This project aims to generate *synthetic Renaissance-style* printed text images that authentically replicate the *visual characteristics* of *17th-century print media*. The objective is to simulate *realistic historical degradations* such as *ink bleed*, *smudging*, *fading*, and *printing inconsistencies*, enabling the creation of enriched datasets for training and evaluating *document restoration*, *OCR*, and *historical text analysis* models. By blending *historical aesthetics* with *modern generative techniques*, this work supports advancements in *digital humanities* and the preservation of *early modern printed culture*.

---

## 🔎 **Approach**

### **1. Dataset Preparation**
- **PDF to Image Conversion**: All available Renaissance PDFs are converted into high-resolution `.jpg` images using the `pdf2image` library. Each page is stored with a consistent naming scheme, enabling traceability between original documents and their image representations.  
- **Text Extraction from `.docx`**: Transcriptions corresponding to the scanned documents are extracted from `.docx` files using the `python-docx` library. These serve as the semantic textual inputs for guiding image generation.  
- **Image Preprocessing**: Each image is resized to a fixed resolution (**256×256**), normalized to the range `[-1, 1]`, and converted to tensors. These transformations are necessary for consistent GAN training.

### **2. Text Embedding Using BERT**
- Transcription texts are tokenized and embedded using the pretrained **`bert-base-uncased`** model from *HuggingFace Transformers*.
- The `[CLS]` token representation is averaged across sequence length to form a dense vector (**768 dimensions**), capturing semantic information of the input text.
- These embeddings are later spatially expanded and fused with visual data to condition the generation process.

### **3. GAN Architecture**
This project uses a *Generative Adversarial Network (GAN)* architecture guided by BERT text embeddings to simulate Renaissance-style degradations on textual images.

#### **Generator**
- **Inputs**: A real image and its corresponding BERT embedding.  
- The BERT embedding is projected to a spatial map and concatenated as an additional channel to the image.  
- The network contains multiple *residual blocks*, enabling it to learn complex transformations while preserving visual structure.  
- **Output**: A synthetic version of the image with learned historical degradations (e.g., *faded ink*, *smudges*).

#### **Discriminator**
- A *convolutional classifier* trained to distinguish between real Renaissance images and the generated ones.
- It guides the generator through adversarial training by penalizing unrealistic features.

#### **Loss Functions**
- Combines *adversarial loss* (real vs fake classification) with an *L1 cycle-consistency loss* to retain image structure while applying degradation.

### **4. Training Strategy**

#### **Loss Functions**
- **Adversarial Loss (Binary Cross-Entropy)**: Ensures the generator produces realistic outputs that can fool the discriminator.
- **Cycle-Consistency Loss (L1)**: Encourages the generator to preserve the structural fidelity of the input image.

#### **Optimization**
- Both networks are trained using the *Adam optimizer* with tuned learning rates and momentum parameters (betas).

#### **Training Details**
- Trained for **150 epochs** using **batch size = 8**.
- At each step, the generator and discriminator are updated in tandem.
- Intermediate results are saved at the end of each epoch.

### **5. Image Generation Pipeline**
- Given a test image and optional transcription, the model generates a *historically-degraded synthetic version*.
- The generator runs in evaluation mode, and outputs are stored in a dedicated output directory for inspection.

### **6. Visual Degradation Simulation**
The GAN implicitly learns to apply historical degradation effects such as:
- *Ink bleeding* and *inconsistent printing*
- *Smudges* and *faded strokes*
- *Alignment noise* and *paper texture artifacts*

These effects are learned directly from training data, avoiding the need for explicit rules or filters.

---

## ✔ **Evaluation Metrics**

To evaluate the performance of the GAN model in generating synthetic Renaissance-style printed text images, I employed a combination of *pixel-level*, *structural*, and *perceptual* metrics. Each of these offers unique insights into how well the generated images replicate the visual characteristics of historical print media.

### **1. Structural Similarity Index (SSIM)**
Assesses whether the generated images preserve the *structural integrity* of the original printed text. A higher SSIM score indicates strong *layout and texture similarity*.

### **2. Peak Signal-to-Noise Ratio (PSNR)**
Provides a measure of *pixel-wise fidelity* between generated and real images. Though not perceptually rich, it ensures the GAN avoids excessive noise or distortions.

### **3. Learned Perceptual Image Patch Similarity (LPIPS)**
Measures *perceptual similarity* using deep features and aligns with human visual judgment. A lower LPIPS score indicates higher *visual realism*.

---

## 👀 **Results Analysis**

| **Metric** | **Best Score** | **Average Score** | **Median Score** | **Maximum Score** | **Minimum Score** |
|-----------|----------------|-------------------|------------------|-------------------|-------------------|
| **SSIM**     | 0.871348       | 0.753159          | 0.776838         | 0.871348          | 0.580901          |
| **PSNR**     | 27.818388      | 27.393436         | 27.386749        | 27.818388         | 26.969746         |
| **LPIPS**    | 0.109699       | 0.260597          | 0.252469         | 0.467809          | 0.109699          |

---

## **Evaluation Summary**

### **SSIM**
The highest score (**0.8713**) shows close structural resemblance, with an average of **0.7531** and median **0.7768**, suggesting reliable preservation of *layout and spatial coherence*.

### **PSNR**
Best score was **27.81 dB**, with average **27.39 dB**, showing *pixel accuracy* and limited noise introduction.

### **LPIPS**
Lowest LPIPS (**0.1097**) indicates *high perceptual realism*. The average (**0.2605**) and median (**0.2524**) confirm that the generator captures the *visual style* of Renaissance prints well.

---

## 🖼 **Sample Generated Images**

![Image 1](GEN_IMAGES/54.jpg)  
![Image 2](GEN_IMAGES/27.jpg)  
![Image 3](GEN_IMAGES/13.jpg)  
![Image 4](GEN_IMAGES/17.jpg)  
![Image 5](GEN_IMAGES/2.jpg)  
![Image 6](GEN_IMAGES/37.jpg)  
![Image 7](GEN_IMAGES/49.jpg)  
![Image 8](GEN_IMAGES/5.jpg)  
![Image 9](GEN_IMAGES/9.jpg)  
![Image 10](GEN_IMAGES/fake_img_109.png)  
![Image 11](GEN_IMAGES/fake_img_116.png)  
![Image 12](GEN_IMAGES/fake_img_141.png)

---

## 🚀 **Future Improvements**

### **1. Multi-Scale Discriminator**
Can help the model learn both *global composition* and *local texture*.

### **2. Higher Resolution Support**
Using *progressive GANs* or *ESRGAN* could enhance *legibility* and support OCR.

### **3. Attention-Based Fusion**
*Cross-attention* between text and image features could yield more *context-aware degradations*.

### **4. Larger Dataset**
Data augmentation or synthetic layout variations can improve *generalization*.

---

## 🔗 **Download Trained Model**

You can download the trained generator and discriminator weights here:
- [**Generator Weights** (`generator_renaissance.pth`)](https://drive.google.com/file/d/1H7wMh_L24c4AzKGyD9YH60hgfO_ycgzI/view?usp=sharing)
- [**Discriminator Weights** (`discriminator_renaissance.pth`)](https://drive.google.com/file/d/1gJFDkU_iOpb2VfhM2T8YiFHf3o7tjq4f/view?usp=sharing)

---

## 💻 **Implementation Guide**

Refer to this [**notebook**](synthetic_generation.ipynb) for the general guideline.

### **1. Install Required Packages**

Ensure that you have the required packages installed by running:

```bash
pip install -r requirements.txt
