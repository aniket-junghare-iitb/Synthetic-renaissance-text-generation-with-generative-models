# Abstract

This project aims to generate *synthetic Renaissance-style* printed text images that authentically replicate the *visual characteristics* of *17th-century print media*. The objective is to simulate *realistic historical degradations* such as *ink bleed*, *smudging*, *fading*, and *printing inconsistencies*, enabling the creation of enriched datasets for training and evaluating *document restoration*, *OCR*, and *historical text analysis* models. By blending *historical aesthetics* with *modern generative techniques*, this work supports advancements in *digital humanities* and the preservation of *early modern printed culture*.

## 🔎Approach

### Dataset Preparation
PDF to Image Conversion: All available Renaissance PDFs are converted into high-resolution .jpg images using the pdf2image library. Each page is stored with a consistent naming scheme, enabling traceability between original documents and their image representations.
Text Extraction from .docx: Transcriptions corresponding to the scanned documents are extracted from .docx files using the python-docx library. These serve as the semantic textual inputs for guiding image generation.
Image Preprocessing: Each image is resized to a fixed resolution (256×256), normalized to the range [-1, 1], and converted to tensors. These transformations are necessary for consistent GAN training.
