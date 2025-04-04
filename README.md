## Abstract

This project aims to generate *synthetic Renaissance-style* printed text images that authentically replicate the *visual characteristics* of *17th-century print media*. The objective is to simulate *realistic historical degradations* such as *ink bleed*, *smudging*, *fading*, and *printing inconsistencies*, enabling the creation of enriched datasets for training and evaluating *document restoration*, *OCR*, and *historical text analysis* models. By blending *historical aesthetics* with *modern generative techniques*, this work supports advancements in *digital humanities* and the preservation of *early modern printed culture*.

## 🔎Approach

### Dataset Preparation
* PDF to Image Conversion: All available Renaissance PDFs are converted into high-resolution .jpg images using the pdf2image library. Each page is stored with a consistent naming scheme, enabling traceability between original documents and their image representations.
* Text Extraction from .docx: Transcriptions corresponding to the scanned documents are extracted from .docx files using the python-docx library. These serve as the semantic textual inputs for guiding image generation.
* Image Preprocessing: Each image is resized to a fixed resolution (256×256), normalized to the range [-1, 1], and converted to tensors. These transformations are necessary for consistent GAN training.

### Text Embedding Using BERT
* Transcription texts are tokenized and embedded using the pretrained bert-base-uncased model from HuggingFace Transformers.
* The [CLS] token representation is averaged across sequence length to form a dense vector (768 dimensions), capturing semantic information of the input text.
* These embeddings are later spatially expanded and fused with visual data to condition the generation process.

### GAN Architecture
 ####Generator:
* Inputs: A real image and its corresponding BERT embedding.
* The BERT embedding is projected to a spatial map and concatenated as an additional channel to the image.
* The network contains multiple residual blocks, enabling it to learn complex transformations while preserving visual structure.
* Output: A synthetic version of the image with learned historical degradations (e.g., faded ink, smudges).

 Discriminator:

A convolutional classifier trained to distinguish between real Renaissance images and the generated ones.

It guides the generator through adversarial training by penalizing unrealistic features.
