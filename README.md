# Coin Detection Challenge 

EE-451 Image analysis and pattern recognition

| **Author**                | **Sciper** |
|---------------------------|------------|
| Jan Peter Reinhard Clevorn | 377937     |
| Carlos Collado Capell      | 377896     |
| Alejandro Lopez Rodriguez  | 369471     |

---

## Overview

This repository contains our solution to a coin detection and classification challenge from the IAPR course. The goal was to detect individual coins in images (sometimes with noisy backgrounds or with hands holding them) and classify each coin into its respective denomination. Additionally, we detect when a coin-like object does not belong to any supported class (OOD, or out-of-distribution).

**Key achievements:**
- Achieved a **public score of 0.9740** on the competition leaderboard (4th highest in class).
- Devised a robust segmentation pipeline using a circle-detection (Hough Transform) approach.
- Used a transfer learning with **AlexNet** to classify each coin patch into 16 possible classes (15 known denominations + 1 OOD class).

## Repository Structure

Here is the recommended folder structure for this project:

```
root/
├── best_model.pth               # Saved model weights
├── notebook.ipynb               # Jupyter Notebook with core experimentation
├── notebook_in_script_format.py # Script version of the notebook
├── requirements.txt             # Python dependencies
├── COMBINED_LABELS.csv          # CSV: Manually annotated coin patches (training set)
│
├── ref/                         # Reference images
│   ├── ref_chf.JPG
│   └── ref_eur.JPG
│
├── test/                        # Test dataset folder
├── train/                       # Training dataset folder
├── aggregated_patch_counts.csv  # Output predictions (aggregated at the image level)
└── README.md                    # This README file
```

- **ref/**: Contains example/reference coin images, useful for color/size comparison.  
- **train/**: Folder with training images.  
- **test/**: Folder with test images (unlabeled).  
- **notebook.ipynb**: The main exploratory notebook.  
- **notebook_in_script_format.py**: A script version of the Jupyter notebook.  
- **best_model.pth**: Trained AlexNet weights.  
- **COMBINED_LABELS.csv**: Contains the coin-level labels for all extracted patches in the training set (denominations and OOD).  
- **aggregated_patch_counts.csv**: Final results table from the classifier, aggregated by test image.  
- **requirements.txt**: Required Python packages.


## Step-by-Step Workflow

Below is the general workflow implemented in this project:

1. **Segmentation (Circle Detection)**
   - We use **OpenCV’s Hough Transform** (`cv2.HoughCircles`) to detect circular objects in each full image.  
   - Because the raw images can be large, they are **resized** to speed up processing.  
   - A **single-channel** (e.g., a Y or I channel in YIQ color space) is used to run the circle-detection more accurately.  
   - **Concentric circles** are removed (we only keep the outermost circle if the algorithm detects multiple concentric ones for the same coin).

2. **Patch Extraction**
   - For each detected circle, we crop a square patch containing the coin and place it onto a **1000×1000** black background.  
   - This preserves information about coin size (since smaller coins occupy fewer non-black pixels).

3. **Manual Annotation**
   - Each extracted patch in the **training** set is manually labeled with its coin denomination or marked as **OOD**.  
   - The file **COMBINED_LABELS.csv** contains these coin-level labels.

4. **Data Augmentation**
   - We address **class imbalance** (some coins appear fewer times than others) by:
     - Applying random flips, rotations, brightness changes, Gaussian blur, etc.  
     - Converting patches to grayscale or lightly sharpening them for increased contrast.  
     - Generating enough augmented samples so each class reaches a target count (e.g., 1000 images per class).

5. **Model & Training**
   - We use **AlexNet** (pre-trained on ImageNet) as a feature extractor and add a final classification head to predict 16 classes (15 coin denominations + 1 OOD).  
   - We adopt **CrossEntropyLoss** with **class weights** to handle residual imbalance.  
   - We use an **AdamW** optimizer with a **Cosine Annealing** learning rate schedule (including a warm-up period).

6. **Inference on Test Set**
   - The same **segmentation + patch extraction** pipeline is applied to each test image.  
   - Each patch is fed into the trained AlexNet model.  
   - Model predictions are stored in a CSV and then **aggregated** (`aggregated_patch_counts.csv`) at the per-image level.

7. **Results**
   - We provide the coin counts for each test image, which can be further used for calculating total values or other competition metrics.


## Quick Start

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure Folder Structure**  
   Place your `train/` and `test/` datasets within the root folder. These are not provided in the GitHub. Ask to the course instructors for them if interested.
3. **Segmentation & Patch Extraction**  
   - Run the segmentation/patch-extraction cells in `notebook.ipynb`.  
   - This step creates coin patches from your training and test images.
4. **Label Data**  
   - (Already done) but if needed, manually annotate each patch’s coin type in the CSV file.
5. **Data Augmentation & Model Training**  
   - Run the augmentation cells (optional if you already have `COMBINED_AUGMENTED` and the augmented CSV).  
   - Train the model by executing the training cells. A checkpoint (`best_model.pth`) will be saved.
6. **Inference**  
   - Apply the same patch-extraction procedure on the test dataset.  
   - Use the trained model to classify each patch; results are aggregated in `aggregated_patch_counts.csv`.
