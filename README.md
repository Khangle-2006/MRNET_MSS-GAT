
# MSS-GAT


[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Task](https://img.shields.io/badge/Task-Medical%20Imaging-green.svg)](https://stanfordmlgroup.github.io/competitions/mrnet/)

> **State-of-the-Art approach for diagnosing ACL tears and Meniscal injuries on the Stanford MRNet dataset.**

This repository contains the official implementation of **MSS-Net** (Multi-Scale Shifted Network). Our method leverages a novel **RGB Multi-Scale Input Strategy** combined with **Spatial Graph Reasoning** to capture both global anatomical context and fine-grained lesion details (e.g., tiny meniscal tears).

<p align="center">
  <img src="pipeline.png" width="850" alt="Pipeline Architecture">
  <br>
  <em>Figure 1: The architecture of MSS-Net featuring the Dynamic Multi-Scale Processor and Hierarchical GAT.</em>
</p>

## 🌟 Key Features

* **🎨 RGB Multi-Scale Input:** Instead of standard grayscale, we utilize the 3 color channels to represent 3 different zoom levels:
    * **<span style="color:red">R (Red):</span> Global View** (Context).
    * **<span style="color:green">G (Green):</span> Shifted Mid View** (Targeting Meniscus with random shifts).
    * **<span style="color:blue">B (Blue):</span> Focal Close View** (Targeting ACL with center zoom).
* **🧠 Hybrid Backbone:** ResNet-18 initialized with ImageNet weights, enhanced with **3D MaxPool Adapters** to preserve high-intensity lesion signals.
* **🕸️ Spatial Graph Reasoning:** Converts feature maps into graph nodes using K-Means and processes them via Hierarchical Graph Attention Networks (GAT).
* **🎯 Focal Loss:** Optimized to handle class imbalance, boosting performance on hard classes like Meniscus.

## 📊 Performance

Our model achieves competitive performance on the MRNet Validation set:

| Task | AUC | Accuracy | Sensitivity | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ACL** | **0.9517** | 88.33% | 0.8704 | 0.8939 | 0.8704 |
| **Meniscus** | **0.8162** | 74.17% | 0.8654 | 0.5294 | 0.7207 |
| **Abnormal** | **0.9128** | 87.50% | 0.9579 | 0.4400 | 0.9100 |
| **Average** | **0.8935** | 83.33% | 0.8979 | 0.6211 | 0.8337 |

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/MSS-Net.git](https://github.com/your-username/MSS-Net.git)
    cd MSS-Net
    ```

2.  **Create Environment:**
    We recommend using Conda with Python 3.10.
    ```bash
    conda create -n mrnet python=3.10 -y
    conda activate mrnet
    ```

3.  **Install Dependencies:**
    Install all required packages (including PyTorch with CUDA 11.8 support) using:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

1.  Download the [MRNet Dataset](https://stanfordmlgroup.github.io/competitions/mrnet/).
2.  Organize the data structure as follows:
    ```text
    MRNet-v1.0/
    ├── train/
    │   ├── sagittal/
    │   ├── coronal/
    │   └── axial/
    ├── valid/
    │   ├── sagittal/
    │   ├── coronal/
    │   └── axial/
    ├── train-acl.csv
    ├── train-meniscus.csv
    ├── valid-acl.csv
    └── ...
    ```

---

## 🔄 Backbone Pre-training (Optional)

1. To further boost performance, we pre-train the ResNet-18 backbone on a large-scale medical dataset (RadImageNet) to learn domain-specific features (Modality & Anatomy classification).
2. Download the [RadImageNet Dataset](https://www.radimagenet.com/).

**Data Structure for Pre-training:**
Ensure your pre-training data is organized by `Modality` -> `Anatomy` -> `Pathology` -> `Images`:
```text
Radiology_Dataset/
├── MR/
│   ├── knee/
│   │   ├── pathology_1/
│   │   └── ...
│   ├── spine/
│   └── ...
├── CT/
│   ├── abdomen/
│   └── ...
```

**Run Pre-training:**
```bash
python pretrain_backbone.py 
```
**Key Configuration (inside script):**
* `BATCH_SIZE`: 128
* `EPOCHS`: 100
*This will generate a `resnet18_medical_pretrain.pth` file. Use this path for `BACKBONE_WEIGHTS` in the main training script.*

---

## 🚀 Training (Main)

To train the model with the **Multi-Scale Shifted Strategy** (Best for Meniscus & ACL balance):

```bash
python train.py
```

**Key Configuration (inside script):**
* `BATCH_SIZE`: 14
* `ZOOM_MENISCUS`: 0.75 (Mid view)
* `ZOOM_ACL`: 0.55 (Close view)
* `BACKBONE_WEIGHTS`: Path to your pre-trained weights (default: ImageNet).

> **Note:** We use a "Smart Weight Loading" mechanism. If kernel sizes mismatch (e.g., 3x3 vs 7x7), the system automatically handles it by inflating or skipping layers without crashing.

---

## ⚡ Inference & Evaluation

To evaluate the trained model on the validation set and compute full metrics (AUC, Accuracy, Sensitivity, Specificity):

```bash
python inference.py
```

**What this script does:**
1.  Loads the best checkpoint (e.g., `best_model_v9_meniscus_final.pth`).
2.  Applies the **Multi-Scale Preprocessing** (Global-Mid-Close) to validation data.
3.  Outputs a detailed metrics table for all 3 tasks.

---

## 🎨 Visualization (Explainable AI)

We provide a powerful visualization tool using **Multi-View Grad-CAM** overlaid on the original MRI slices. This helps interpret *where* and *what* the model is looking at.

```bash
python visualize.py
```

**Visual Output Explanation:**
The script generates images in `vis_results_Static_MultiScale/` containing:
* **Heatmap (Jet Colormap):** Indicates high-attention regions (Red = High, Blue = Low).
* **<span style="color:green">Green Box:</span>** Represents the **Close View (Zoom 0.55)**, focusing on the ACL region.
* **<span style="color:blue">Blue Box:</span>** Represents the **Mid View (Zoom 0.75)**, covering the Meniscus boundaries.

<p align="center">
  <img src="MultiScale_1172.png" width="800" alt="Visualization Example">
  <br>
  <em>Figure 2: Grad-CAM Visualization with Multi-Scale bounding boxes showing precise localization of ACL and Meniscal tears.</em>
</p>

---

## 💾 Model Zoo & Checkpoints

We provide pre-trained weights for our best models. You can download them directly to reproduce our results.

| Model Variant | Description | Meniscus AUC | ACL AUC | Download |
| :--- | :--- | :---: | :---: | :---: |
| **MSS-Net (Best Balanced)** | Multi-Scale + Shift + Focal Loss | **0.8162** | **0.9517** | [Google Drive Link](https://drive.google.com/drive/u/0/folders/1jbHcjEdcXagtRVd-leYx3S3GZNE9J3sY) |
| **Backbone Only** | Pre-trained ResNet-18 (Medical) | N/A | N/A | [Google Drive Link](https://drive.google.com/drive/u/0/folders/1jbHcjEdcXagtRVd-leYx3S3GZNE9J3sY) |
| **Gradcam** | ACL and Meniscus using Gradcam | N/A | N/A | [Google Drive Link](https://drive.google.com/drive/u/0/folders/1jbHcjEdcXagtRVd-leYx3S3GZNE9J3sY) |

*Please download the `.pth` files and place them in the root directory.*



