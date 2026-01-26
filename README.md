**Computationally Explainable Multimodal Deep Learning for Discriminative Histopathological Classification of Head and Neck B-Cell and T-Cell Lymphomas** 

Author: Lucas Lacerda de Souza

Year: 2025
________________________________________
**1. Project Overview**

Multimodal AI pipeline for classifying Diffuse Large B Cell Lymphoma (DLBCL) and Extranodal Natural Killet T Cell Lymphoma, Nasal Type (ENKTCL-NT) using histopathological image patches, clinicopathological data, and nuclear morphometric features. The pipeline combines: Traditional machine learning (XGBoost with SHAP), Deep learning (CNNs + multilayer perceptron), Vision Transformer–based cell modelling (CellViT++), Explainable AI methods (Grad-CAM and SHAP).

________________________________________
**2. Pipeline**


<img width="1109" height="828" alt="Figure 1" src="https://github.com/user-attachments/assets/f8d41bce-d94e-472d-a172-8d9dd902477f" />



________________________________________
**3. Environment and Hardware**

All experiments were performed using the following configuration:

**Operating System:** Ubuntu 20.04.1 LTS

**Python Version:** 3.12.11

**PyTorch Version:** 2.8.0 (CUDA 12.8)

**CPU:** Intel Xeon W-2295 (18 cores / 36 threads)

**RAM:** 125 GB

**GPUs:** 3 × NVIDIA GeForce RTX 3090 (24 GB each)
________________________________________
**4. Environment Files**

**Channels:**

  • pytorch
  
  • nvidia
  
  • defaults
  
**Dependencies:**

  • python=3.12.11
  
  • pytorch=2.8.0
  
  • torchvision=0.19.0
  
  • torchaudio=2.8.0
  
  • cudatoolkit=12.8
  
  • numpy=1.26.4
  
  • pandas=2.2.3
  
  • scikit-learn=1.5.2
  
  • matplotlib=3.9.2
  
  • seaborn=0.13.2
  
  • pillow=10.4.0
  
  • tqdm=4.66.5
  
  • openpyxl=3.1.5
________________________________________
**5. Model Architectures**

•	XGBoost + SHAP

•	U-Net++

•	AlexNet + Multilayer perceptron

•	ResNet50 + Multilayer perceptron

•	ConvNeXt-XLarge + Multilayer perceptron

•	CellViT++

•	GradCam

Note: CellViT++ are established architectures. Their original implementations were used without architectural modification. Only inference, downstream analysis, and integration code are included in this repository.

________________________________________
**6. Features Used**

• H&E image patches (299 × 299 pixels, 20×)

• Segmented patches (U-Net++)

• Cell-level embeddings and nuclear masks (CellViT++)

• Nucleus-based morphometric features

• Clinicopathological features (age, sex, lesion location)

________________________________________
**7. Evaluation Metrics**
   
•	XGBoost + SHAP – Classification (accuracy, area under the curve (AUC), F1-score, precision, recall and SHAP).

•	U-Net++/CellVit++ (Loss, Accuracy, Precision, Recall, IoU and Dice coefficient).

•	AlexNet (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	ResNet50 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	ConvNeXt-XLarge (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	GradCam - XGBoost - Classification (accuracy, area under the curve (AUC), F1-score, precision, recall). 

________________________________________

**8. Repository Structure**
   
## 📂 Repository Structure

DATA - Data used in the training

MODELS - Models used in the study

RESULTS - Results of the study

INFERENCE.py — Inference Script Example

LICENSE.txt — Project license

MODEL_CARD.txt — Description of the essential information of the study

README.md — Documentation and usage instructions

REQUIREMENTS.txt — Dependencies

________________________________________

**9. Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification-DLBCL-ENKTCL-NT.git
cd Classification-DLBCL-ENKTCL-NT

________________________________________

**10. Quick Start Guide**

**10.1. Clone the repository**

git clone https://github.com/lucas-lacerda-de-souza/Classification-DLBCL-ENKTCL-NT.git
cd Classification-DLBCL-ENKTCL-NT

**10.2. Create and activate the environment**

conda env create -f environment.yml
conda activate dlbcl-enktcl-ai

**10.3. Run inference**

python inference.py --input_dir ./data/test/ --output_dir ./results/

**10.4. Generate Grad-CAM heatmaps**

python scripts/visualize_gradcam.py \
  --model resnet50 \
  --input_dir ./data/test/ \
  --output_dir ./gradcam/heatmaps/
________________________________________

**11. Compliance with TRIPOD-AI and CLAIM 2024 Guidelines**

This repository has been structured to meet the TRIPOD-AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis – 
AI extension) and CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging) requirements for transparent and reproducible AI in healthcare.

**Data Source and Splits**

Detailed in README.md → Dataset Organization and METHODS.md.
Data divided into 80% training, 10% validation, and 10% testing.
Two independent external validation cohorts used to assess generalizability.

**Model Architecture and Training**

Documented in /models and individual training scripts.
Includes optimizer (AdamW), learning rate, batch size, epochs, and loss functions.

**Performance Metrics**

Internal and external validation results summarized in /results
Cross-institutional evaluation demonstrates robustness to domain shifts.

**Interpretability and Explainability**

SHAP feature importance for XGBoost models and Grad-CAM heatmaps for CNNs included.
Code and examples available in /models and /data.

**Clinical and Biological Relevance**

Described in MODEL_CARD.md → Intended Use.
Designed to assist diagnostic workflows, not to replace expert evaluation.

**Limitations and Potential Biases**

Outlined in MODEL_CARD.
Includes dataset size, center-specific staining differences, and potential bias from single-institution data predominance.

**Ethical Considerations**

Discussed in MODEL_CARD.md → Ethical and Practical Considerations.
Model not intended for autonomous clinical use; human oversight required at all stages.

________________________________________

**12. Ethics**

This study was approved by the Ethics Committee of the Piracicaba Dental School, University of Campinas, Piracicaba, Brazil (protocol no. 67064422.9.1001.5418), 
and by the West of Scotland Research Ethics Service (20/WS/0017). The study was performed according to the clinical standards of the 1975 and 1983 Declaration of Helsinki. 
Written consent was not required as data was collected from surplus archived tissue. Data collected were fully anonymised.

________________________________________

**13. Data availability**

All the data derived from this study are included in the manuscript. We are unable to share the whole slide images and clinical data, due to restrictions in the 
ethics applications. However, we created synthetic slides to show the structure of the project.

________________________________________

**14. Code availability**

We have made the codes publicly available online, along with model weights ([https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL](https://github.com/lucas-lacerda-de-souza/Classification-DLBCL-ENKTCL-NT)). All code was written 
with Python Python 3.12.11, along with PyTorch 2.8.0. The full implementation of the model, including the code and documentation, has been deposited in the Zenodo repository 
and is publicly available (https://doi.org/10.5281/zenodo.17661989). 

________________________________________
**15. Citation**

@article{delasouza2025classification,
  title={Computationally Explainable Multimodal Deep Learning for Discriminative Histopathological Classification of Head and Neck B-Cell and T-Cell Lymphomas},
  author={Souza, Lucas Lacerda de, Chen, Zhiyang […] Khurram, Syed Ali and Vargas, Pablo Agustin},
  journal={(journal/ 2025)},
  year={2025}
}
________________________________________
**17. License**

MIT License © 2025 Lucas Lacerda de Souza





