**Computationally Explainable Multimodal Deep Learning for Discriminative Histopathological Classification of Head and Neck B-Cell and T-Cell Lymphomas** 

Author: Lucas Lacerda de Souza

Year: 2025
________________________________________
**1. Project Overview**

Multimodal AI pipeline for classifying DLBCL vs. ENKTCL using three data streams: (1) histopathology patches processed with CNN + MLP fusion, (2) structured clinicopathologic data, and (3) morphometric nuclear features evaluated with XGBoost + SHAP. Interpretability is provided through Grad-CAM and SHAP-based explanations.
________________________________________
**2. Pipeline**

<img width="1476" height="1138" alt="Captura de tela 2025-11-18 114242" src="https://github.com/user-attachments/assets/42e234a2-95af-4e96-bdb6-48144a1ce9f2" />

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

•	GradCam

________________________________________
**6. Features Used**

• Patches (H&E)

• Patches (Unet++)
   
•	Morphometric features (nucleus-based)

•	Clinicopathologic features (age, sex, location)
________________________________________
**7. Evaluation Metrics**
   
•	XGBoost + SHAP – Classification (accuracy, area under the curve (AUC), F1-score, precision, recall and SHAP).

•	U-Net++ (Loss, Accuracy, Precision, Recall, IoU and Dice coefficient).

•	AlexNet (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	ResNet50 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	ConvNeXt-XLarge (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	GradCam - XGBoost - Classification (accuracy, area under the curve (AUC), F1-score, precision, recall). 

________________________________________
**8. Repository Structure**
   
## 📂 Repository Structure

INFERENCE.py — Inference Script Example

LICENSE.txt — Project license

MODEL_CARD.txt — Description of the essential information of the study 

README.md — Documentation and usage instructions

REQUIREMENTS.txt — Dependencies


data/

patches/

 ├── gradcam/
 
 │ ├── heatmaps/
 
 │ │ └── heatmap.png files
 
 │ └── patches/
 
 │  └── patch.png files

 │ └── wsi_heatmaps/
 
 │  └── wsi.png files
 
 ├── masks/
 
 │ ├── train/
 
 │ ├── val/
 
 │ └── test/
 
 │  └── mask.png files
 
 └── patches/
 
  ├── train/
  
  ├── val/
  
  └── test/
  
   └── patch.png files
   
 models/

 ├── multimodal_alexnet_patch_level.py
 
 ├── multimodal_alexnet_patient_level.py
 
 ├── multimodal_resnet50_patch_level.py
 
 ├── multimodal_resnet50_patient_level.py
 
 ├── multimodal_convnextxlarge_patch_level.py
 
 ├── multimodal_convnextxlarge_patient_level.py
 
 ├── segmentation_unet++.py
 
 ├── xgboost_classification_cpc_mpa.R
 
 └── xgboost_classification_gradcam.R

results/

 └── metrics

________________________________________

**9. Run models and reproduce tables**


