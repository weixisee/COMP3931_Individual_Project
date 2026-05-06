# Removing Demographic Bias in Chest X-ray Classification Model using Adversarial Unlearning
Individual Project (COMP3931) for BSc Computer Science 2025/26 at the University of Leeds by Wei Xi See.


## Project Overview
This project investigates whether adversarial unlearning via Domain Adversarial Neural Network (DANN) can reduce demographic bias across sex and age subgroups in chest X-ray classification while maintaining high diagnostic accuracy. A DenseNet-121 baseline classifier was trained using the NIH ChestX-ray14 dataset for binary atelectasis classification and three adversarial unlearning variants were developed, including: **DANN-Sex**, **DANN-Age** and **DANN-Both**.

## Project Structure
```
COMP3931_Individual_Project/
├── README.md
├── notebook/
│   ├── 1_DataPreprocessing.ipynb             
│   ├── 2_3_BinaryBaseline_Atelectasis.ipynb   
│   ├── DANN_sex.ipynb                        
│   ├── DANN_age.ipynb                         
│   └── DANN_both.ipynb                        
└── src/
    └── evaluation.py                         
```

## Project Requirements
```
- Python 3.12.13
- PyTorch 2.10.0 with CUDA 12.8
- torchvision 0.25.0
- scikit-learn 1.6.1
- NumPy 2.0.2
- h5py
- tqdm
- pytorch-grad-cam
  ```

All the notebooks were implemented on Google Colab using NVIDIA Tesla T4 GPU.

## Running Instructions
1) `1_DataPreprocessing.ipynb` inspects the dataset and set up the processed dataset.
2) `2_3_BinaryBaseline_Atelectasis.ipynb` trains and evalautes the baseline classifier
3) `DANN-Sex.ipynb`, `DANN_Age.ipynb`, `DANN-Both.ipynb` trains and evaluate three different adversarial unlearning variants


## Dataset 
This project used the NIH ChestX-ray14 dataset accessed via Medical Imaging Meta-Dataset (MedIMeta) where the images were resized to 224 x 224 pixels.

MedIMeta version: https://zenodo.org/records/7884735

Original NIH version: https://nihcc.app.box.com/v/ChestXray-NIHCC
