# Removing Demographic Bias in Chest X-ray Classification using Adversarial Unlearning
Individual Project (COMP3931) for BSc Computer Science 25/26 at Unversity of Leeds. 

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
