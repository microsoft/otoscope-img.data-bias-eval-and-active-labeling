# Towards Reliable AI: Bias Identification, Prevention and Quality Improvement in Otoscopic Images

# Introduction 
This README describes how to reproduce results for the paper "Towards Reliable AI: Bias Identification, Prevention and Quality Improvement in Otoscopic Images". 

# Data Preparation
1. Download all the three public datasets. 
    - The Chile dataset: https://figshare.com/articles/dataset/Ear_imagery_database/11886630
        - Viscaino, Michelle, et al. "Computer-aided diagnosis of external and middle ear conditions: A machine learning approach." Plos one 15.3 (2020): e0229226.
    - The Ohio dataset: https://zenodo.org/records/4558155#.YXYYyC8Ro6U
        - Camalan, Seda, et al. "OtoMatch: Content-based eardrum image retrieval using deep learning." Plos one 15.5 (2020): e0232776.
    - The Turkey dataset: The original data link is currently inaccessible. For data access, please reach out to the author of the paper.
        - Zafer, Cömert. "Fusing fine-tuned deep features for recognizing different tympanic membranes." Biocybernetics and Biomedical Engineering 40.1 (2020): 40-51.
  
2. Rename their folder names as 'Chile',  'Ohio', and 'Turkey', respectively. 
3. Put all three datasets in folder DATA_MAIN_DIR=`../data/eardrum_public_data`. 
4. `data_bias_evaluation_framework/metadata/metadata.csv` is a dataframe including the relative path, source, class and binary class for each image. To reproduce this dataframe, run `data_bias_evaluation_framework/data_bias_evaluation_framework/prepare_dataset/generate_metadata.ipynb`.   
## Data Structure
```bash
DATA_MAIN_DIR
├── Chile
│   ├── Testing
│   │   ├── Chronic otitis media
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Earwax plug
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Myringosclerosis
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Normal
│   │   │   ├── Image1
│   │   │   ├── ...
│   ├── Training-validation
│   │   ├── Chronic otitis media
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Earwax plug
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Myringosclerosis
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Normal
│   │   │   ├── Image1
│   │   │   ├── ...
├── Ohio
│   ├── Tube_Effusion_Normal - 11_7_19
│   │   ├── Effusion
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Normal
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── Tube
│   │   │   ├── Image1
│   │   │   ├── ...
├── Turkey
│   ├── abnormal
│   │   ├── aom
│   │   │   ├── Test_aom
│   │   │   │   ├── Image1
│   │   │   │   ├── ...
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── csom
│   │   │   ├── Test_cosm
│   │   │   │   ├── Image1
│   │   │   │   ├── ...
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── earVentilationTube
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── earwax
│   │   │   ├── Test_earwax
│   │   │   │   ├── Image1
│   │   │   │   ├── ...
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── foreignObjectEar
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── otitisexterna
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── pseudoMembranes
│   │   │   ├── Image1
│   │   │   ├── ...
│   │   ├── tympanoskleros
│   │   │   ├── Image1
│   │   │   ├── ...
│   ├── normal
│   │   ├── Test_normal
│   │   │   ├── Images1
│   │   │   ├── ...
│   │   ├── Image1
│   │   ├── ...
```

# Environment Setup
Our python environment is summarized in `requirements.txt`. Note that CUDA version = 11.0 for our system, you might want to adjust the file to match your CUDA version. 
# Quantitative Data Bias Assessment
## Counterfactual Experiment I
To repropduce the Counterfactual Experiment I's results on the three public datasets, run the following commands to train a model on the Eclipsed Dataset. 
```bash
cd data_bias_evaluation_framework/train_model
python run_binary_classification_gen.py --model_name 'vit_b_16_384' --num_epoch 100 --eclipse --eclipse_extent 1.0  --cudaID 0  --elastic_tf  --lr 0.01
```
Run the following commands to train a model on the original dataset (Eclipsed Extent = 0). 
```bash
cd data_bias_evaluation_framework/train_model
python run_binary_classification_gen.py --model_name 'vit_b_16_384' --num_epoch 100 --cudaID 0  --elastic_tf  --lr 0.01
```
Run the notebook `data_bias_evaluation_framework/post_training/summarize_result.ipynb` to reproduce the visualization. 
## Counterfactual Experiment II
To repropduce Counterfactual Experiment II's results, run the notebook `data_bias_evaluation_framework/train_model/logistic_regression.ipynb`

# Qualitative Data Bias Assessment
## Detect near-duplicate images and images of similar style
Run the notebook `data_bias_evaluation_framework/post_training/qualitative_databias_assess.ipynb`
## Feature embeddings
Note that the feature embeddings were extracted from models stored in `data_bias_evaluation_framework/experiment/vit_b_16_384_False_0.0_False_32_1234_100_True_False_0.05_0.01_0_0.9`. To train your own models, run the following commands 
```bash
cd data_bias_evaluation_framework/train_model
python run_binary_classification_cv.py  --model_name 'vit_b_16_384' --num_epoch 100  --cudaID 0  --elastic_tf --lr 0.01
```
# Active Labeling
This part of the paper was based on a private dataset. To prepare your own dataset, use `/active_labeling/prepare_dataset/prepare_hierch_dataset.py`. The multitask model is available at `/active_labeling/models/models_hierch.py`. We used the function train_model_multitask in `/active_labeling/train_model/train_model.py` to train the multitask model. 
