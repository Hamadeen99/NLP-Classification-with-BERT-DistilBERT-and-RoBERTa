# NLP Classification with BERT, DistilBERT, and RoBERTa

## Overview
This project explores the use of pre-trained transformer models for NLP classification tasks, specifically on the emotion dataset (4 classes) and the DBpedia dataset (14 classes). The models used include:
- **BERT (Bidirectional Encoder Representations from Transformers)**
- **DistilBERT (a distilled version of BERT)**
- **RoBERTa (Robustly Optimized BERT)**

### Purpose
The primary objective is to fine-tune these models for multiclass classification on:
1. **Emotion Dataset**: Includes six classes:
   - Sadness
   - Joy
   - Love
   - Anger
   - Fear
   - Surprise
     
2. **DBpedia Dataset**: A dataset for large-scale text classification, containing 14 non-overlapping categories such as Company, Educational Institution, Artist, Athlete, etc.

The project compares the performance of these models based on  F1-score.

---

## Files in the Repository

### `Utils.py`
Contains utility functions for:
- Loading and preparing the emotion dataset and DBpedia dataset.
- Tokenizing text using BERT, DistilBERT, or RoBERTa tokenizers.
- Creating dataloaders for training, validation, and testing.

### `MyBertModel.py`
Defines the architecture for fine-tuning the pre-trained models:
- A classification head is added to the transformer output.
- Supports freezing specific layers of the model for partial fine-tuning.

### `BertMulticlassClassification.py`
Main script to:
- Train and evaluate the model.
- Plot training and validation loss.
- Display confusion matrices.
- Generate classification reports.

### `results/`
A directory to store:
- Training logs.
- Confusion matrices.
- Plots for training and validation losses.
- Final metrics and evaluation results.

---

## Installation

### Requirements
Create a Python environment and install the following dependencies:

```bash
pip install torch transformers datasets matplotlib scikit-learn tqdm
```

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Hamadeen99/NLP-Classification-with-BERT-DistilBERT-and-RoBERTa
   cd NLP-Classification-with-BERT-DistilBERT-and-RoBERTa
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a compatible GPU for training (optional but recommended).

---

## Usage

### Train and Evaluate
Run the main script to train and evaluate the model:
```bash
python BertMulticlassClassification.py
```

### Model Options
- Uncomment specific lines in `MyBertModel.py` to freeze initial layers for partial fine-tuning.
- Modify `Utils.py` to switch between BERT, DistilBERT, and RoBERTa models.

---

## Results


#### Emotion Dataset

Note that the Number of Epochs = 15

1. **BERT**:
   - F1-macro with freezing: **0.89**
   - F1-macro without freezing: **0.82**
     
2. **DistilBERT**:
   - F1-macro: **0.84**
   
     
3. **RoBERTa**:
   - F1-macro with freezing: **0.89**
   - F1-macro without freezing: **0.76**


#### DBpedia Dataset
1. **BERT**:
   - F1-macro: **0.93**
  
     
2. **DistilBERT**:
   - F1-macro: **0.89**
   
     
3. **RoBERTa**:
   - F1-macro: **0.91**
   

---

### Insights
- Fine-tuning the entire model yields better results compared to freezing the initial layers.
- BERT provides the best performance but is computationally heavier.
- DistilBERT is faster but sacrifices some accuracy.
- RoBERTa balances performance and computational efficiency.

---

## Future Work
1. Experiment with larger datasets and more than 14 classes.
2. Implement advanced techniques like knowledge distillation or pruning to optimize model size and performance.
3. Explore additional transformer architectures for NLP tasks.

##

This project was assigned as part of the NLP and LLM Course, instructed by Professor Ausif mahmood.


