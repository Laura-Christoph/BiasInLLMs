# BIAS IN LARGE LANGUAGE MODELS  
### Discrepancies in the Categorization of Negative Religious Hate Comments Across Cultural Contexts

This repository contains the code and sample files used in my bachelor's thesis analyzing bias in large language models (LLMs). The core objective of the project was to explore how well these models identify religious hate speech across different cultural contexts, with a focus on inconsistencies and potential biases in categorization.

---

## Repository Structure

### Scripts

- **`senti_preclassify.py`**  
  Used in the early stages of the project, this script combines sentiment-based pre-classification with manual keyword filtering and annotation to generate training data. This process laid the foundation for the binary classification task: _"Religious hate speech"_ vs. _"Not religious hate speech"_.

- **`hatetrain.py`**  
  Script for training initial models on the annotated dataset.

- **`hatefinetune.py`**  
  Built upon the training phase, this script was used for iterative fine-tuning of the models to improve performance.

- **`hateclassify.py`**  
  Applies the final fine-tuned model to the full dataset to classify Reddit posts according to the binary hate speech labels.

### Jupyter Notebooks

- **`code.ipynb`**  
  Contains exploratory code snippets used throughout the project to analyze and inspect data. This notebook is less structured and serves as a lab for quick testing and ad-hoc experimentation.

- **`BiasAnalysis.ipynb`**  
  Focused on post-classification analysis. This notebook examines the results of LLM inference and explores potential cultural and linguistic biases in model predictions.

- **`LLMsCategorization.ipynb`**
  Used to categorize the hate speech sample as well as to create a topic model and concept map

### Folders

- **`LLMs_Stance and Category/`**  
  Contains inference results categorized by model, including stance detection and classification outcomes for religious hate speech across different LLMs.

- **`best_distilbert_model_7/`**  
  Final trained binary classifier model based on DistilBERT.  
  - **Note:** While this model performs well in detecting _non-hateful_ content, it struggles with identifying actual religious hate speech and would benefit from additional manual annotation and training data.

---

## Notes

- The dataset includes Reddit posts and is annotated based on both manual inspection and semi-automated filtering.
- The work focuses on model behavior across cultural contexts, not just model accuracy, and highlights discrepancies and possible underlying biases.
- This repository includes partial code and sample files only. Sensitive data or full datasets are not included due to ethical and legal considerations.

---

## Thesis Overview

> This thesis investigates **bias in large language models** with regard to their ability to consistently and accurately categorize **negative religious hate comments**. The research includes both the creation of a supervised classifier and an evaluation of several LLMs' performances in this domain.

---

## Technologies Used

- Python (3.8+)
- PyTorch & Transformers (Hugging Face)
- Scikit-learn
- Jupyter Notebook
- Pandas, NumPy, Matplotlib
