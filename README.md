# Comparison of RoBERTa Transformer and VADER for Sentiment Analysis of Amazon Reviews

This project compares the performance of **RoBERTa**, a transformer-based NLP model with **VADER (Valence Aware Dictionary and sEntiment Reasoner)** for sentiment analysis of Amazon fine food reviews.

**[Open the Colab Notebook](https://colab.research.google.com/github/ZeshanRasul/RoBERTa_VADER_NLP_Evaluation/blob/main/VADER_RoBERTa_Sentiment_Analysis_Comparison.ipynb) to explore the code, results and analysis.**

## Project Overview

- **Objective:** Compare a deep-learning based transformer model (**RoBERTa**) with a lexicon-based approach (**VADER**) for sentiment classification.
- **Dataset:** This project uses a Kaggle dataset of Amazon fine food reviews, which can be found [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
- **Models:**
  - **RoBERTa:** Pretrained transformer model trained to perform sentiment analysis.
  - **VADER:** Rule-based sentiment scoring system.
- **Evaluation:** Comparison of model analysis results using Seaborn. Inspection of edge cases.
- **Tools:** Python, Hugging Face, NumPy, Pandas, Matplotlib, Seaborn, NLTK.

## Key Features

- Preprocessing pipeline with tokenization with NLTK and Hugging Face tokenizer
- Sentiment classification using RoBERTa and VADER
- Model score comparisons using Seaborn
- Interactive Google Colab notebook with detailed, step-by-step explanations

## How to Use

- Run the Google Colab Notebook: [Click here to open](https://colab.research.google.com/github/ZeshanRasul/RoBERTa_VADER_NLP_Evaluation/blob/main/VADER_RoBERTa_Sentiment_Analysis_Comparison.ipynb)
- Clone this repo and run locally with your own modifications:
```bash
git clone https://github.com/ZeshanRasul/RoBERTa_VADER_NLP_Evaluation
cd RoBERTa_VADER_NLP_Evaluation
