# Naive Bayes Classifier

This project implements a **Bernoulli Naive Bayes** classifier for credit risk prediction. The model includes functionality for fairness evaluation based on sensitive attributes like sex and age.

## Features

- **Bernoulli Naive Bayes**: Handles binary attribute distributions.
- **Maximum Likelihood Training**: Computes priors and attribute distributions.
- **Evaluation Metrics**: Computes accuracy on training and validation sets.
- **Fairness Metrics**: Evaluates disparate impact, group-conditioned error rates (FPR, FNR), and Calders & Verwer (CV) measure.
- **Sensitive Attributes**: Supports analysis using sex, age, or combined attributes.

## How to Run

The main entry point is `main.py`. Running it will train the model on the German Credit dataset and print accuracy and fairness metrics:

```bash
# Run Naive Bayes classifier
python main.py
```

The script will:
- Load and preprocess the dataset from the data/ folder.
- Convert continuous variables into categorical bins.
- Encode categorical variables with one-hot encoding.
- Balance the dataset by downsampling negative examples.
- Split data into training and validation sets.
- Train a Bernoulli Naive Bayes model.
- Print training accuracy, test accuracy, and fairness measures.

## Project Structure
```text
.
├── data/                   # Folder containing all dataset files
├── main.py                 # Main script to train and evaluate the model
├── models.py               # Implementation of NaiveBayes class
└── README.md               # Project documentation
