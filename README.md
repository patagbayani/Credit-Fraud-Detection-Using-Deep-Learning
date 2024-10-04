# Credit Card Fraud Detection

This project demonstrates the detection of fraudulent credit card transactions using machine learning techniques. It includes data preprocessing, feature engineering, and model development to accurately identify fraudulent transactions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview

Credit card fraud is a significant issue, and detecting fraudulent transactions is crucial for reducing financial losses. This project builds a machine learning model to classify transactions as fraudulent or legitimate, utilizing various techniques such as data balancing, feature selection, and different classification algorithms.

### Objective:
- Build a machine learning pipeline to detect fraudulent credit card transactions.
- Experiment with different classification algorithms and compare performance.

## Dataset

The dataset used is a public dataset containing anonymized credit card transactions. The data includes various features representing transaction details and a target variable indicating whether a transaction is fraudulent.

- **Features:** Anonymized numerical features representing transaction details.
- **Target:** Binary variable where `1` represents a fraudulent transaction and `0` represents a legitimate transaction.

## Project Structure

- **CreditCardFraudDetection.ipynb:** The Jupyter notebook containing all the code for data preprocessing, model building, evaluation, and analysis.
- **data/**: Folder where you place the dataset.
- **models/**: (Optional) Directory where saved models can be stored.
- **README.md**: Project documentation.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/username/CreditCardFraudDetection.git
    ```
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
   The major libraries used include:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `keras`
   - `matplotlib`
   - `seaborn`

3. Download the dataset and place it in the `data/` directory.

## How to Run

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook CreditCardFraudDetection.ipynb
   ```
2. Follow the steps in the notebook to:

   - Load and preprocess the dataset using `pandas`.
   - Visualize the distribution of fraudulent vs. legitimate transactions using `matplotlib` or `seaborn`.
   - Handle imbalanced data by applying techniques such as:
     - Synthetic Minority Over-sampling Technique (SMOTE)
   - Engineer relevant features from the dataset, particularly scaling the `Amount` and `Time` columns, and normalizing the features.
   - Build and train machine learning models, such as:
     - Neural Networks (via `keras`)
   - Evaluate model performance using metrics like:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC Curve
   - Visualize the confusion matrix and other metrics to compare different models' performance.
   
3. Optionally, save trained models in the `models/` directory for future inference.

## Results

The notebook presents a detailed analysis of different models applied to the dataset. Below are some key takeaways:

- **Neural Network models** (using `keras`) achieve higher accuracy but are more computationally expensive.
  
The project demonstrates how well this model can identify fraudulent transactions from highly imbalanced data. Precision and recall are crucial metrics in this case, as minimizing false positives and negatives directly impacts the financial system.

### Performance Metrics

For each model, the following metrics are calculated:
- **Accuracy**: The overall correctness of the model.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC Score**: The area under the ROC curve, which measures the model's ability to distinguish between the classes.

## Contributing

Contributions are welcome! If you have any suggestions for improvements or want to add additional models/visualizations, feel free to fork the repository and create a pull request.

When contributing, please ensure your changes are well-documented and tested, following the repository's code structure.




