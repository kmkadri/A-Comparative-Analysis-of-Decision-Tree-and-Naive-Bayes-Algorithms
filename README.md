# Breast Cancer Diagnosis: A Comparative Analysis of Decision Tree and Naive Bayes Algorithms

## Introduction
This project aims to compare the performance of Decision Tree and Naive Bayes algorithms on a breast cancer dataset. The goal is to evaluate the effectiveness of these algorithms in predicting the presence or absence of breast cancer in patients. The motivation for this comparison is to identify which algorithm is better suited for breast cancer diagnosis and to understand the strengths and weaknesses of each approach.

## Dataset
The dataset used in this project is sourced from the UCI Machine Learning Repository. It consists of 570 rows and 32 columns, including 30 features and a target class. For this analysis, the top 10 features most strongly correlated with the diagnosis variable were selected.

## Analysis
The analysis includes:
- **Initial Data Exploration:** Basic statistical measures (mean, standard deviation, etc.) and visualization techniques such as heatmaps and bar plots.
- **Model Implementation:** 
  - **Decision Tree (DT):** Models were created with varying parameters, and hyperparameters optimized using grid search.
  - **Naive Bayes (NB):** Implemented and evaluated using accuracy, precision, recall, and F1 score.

## Experimental Results
- **Decision Tree:**
  - Training Accuracy: 96.29%
  - Test Accuracy: 96.55%
  - Precision: 95.76%
  - Recall: 96.40%
  - F1 Score: 96.08%
  - Evaluation Time: 0.078864 seconds
- **Naive Bayes:**
  - Training Accuracy: 93.95%
  - Test Accuracy: 100%
  - Precision: 93.48%
  - Recall: 93.73%
  - F1 Score: 93.60%
  - Evaluation Time: 0.35022 seconds

## Conclusion
- **Decision Tree:** The DT model demonstrated high accuracy on both training and test sets, indicating its effectiveness in classifying the dataset. However, it was prone to overfitting, particularly on the training data.
- **Naive Bayes:** The NB model, while slightly less accurate on the training set, showed superior generalization with a perfect accuracy on the test set, suggesting it may be more robust when applied to new data.

## Future Work
- Investigate the impact of different pre-processing techniques, such as feature scaling or feature selection, on model performance.
- Explore other validation methods, like k-fold cross-validation, to assess why DT performed well on the training set but not as well on the test set.

## How to Run the Project
1. Clone the repository.
2. Ensure you have MATLAB or an equivalent tool for running `.mat` and `.m` files.
3. Run the provided scripts (`.m` and `.mat` files) in MATLAB to reproduce the results.

## Dependencies
- MATLAB
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## References
- UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Additional references as cited in the project.
