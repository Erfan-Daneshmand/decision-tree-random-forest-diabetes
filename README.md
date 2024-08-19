# Diabetes Risk Prediction Using Decision Tree and Random Forest Models

This project aims to predict the risk of diabetes in patients using medical data. The models used in this project are Decision Tree and Random Forest, and the project includes steps such as data preprocessing, model training, hyperparameter tuning, and evaluation. This repository contains the code, data, and a detailed report on the process and outcomes.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Selection](#model-selection)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Feature Importance Analysis](#feature-importance-analysis)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Challenges and Learnings](#challenges-and-learnings)
- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to classify whether a patient has diabetes based on several medical attributes such as glucose level, BMI, and age. The classification models used are Decision Tree and Random Forest, which are powerful tools for handling classification tasks with structured data. This project explores the end-to-end machine learning pipeline, from data preprocessing to model evaluation.

## Dataset
The dataset used for this project is the Pima Indians Diabetes Database, which includes the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function that represents diabetes history in relatives
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1), 0 indicates non-diabetic and 1 indicates diabetic

## Project Workflow

### Data Preprocessing
The initial stage involved cleaning and preparing the data:
- **Handling Missing Values:** Some features had zero values, which were treated as missing and imputed using the median value of the respective feature.
- **Feature Scaling:** Although scaling is not necessary for tree-based models, it was considered to ensure consistency across the pipeline.

### Model Selection
Two models were selected for this project:
- **Decision Tree:** A simple, interpretable model that builds a tree structure where nodes represent decisions based on feature values.
- **Random Forest:** An ensemble method that constructs multiple decision trees and merges them to reduce variance and improve accuracy.

### Hyperparameter Tuning
The performance of the models, particularly the Random Forest, was improved through hyperparameter tuning using `GridSearchCV`. The parameters tuned included:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `n_estimators`
- `max_features`

### Feature Importance Analysis
The Random Forest model provided insights into which features were most important for predicting diabetes. Features such as `Glucose`, `BMI`, and `Age` were identified as key predictors.

### Model Evaluation
The models were evaluated using a variety of metrics:
- **Accuracy:** Overall correctness of the model.
- **Precision:** Proportion of true positive predictions among all positive predictions.
- **Recall:** Proportion of true positive predictions among all actual positives.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC Score:** Measure of the model's ability to distinguish between classes.

## Results
- **Best Model:** The Random Forest model with the following hyperparameters:
  - `max_depth`: 7
  - `min_samples_leaf`: 1
  - `min_samples_split`: 7
  - `n_estimators`: 100
- **Performance:**
  - **Cross-Validation Accuracy:** 78.52%
  - **Test Accuracy:** 77.92%
  - **Precision:** 69.05%
  - **Recall:** 58.18%
  - **ROC-AUC Score:** Demonstrated good discriminative ability.

## Challenges and Learnings
- **Overfitting:** Initial models showed signs of overfitting, which was mitigated through regularization and hyperparameter tuning.
- **Computational Resources:** GridSearchCV was computationally intensive, requiring careful selection of parameters to optimize without overloading resources.
- **Feature Importance:** Analyzing feature importance helped in understanding the model and provided insights into potential feature engineering opportunities.

## Future Work
- **Ensemble Methods:** Implementing a stacking ensemble combining Random Forest with other models like Gradient Boosting Machines could further improve performance.
- **Model Interpretability:** Using tools like SHAP or LIME to better understand and explain the model's predictions.
- **Feature Engineering:** Further exploration of new features or transformations based on domain knowledge.

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-ml.git
   cd diabetes-prediction-ml
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook:**
   Open the `RF and DT.ipynb` file in Jupyter Notebook to explore the code and results.

## Contributing
Contributions are welcome! If you have suggestions for improving the model, enhancing the documentation, or adding new features, feel free to open a pull request or issue.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
