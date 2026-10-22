# Machine Learning 

A collection of machine learning and statistical learning practical exercises implemented in Python. This repository documents exploratory data analysis, regression, classification, resampling, regularization, and dimension-reduction methods using datasets from the `ISLP` package and related CSV files.

## Project Overview

The practical record applies statistical learning techniques to real-world-style datasets to understand data distributions, explore relationships between variables, fit predictive models, evaluate classification and regression performance, and compare alternative modelling approaches.

**Primary language:** Python  
**Environment:** Jupyter Notebook   
**Main libraries:** pandas, NumPy, Matplotlib, Seaborn, statsmodels, scikit-learn, ISLP

## Objectives

- Explore datasets using descriptive statistics and visualizations.
- Identify relationships between predictors and response variables.
- Build and interpret simple and multiple linear regression models.
- Apply logistic regression and classification algorithms.
- Evaluate models with confusion matrices, accuracy, test error, and mean squared error.
- Estimate model performance using validation sets and leave-one-out cross-validation (LOOCV).
- Compare regularization and dimension-reduction approaches.
- Interpret statistical significance, model coefficients, and practical limitations.

## Practical Exercises

The following summary follows the topics visible in the practical record. The record contains exercises on the topics listed below; check the notebook or report for the exact implementation and outputs.

| Exercise | Topic | Dataset(s) | Main methods / concepts |
|---|---|---|---|
| 1 | Exploratory Data Analysis (EDA) | Auto | Variable types, ranges, means, standard deviations, pairplots, scatterplots, predictor relationships |
| 2 | Exploring a college dataset | College | Descriptive statistics, scatterplot matrix, boxplots, histograms, categorical grouping, correlations |
| 3 | Multiple linear regression | Auto | Correlation matrix, OLS regression, ANOVA, coefficient significance, residual diagnostics, interaction terms |
| 4 | Linear regression model building | Boston | Simple and multiple linear regression, p-values, correlation heatmap, model interpretation |
| 5 | Classification algorithms | Weekly | Logistic regression, confusion matrix, validation period, LDA, QDA, K-nearest neighbours (KNN) |
| 6 | High/low mileage classification | Auto | Median-based binary target, train/test split, LDA, QDA, logistic regression, KNN, test error |
| 7 | Validation-set approach | Default | Logistic regression, train/validation split, repeated splits, test error comparison |
| 8 | Leave-One-Out Cross-Validation (LOOCV) | Weekly | Logistic regression, repeated fitting, predicted probabilities, LOOCV error |
| 9 | Regularization and dimension reduction | College | Linear regression, Ridge, Lasso, Principal Component Regression (PCR), Partial Least Squares (PLS), cross-validation |
| 10 | Estimating a population mean and its uncertainty | Boston | Sample mean, standard error, bootstrap estimation |

## Datasets Used

The practical record uses datasets including:

- **Auto** — vehicle characteristics and fuel mileage (`mpg`).
- **College** — college characteristics, applications, enrolment, tuition, and graduation-related variables.
- **Boston** — housing-area characteristics, including median home value (`medv`) and crime rate (`crim`).
- **Weekly** — weekly market returns and market direction.
- **Default** — customer default status, income, balance, and student status.

Several datasets are loaded from the `ISLP` Python package. The College exercise also demonstrates reading a CSV file with pandas.

## Tools and Libraries

- **Python** — implementation and analysis.
- **pandas** — data loading, cleaning, manipulation, and descriptive summaries.
- **NumPy** — numerical operations and target-variable creation.
- **Matplotlib** — plotting and visualization.
- **Seaborn** — statistical plots, pairplots, boxplots, and heatmaps.
- **statsmodels** — OLS and generalized linear models, including logistic regression, statistical summaries, and ANOVA.
- **scikit-learn** — train/test splitting, LDA, QDA, KNN, scaling, cross-validation, Ridge, Lasso, PCA, PLS, and model metrics.
- **ISLP** — textbook datasets and selected statistical-learning utilities.

## Methods Covered

### Exploratory Data Analysis
- Dataset inspection and descriptive statistics
- Quantitative and qualitative variable identification
- Range, mean, and standard deviation
- Histograms, scatterplots, pairplots, boxplots, and correlation heatmaps
- Exploration of relationships and possible multicollinearity

### Regression
- Simple linear regression
- Multiple linear regression
- Logistic regression
- Coefficient estimates and p-values
- R-squared and adjusted R-squared
- ANOVA for model comparison
- Residual diagnostics and interaction terms

### Classification
- Binary target creation
- Logistic regression
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- K-Nearest Neighbours (KNN)
- Confusion matrices, accuracy, and classification error

### Resampling and Model Evaluation
- Training and test sets
- Validation-set approach
- Repeated random splits
- Leave-One-Out Cross-Validation (LOOCV)
- Cross-validation for hyperparameter selection

### Regularization and Dimension Reduction
- Ordinary least squares
- Ridge regression
- Lasso regression
- Principal Component Regression (PCR)
- Partial Least Squares (PLS)
- Feature scaling and one-hot encoding

## Key Findings Highlighted in the Record

- In the Auto dataset, fuel mileage tends to be lower for heavier, higher-horsepower, and larger-displacement cars; model year is positively associated with mileage.
- In the College dataset, applications and enrolment tend to increase together. Private and elite colleges show higher out-of-state tuition in the reported comparisons.
- In the Auto multiple-regression exercise, the full model explains a substantial portion of variation in mileage, while correlations among predictors mean individual coefficients should be interpreted carefully.
- In the Weekly classification exercise, market direction is difficult to predict reliably from lagged returns alone; the reported test performance is only modestly above a simple baseline.
- In the high/low mileage classification exercise, LDA and QDA perform well on the recorded split; KNN results require careful checking because the documented K=3 and K=5 cells appear to use `n_neighbors=1`.
- In the Default exercise, the validation error is low, but the default class is uncommon, so accuracy/error should be compared with a majority-class baseline.
- In the College applications exercise, Linear Regression, Ridge, Lasso, PCR, and PLS have very similar recorded test errors; Ridge is marginally best in the report, with a negligible difference.

These are summaries of the report's stated interpretations.

## Installation


Install the libraries:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn ISLP jupyter
```

## Running the Practicals

For datasets supplied by `ISLP`, use the loading pattern shown in the practical record:

```python
from ISLP import load_data

auto = load_data("Auto")
```

For a local CSV dataset:

```python
import pandas as pd

college = pd.read_csv("College.csv", index_col=0)
```

## Repository Structure

```text
machine-learning/
├── README.md
├── 01_eda_auto.ipynb
├── 02_college_eda.ipynb
├── 03_multiple_linear_regression_auto.ipynb
├── 04_linear_regression_boston.ipynb
├── 05_weekly_classification.ipynb
├── 06_auto_mileage_classification.ipynb
├── 07_validation_set_default.ipynb
├── 08_loocv_weekly.ipynb
├── 09_regularization_college.ipynb
├── 10_bootstrap_boston.ipynb
```

## Important Reproducibility Notes

- Some datasets are loaded from `ISLP`; install the package before running those cells.
- Random train/test splits can change results unless a `random_state` or random seed is fixed.
- KNN is distance-based and generally benefits from standardizing features.
- Check the K value in each KNN model before comparing results.
- Ensure that the response variable and predictors are correctly aligned after train/test splitting.
- For imbalanced classification, compare results against a baseline that predicts the majority class.
- Check that each GLM intended for logistic regression uses a binomial family.
- Some code cells in the report appear to contain variable-name inconsistencies; verify variable definitions and rerun cells before treating results as reproducible.
- Dataset names, metrics, and conclusions in this README should be updated if your final runnable code differs from the PDF.

## Learning Outcomes

By completing these practicals, a learner gains hands-on experience with data exploration, statistical inference, regression modelling, classification, resampling-based evaluation, regularization, dimension reduction, and interpreting model performance in Python.

## Author

**Name:** ROBISHA R P 
**Course:** MBA   
**Institution:** RVS College of Arts and Science


