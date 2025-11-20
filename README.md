# ML Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## Overview

An advanced machine learning pipeline for housing price prediction leveraging ensemble methods, stacking models, and automated feature selection. This project implements multiple state-of-the-art regression algorithms with comprehensive hyperparameter optimization to achieve competitive prediction accuracy.

## Key Features

- **Multiple Regression Models**: Random Forest, Gradient Boosting, MLP, SVR, Decision Tree
- **Ensemble Learning**: One-layer stacked model combining multiple base regressors
- **Feature Engineering**: Automated feature selection using SelectFromModel, SequentialFeatureSelector, and GeneticSelectionCV
- **Hyperparameter Optimization**: Grid search and randomized search for optimal model parameters
- **Comprehensive EDA**: Systematic exploratory data analysis and visualization
- **Data Preprocessing**: Robust data cleaning, normalization, and transformation pipelines
- **Model Evaluation**: Cross-validation, learning curves, and performance metrics analysis

## Tech Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Feature Selection**: sklearn-genetic, mlxtend

## Project Structure

```
ML-Housing-Price-Prediction/
│
├── ML Python.ipynb          # Main Jupyter notebook with complete pipeline
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore               # Git ignore file
```

## Models Implemented

### Base Models
1. **Random Forest Regressor** - Ensemble of decision trees
2. **Gradient Boosting Regressor** - Sequential ensemble method
3. **Multi-Layer Perceptron (MLP)** - Neural network regressor
4. **Support Vector Regressor (SVR)** - Kernel-based regression
5. **Decision Tree Regressor** - Single tree model

### Meta Model
- **Stacked Regressor** - Combines predictions from multiple base models for improved accuracy

## Installation

```bash
# Clone the repository
git clone https://github.com/RaviTeja-Kondeti/ML-Housing-Price-Prediction.git
cd ML-Housing-Price-Prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Open the Jupyter notebook
jupyter notebook "ML Python.ipynb"

# Run all cells to:
# 1. Load and explore the dataset
# 2. Perform feature engineering
# 3. Train multiple models
# 4. Evaluate model performance
# 5. Generate predictions
```

## Pipeline Workflow

1. **Data Loading & Exploration**
   - Import housing dataset
   - Statistical analysis
   - Distribution visualization

2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical features
   - Scale numerical features
   - Remove outliers

3. **Feature Selection**
   - Correlation analysis
   - Recursive feature elimination
   - Genetic algorithm selection

4. **Model Training**
   - Train individual base models
   - Hyperparameter tuning
   - Cross-validation

5. **Model Stacking**
   - Combine base model predictions
   - Train meta-learner

6. **Evaluation & Results**
   - RMSE, MAE, R² scores
   - Residual analysis
   - Feature importance

## Performance Metrics

The models are evaluated using:
- **R² Score** (Coefficient of Determination)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Cross-Validation Score** (5-fold)

## Results

The stacked ensemble model achieves competitive Kaggle scores through:
- Systematic feature engineering
- Optimal hyperparameter configuration
- Robust cross-validation strategy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Ravi Teja Kondeti**
- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)

## Acknowledgments

- Dataset: Housing prices dataset
- Inspired by Kaggle competitions and best practices in ML pipelines
- Built using open-source libraries and frameworks

---

⭐ If you find this project useful, please consider giving it a star!
