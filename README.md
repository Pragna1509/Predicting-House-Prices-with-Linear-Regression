# Predicting House Prices with Linear Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pragna1509/Predicting-House-Prices-with-Linear-Regression/blob/main/Predicting%20House%20Prices%20with%20Linear%20Regression.ipynb)

A simple machine learning project that trains a **Linear Regression** model to predict house prices based on selected features from a housing dataset.

## Overview

The notebook (`Predicting House Prices with Linear Regression.ipynb`) walks through a standard regression workflow:

1. Load the dataset with `pandas`
2. Explore it with `.head()`, `.info()`, and `.describe()`
3. Handle missing values using forward-fill (`fillna(method='ffill')`)
4. Select relevant features (`X`) and the target column, `house_price` (`y`)
5. Split the data into training and testing sets (80/20 split)
6. Train a `LinearRegression` model from `scikit-learn`
7. Predict house prices on the test set
8. Evaluate the model using **Mean Squared Error (MSE)** and **R-squared (R²)**
9. Visualize actual vs. predicted prices with a scatter plot

## Requirements

- Python 3.7+
- Jupyter Notebook or Google Colab
- Required packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

## Installation

```bash
git clone https://github.com/Pragna1509/Predicting-House-Prices-with-Linear-Regression.git
cd Predicting-House-Prices-with-Linear-Regression
pip install -r requirements.txt
```

## Usage

1. Add your dataset as a CSV file in the project folder, and update this line in the notebook to point to it:
   ```python
   data = pd.read_csv('your_dataset.csv')
   ```
2. Update the feature list to match your dataset's actual column names:
   ```python
   X = data[['feature1', 'feature2', ...]]
   y = data['house_price']
   ```
3. Run the notebook cell by cell — either locally with Jupyter:
   ```bash
   jupyter notebook "Predicting House Prices with Linear Regression.ipynb"
   ```
   or open it directly in Google Colab using the badge above.
4. The notebook will print the dataset preview, summary stats, model evaluation metrics (MSE and R²), and display a scatter plot comparing actual vs. predicted prices.

## Model Evaluation

The model's accuracy is measured with:
- **Mean Squared Error (MSE)** — average squared difference between predicted and actual prices (lower is better)
- **R-squared (R²)** — proportion of variance in house prices explained by the model (closer to 1 is better)

## Project Structure

```
.
├── Predicting House Prices with Linear Regression.ipynb   # Main notebook
├── your_dataset.csv                                        # Dataset (add your own)
├── requirements.txt                                        # Python dependencies
├── .gitignore                                              # Files/folders excluded from git
└── README.md                                                # This file
```

## Notes & Suggested Improvements

- **Dataset placeholder** — the notebook currently references `'your_dataset.csv'` and generic `'feature1'`, `'feature2'` columns. Replace these with your actual file name and column names before running.
- **Missing value handling** — forward-fill (`ffill`) is used, which may not be ideal for all columns (e.g., numeric columns might be better served by mean/median imputation). Consider tailoring this per column.
- **Categorical features** — if your dataset includes non-numeric columns (like location or property type), these will need to be encoded (e.g., one-hot encoding) before being passed into `LinearRegression`.
- **Feature scaling** — Linear Regression can benefit from scaled features, especially if they're on very different ranges (e.g., area in sq. ft. vs. number of bedrooms).
- **Further evaluation** — consider adding cross-validation, or comparing against other models (Ridge, Lasso, Random Forest) for a stronger baseline.

## License

This project is open source and available for learning and personal use.
