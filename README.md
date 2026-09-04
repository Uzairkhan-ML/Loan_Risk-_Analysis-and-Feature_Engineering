# Loan Risk Analysis & Feature Engineering
A Python project for credit risk analysis and feature engineering that cleans raw loan data, handles outliers via percentile clipping, and constructs Debt-to-Income (DTI) ratios and risk categories to prepare datasets for machine learning models.


This repository contains a Python script for performing exploratory data analysis (EDA), data cleaning, outlier handling, and feature engineering on credit risk data (`loan_data.csv`). The pipeline prepares raw credit and income features for machine learning models by engineering risk indicators and analyzing historical default rates.

## Features & Processing Pipeline

* **Missing Value Imputation:** Fills missing entries in the `income` column using median value imputation.
* **Correlation Analysis:** Generates a Seaborn heatmap to evaluate linear relationships across loan features.
* **Outlier Detection & Clipping:** Identifies extreme values in `loan_amount` using boxplots and clips them between the 5th and 95th percentiles to create `loan_amount_cleaned`.
* **Feature Engineering:**
  * **Debt-to-Income Ratio (`dti_ratio`):** Calculates borrower leverage by dividing cleaned loan amounts by annual income (`loan_amount_cleaned / income`).
  * **Risk Categorization (`risk_category`):** Bins credit scores into standardized risk buckets (`High Risk`, `Fair`, `Good`, `Execellent`) using `pd.cut()`.
* **Default Rate Analysis:** Measures the historical default percentage across engineered credit risk categories.

## Visualizations Included

* **Correlation Heatmap:** Color-coded matrix showing variable interactions.
* **Raw Outlier Boxplot:** Visualizes extreme distribution tails in uncleaned loan amounts.
* **Clipped Distribution Boxplot:** Displays the distribution after applying 5th-95th percentile quantile boundaries.

## Technologies Used

* **Python**
* **Pandas** (Data cleaning, binning, and aggregation)
* **Seaborn & Matplotlib** (Statistical visualizations and boxplots)

## How to Run

1. Clone this repository to your local machine.
2. Ensure `loan_data.csv` is placed in the project root directory.
3. Install the required dependencies:
   ```bash
   pip install pandas seaborn matplotlib
