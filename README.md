# UPI Fraud Statistical Signals

## Project Overview

UPI Fraud Statistical Signals is a statistical analysis project that compares normal and fraudulent transactions to identify measurable fraud-signal differences.

## Dataset

Dataset Used: Credit Card Fraud Detection Dataset (`creditcard.csv`)

The analysis uses the 28 V features from `V1` to `V28`.

## Objective

The main objective of this project is to statistically compare normal and fraudulent transactions and identify measurable differences using statistical methods.

## Statistical Analysis Performed

The project includes:

- Variance Analysis
- Skewness Analysis
- Outlier Detection
- Kurtosis Analysis
- Median Comparison
- Quartile and IQR Analysis
- Mean Comparison
- Minimum Comparison
- Maximum Comparison
- Range Comparison
- Standard Deviation Comparison
- Correlation Analysis
- Covariance Analysis
- Coefficient of Variation Analysis
- Z-Score Analysis
- Statistical Significance Analysis
- Feature-wise Fraud Signal Ranking
- Top Fraud Signal Identification
- Transaction Amount Analysis
- Fraud vs Normal Amount Comparison
- Final Statistical Summary
- Final Fraud-Signal Report
- Final Project Conclusion

## Final Results

- Features Analyzed: 28
- Statistically Significant Features (p < 0.05): 24
- Top Combined Fraud Signal Feature: V7
- Top Combined Fraud Signal Score: 60.0
- Highest Absolute Correlation Feature: V17
- Correlation with Class: -0.326481
- Largest Mean Difference Feature: V3
- Largest Median Difference Feature: V14
- Largest Variance Difference Feature: V7
- Largest Kurtosis Difference Feature: V28
- Largest Z-Score Difference Feature: V28
- Top Fraud Outlier Feature: V14
- Top Fraud Outlier Percentage: 87.398374
- Normal Transaction Amount Mean: 88.291022
- Fraud Transaction Amount Mean: 122.211321
- Transaction Amount Mean Difference: 33.920299

## Project Outputs

The analysis generates CSV result files for each statistical step, including:

- `step23_minimum_comparison.csv`
- `step24_maximum_comparison.csv`
- `step25_range_comparison.csv`
- `step26_standard_deviation_comparison.csv`
- `step27_correlation_analysis.csv`
- `step28_covariance_analysis.csv`
- `step29_coefficient_of_variation_analysis.csv`
- `step30_zscore_analysis.csv`
- `step31_statistical_significance_analysis.csv`
- `step32_feature_wise_fraud_signal_ranking.csv`
- `step33_top_fraud_signal_identification.csv`
- `step34_transaction_amount_analysis.csv`
- `step35_fraud_vs_normal_amount_comparison.csv`
- `step36_final_statistical_summary.csv`
- `step37_final_fraud_signal_report.csv`
- `step38_final_project_conclusion.csv`

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy

## How to Run

1. Keep `analysis.py` and `creditcard.csv` in the project folder.
2. Open Terminal in the project folder.
3. Run:

```bash
python analysis.py