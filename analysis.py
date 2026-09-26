import pandas as pd

# ============================================================
# UPI FRAUD STATISTICAL SIGNALS
# CREDIT CARD FRAUD DETECTION DATASET
# ============================================================

# ------------------------------------------------------------
# STEP 1 - LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("creditcard.csv")

print("=" * 60)
print("STEP 1 - LOAD DATASET")
print("=" * 60)
print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print()


# ------------------------------------------------------------
# STEP 2 - DATASET STRUCTURE
# ------------------------------------------------------------

print("=" * 60)
print("STEP 2 - DATASET STRUCTURE")
print("=" * 60)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print()


# ------------------------------------------------------------
# STEP 3 - FIRST FIVE RECORDS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 3 - FIRST FIVE RECORDS")
print("=" * 60)
print(df.head())
print()


# ------------------------------------------------------------
# STEP 4 - DATASET INFORMATION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 4 - DATASET INFORMATION")
print("=" * 60)
print(df.info())
print()


# ------------------------------------------------------------
# STEP 5 - MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 5 - MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("Missing Values:")
print(missing_values)
print()

print("Total Missing Values:", missing_values.sum())
print()


# ------------------------------------------------------------
# STEP 6 - CLASS DISTRIBUTION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 6 - CLASS DISTRIBUTION")
print("=" * 60)

class_distribution = df["Class"].value_counts()

print(class_distribution)
print()


# ------------------------------------------------------------
# STEP 7 - FRAUD PERCENTAGE
# ------------------------------------------------------------

print("=" * 60)
print("STEP 7 - FRAUD PERCENTAGE")
print("=" * 60)

total_transactions = len(df)
fraud_transactions = (df["Class"] == 1).sum()
normal_transactions = (df["Class"] == 0).sum()

fraud_percentage = (fraud_transactions / total_transactions) * 100
normal_percentage = (normal_transactions / total_transactions) * 100

print("Total Transactions:", total_transactions)
print("Normal Transactions:", normal_transactions)
print("Fraud Transactions:", fraud_transactions)
print("Normal Percentage:", round(normal_percentage, 4), "%")
print("Fraud Percentage:", round(fraud_percentage, 4), "%")
print()


# ------------------------------------------------------------
# STEP 8 - FEATURE SELECTION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 8 - FEATURE SELECTION")
print("=" * 60)

features = [f"V{i}" for i in range(1, 29)]

print("Selected Features:")
print(features)
print()


# ------------------------------------------------------------
# STEP 9 - DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 9 - DESCRIPTIVE STATISTICS")
print("=" * 60)

descriptive_statistics = df[features].describe()

print(descriptive_statistics)
print()


# ------------------------------------------------------------
# STEP 10 - MEAN ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 10 - MEAN ANALYSIS")
print("=" * 60)

mean_values = df[features].mean()

print("Mean Values:")
print(mean_values)
print()


# ------------------------------------------------------------
# STEP 11 - MEDIAN ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 11 - MEDIAN ANALYSIS")
print("=" * 60)

median_values = df[features].median()

print("Median Values:")
print(median_values)
print()


# ------------------------------------------------------------
# STEP 12 - STANDARD DEVIATION ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 12 - STANDARD DEVIATION ANALYSIS")
print("=" * 60)

std_values = df[features].std()

print("Standard Deviation Values:")
print(std_values)
print()


# ------------------------------------------------------------
# STEP 13 - VARIANCE ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 13 - VARIANCE ANALYSIS")
print("=" * 60)

variance_values = df[features].var()

print("Variance Values:")
print(variance_values)
print()


# ------------------------------------------------------------
# STEP 14 - SKEWNESS ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 14 - SKEWNESS ANALYSIS")
print("=" * 60)

skewness_values = df[features].skew()

print("Skewness Values:")
print(skewness_values)
print()


# ------------------------------------------------------------
# STEP 15 - KURTOSIS ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 15 - KURTOSIS ANALYSIS")
print("=" * 60)

kurtosis_values = df[features].kurtosis()

print("Kurtosis Values:")
print(kurtosis_values)
print()


# ------------------------------------------------------------
# STEP 16 - FRAUD VS NORMAL VARIANCE
# ------------------------------------------------------------

print("=" * 60)
print("STEP 16 - FRAUD VS NORMAL VARIANCE")
print("=" * 60)

normal_data = df[df["Class"] == 0]
fraud_data = df[df["Class"] == 1]

normal_variance = normal_data[features].var()
fraud_variance = fraud_data[features].var()

variance_comparison = pd.DataFrame({
    "Normal Variance": normal_variance,
    "Fraud Variance": fraud_variance
})

variance_comparison["Variance Difference"] = (
    variance_comparison["Fraud Variance"]
    - variance_comparison["Normal Variance"]
)

variance_comparison["Absolute Variance Difference"] = (
    variance_comparison["Variance Difference"].abs()
)

print(variance_comparison)
print()

largest_variance_feature = variance_comparison[
    "Absolute Variance Difference"
].idxmax()

largest_variance_difference = variance_comparison.loc[
    largest_variance_feature,
    "Absolute Variance Difference"
]

print("Largest Variance Difference Feature:", largest_variance_feature)
print(
    "Largest Absolute Variance Difference:",
    round(largest_variance_difference, 6)
)
print()


# ------------------------------------------------------------
# STEP 17 - FRAUD VS NORMAL SKEWNESS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 17 - FRAUD VS NORMAL SKEWNESS")
print("=" * 60)

normal_skewness = normal_data[features].skew()
fraud_skewness = fraud_data[features].skew()

skewness_comparison = pd.DataFrame({
    "Normal Skewness": normal_skewness,
    "Fraud Skewness": fraud_skewness
})

skewness_comparison["Skewness Difference"] = (
    skewness_comparison["Fraud Skewness"]
    - skewness_comparison["Normal Skewness"]
)

skewness_comparison["Absolute Skewness Difference"] = (
    skewness_comparison["Skewness Difference"].abs()
)

print(skewness_comparison)
print()

largest_skewness_feature = skewness_comparison[
    "Absolute Skewness Difference"
].idxmax()

largest_skewness_difference = skewness_comparison.loc[
    largest_skewness_feature,
    "Absolute Skewness Difference"
]

print("Largest Skewness Difference Feature:", largest_skewness_feature)
print(
    "Largest Absolute Skewness Difference:",
    round(largest_skewness_difference, 6)
)
print()


# ------------------------------------------------------------
# STEP 18 - V FEATURE OUTLIER DETECTION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 18 - V FEATURE OUTLIER DETECTION")
print("=" * 60)

outlier_results = []

for feature in features:

    Q1 = fraud_data[feature].quantile(0.25)
    Q3 = fraud_data[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    fraud_outliers = fraud_data[
        (fraud_data[feature] < lower_bound)
        | (fraud_data[feature] > upper_bound)
    ]

    outlier_count = len(fraud_outliers)

    outlier_percentage = (
        outlier_count / len(fraud_data)
    ) * 100

    outlier_results.append({
        "Feature": feature,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Fraud Outliers": outlier_count,
        "Fraud Outlier Percentage": outlier_percentage
    })

outlier_df = pd.DataFrame(outlier_results)

top_outlier_feature = outlier_df.loc[
    outlier_df["Fraud Outliers"].idxmax(),
    "Feature"
]

top_outlier_count = outlier_df["Fraud Outliers"].max()

top_outlier_percentage = outlier_df.loc[
    outlier_df["Fraud Outliers"].idxmax(),
    "Fraud Outlier Percentage"
]

print(outlier_df.sort_values(
    by="Fraud Outliers",
    ascending=False
).head(10))
print()

print("Top Fraud Outlier Feature:", top_outlier_feature)
print("Fraud Outliers:", top_outlier_count)
print(
    "Fraud Outlier Percentage:",
    round(top_outlier_percentage, 2),
    "%"
)
print()


# ------------------------------------------------------------
# STEP 19 - FRAUD VS NORMAL KURTOSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 19 - FRAUD VS NORMAL KURTOSIS")
print("=" * 60)

normal_kurtosis = normal_data[features].kurtosis()
fraud_kurtosis = fraud_data[features].kurtosis()

kurtosis_comparison = pd.DataFrame({
    "Normal Kurtosis": normal_kurtosis,
    "Fraud Kurtosis": fraud_kurtosis
})

kurtosis_comparison["Kurtosis Difference"] = (
    kurtosis_comparison["Fraud Kurtosis"]
    - kurtosis_comparison["Normal Kurtosis"]
)

kurtosis_comparison["Absolute Kurtosis Difference"] = (
    kurtosis_comparison["Kurtosis Difference"].abs()
)

print(kurtosis_comparison)
print()

largest_kurtosis_feature = kurtosis_comparison[
    "Absolute Kurtosis Difference"
].idxmax()

largest_kurtosis_difference = kurtosis_comparison.loc[
    largest_kurtosis_feature,
    "Absolute Kurtosis Difference"
]

print("Largest Kurtosis Difference Feature:", largest_kurtosis_feature)
print(
    "Largest Absolute Kurtosis Difference:",
    round(largest_kurtosis_difference, 6)
)
print()


# ------------------------------------------------------------
# STEP 20 - FRAUD VS NORMAL MEDIAN
# ------------------------------------------------------------

print("=" * 60)
print("STEP 20 - FRAUD VS NORMAL MEDIAN")
print("=" * 60)

normal_median = normal_data[features].median()
fraud_median = fraud_data[features].median()

median_comparison = pd.DataFrame({
    "Normal Median": normal_median,
    "Fraud Median": fraud_median
})

median_comparison["Median Difference"] = (
    median_comparison["Fraud Median"]
    - median_comparison["Normal Median"]
)

median_comparison["Absolute Median Difference"] = (
    median_comparison["Median Difference"].abs()
)

print(median_comparison)
print()

largest_median_feature = median_comparison[
    "Absolute Median Difference"
].idxmax()

largest_median_difference = median_comparison.loc[
    largest_median_feature,
    "Absolute Median Difference"
]

print("Largest Median Difference Feature:", largest_median_feature)
print(
    "Largest Absolute Median Difference:",
    round(largest_median_difference, 6)
)
print()


# ------------------------------------------------------------
# STEP 21 - FRAUD VS NORMAL QUARTILE
# ------------------------------------------------------------

print("=" * 60)
print("STEP 21 - FRAUD VS NORMAL QUARTILE")
print("=" * 60)

normal_Q1 = normal_data[features].quantile(0.25)
fraud_Q1 = fraud_data[features].quantile(0.25)

normal_Q3 = normal_data[features].quantile(0.75)
fraud_Q3 = fraud_data[features].quantile(0.75)

normal_IQR = normal_Q3 - normal_Q1
fraud_IQR = fraud_Q3 - fraud_Q1

quartile_comparison = pd.DataFrame({
    "Normal Q1": normal_Q1,
    "Fraud Q1": fraud_Q1,
    "Normal Q3": normal_Q3,
    "Fraud Q3": fraud_Q3,
    "Normal IQR": normal_IQR,
    "Fraud IQR": fraud_IQR
})

quartile_comparison["IQR Difference"] = (
    quartile_comparison["Fraud IQR"]
    - quartile_comparison["Normal IQR"]
)

quartile_comparison["Absolute IQR Difference"] = (
    quartile_comparison["IQR Difference"].abs()
)

print(quartile_comparison)
print()

largest_IQR_feature = quartile_comparison[
    "Absolute IQR Difference"
].idxmax()

largest_IQR_difference = quartile_comparison.loc[
    largest_IQR_feature,
    "Absolute IQR Difference"
]

print("Largest IQR Difference Feature:", largest_IQR_feature)
print(
    "Largest Absolute IQR Difference:",
    round(largest_IQR_difference, 6)
)
print()


# ------------------------------------------------------------
# STEP 22 - FRAUD VS NORMAL MEAN
# ------------------------------------------------------------

print("=" * 60)
print("STEP 22 - FRAUD VS NORMAL MEAN")
print("=" * 60)

normal_mean = normal_data[features].mean()
fraud_mean = fraud_data[features].mean()

mean_comparison = pd.DataFrame({
    "Normal Mean": normal_mean,
    "Fraud Mean": fraud_mean
})

mean_comparison["Mean Difference"] = (
    mean_comparison["Fraud Mean"]
    - mean_comparison["Normal Mean"]
)

mean_comparison["Absolute Mean Difference"] = (
    mean_comparison["Mean Difference"].abs()
)

print(mean_comparison)
print()

largest_mean_feature = mean_comparison[
    "Absolute Mean Difference"
].idxmax()

largest_mean_difference = mean_comparison.loc[
    largest_mean_feature,
    "Absolute Mean Difference"
]

print("Largest Mean Difference Feature:", largest_mean_feature)
print(
    "Largest Absolute Mean Difference:",
    round(largest_mean_difference, 6)
)
print()


# ------------------------------------------------------------
# STEP 23 - FRAUD VS NORMAL MINIMUM
# ------------------------------------------------------------

print("=" * 60)
print("STEP 23 - FRAUD VS NORMAL MINIMUM")
print("=" * 60)

normal_minimum = normal_data[features].min()
fraud_minimum = fraud_data[features].min()

minimum_comparison = pd.DataFrame({
    "Normal Minimum": normal_minimum,
    "Fraud Minimum": fraud_minimum
})

minimum_comparison["Minimum Difference"] = (
    minimum_comparison["Fraud Minimum"]
    - minimum_comparison["Normal Minimum"]
)

minimum_comparison["Absolute Minimum Difference"] = (
    minimum_comparison["Minimum Difference"].abs()
)

print(minimum_comparison)
print()

largest_minimum_feature = minimum_comparison[
    "Absolute Minimum Difference"
].idxmax()

largest_minimum_difference = minimum_comparison.loc[
    largest_minimum_feature,
    "Absolute Minimum Difference"
]

print("Largest Minimum Difference Feature:", largest_minimum_feature)
print(
    "Largest Absolute Minimum Difference:",
    round(largest_minimum_difference, 6)
)

minimum_comparison.to_csv(
    "step23_minimum_comparison.csv"
)

print()
print("Result File:")
print("step23_minimum_comparison.csv")
print()


# ------------------------------------------------------------
# STEP 24 - FRAUD VS NORMAL MAXIMUM
# ------------------------------------------------------------

print("=" * 60)
print("STEP 24 - FRAUD VS NORMAL MAXIMUM")
print("=" * 60)

normal_maximum = normal_data[features].max()
fraud_maximum = fraud_data[features].max()

maximum_comparison = pd.DataFrame({
    "Normal Maximum": normal_maximum,
    "Fraud Maximum": fraud_maximum
})

maximum_comparison["Maximum Difference"] = (
    maximum_comparison["Fraud Maximum"]
    - maximum_comparison["Normal Maximum"]
)

maximum_comparison["Absolute Maximum Difference"] = (
    maximum_comparison["Maximum Difference"].abs()
)

print(maximum_comparison)
print()

largest_maximum_feature = maximum_comparison[
    "Absolute Maximum Difference"
].idxmax()

largest_maximum_difference = maximum_comparison.loc[
    largest_maximum_feature,
    "Absolute Maximum Difference"
]

print("Largest Maximum Difference Feature:", largest_maximum_feature)
print(
    "Largest Absolute Maximum Difference:",
    round(largest_maximum_difference, 6)
)

maximum_comparison.to_csv(
    "step24_maximum_comparison.csv"
)

print()
print("Result File:")
print("step24_maximum_comparison.csv")
print()


# ------------------------------------------------------------
# STEP 25 - FRAUD VS NORMAL RANGE
# ------------------------------------------------------------

print("=" * 60)
print("STEP 25 - FRAUD VS NORMAL RANGE")
print("=" * 60)

normal_range = normal_maximum - normal_minimum
fraud_range = fraud_maximum - fraud_minimum

range_comparison = pd.DataFrame({
    "Normal Range": normal_range,
    "Fraud Range": fraud_range
})

range_comparison["Range Difference"] = (
    range_comparison["Fraud Range"]
    - range_comparison["Normal Range"]
)

range_comparison["Absolute Range Difference"] = (
    range_comparison["Range Difference"].abs()
)

print(range_comparison)
print()

largest_range_feature = range_comparison[
    "Absolute Range Difference"
].idxmax()

largest_range_difference = range_comparison.loc[
    largest_range_feature,
    "Absolute Range Difference"
]

print("Largest Range Difference Feature:", largest_range_feature)
print(
    "Largest Absolute Range Difference:",
    round(largest_range_difference, 6)
)

range_comparison.to_csv(
    "step25_range_comparison.csv"
)

print()
print("Result File:")
print("step25_range_comparison.csv")
print()


# ------------------------------------------------------------
# STEP 26 - FRAUD VS NORMAL STANDARD DEVIATION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 26 - FRAUD VS NORMAL STANDARD DEVIATION")
print("=" * 60)

normal_std = normal_data[features].std()
fraud_std = fraud_data[features].std()

std_comparison = pd.DataFrame({
    "Normal Standard Deviation": normal_std,
    "Fraud Standard Deviation": fraud_std
})

std_comparison["Standard Deviation Difference"] = (
    std_comparison["Fraud Standard Deviation"]
    - std_comparison["Normal Standard Deviation"]
)

std_comparison["Absolute Standard Deviation Difference"] = (
    std_comparison["Standard Deviation Difference"].abs()
)

print("Standard Deviation Comparison:")
print(std_comparison)
print()

largest_std_feature = std_comparison[
    "Absolute Standard Deviation Difference"
].idxmax()

largest_std_difference = std_comparison.loc[
    largest_std_feature,
    "Absolute Standard Deviation Difference"
]

print("Largest Standard Deviation Difference Feature:", largest_std_feature)
print(
    "Largest Absolute Standard Deviation Difference:",
    round(largest_std_difference, 6)
)

std_comparison.to_csv(
    "step26_standard_deviation_comparison.csv"
)

print()
print("STEP 26 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step26_standard_deviation_comparison.csv")
print()


# ------------------------------------------------------------
# STEP 27 - CORRELATION ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 27 - CORRELATION ANALYSIS")
print("=" * 60)

correlation_data = df[features + ["Class"]].corr()

class_correlation = correlation_data["Class"].drop("Class")

correlation_comparison = pd.DataFrame({
    "Feature": class_correlation.index,
    "Correlation with Class": class_correlation.values
})

correlation_comparison["Absolute Correlation"] = (
    correlation_comparison["Correlation with Class"].abs()
)

correlation_comparison = correlation_comparison.sort_values(
    by="Absolute Correlation",
    ascending=False
)

print("Feature Correlation with Fraud Class:")
print(correlation_comparison)
print()

top_correlation_feature = correlation_comparison.iloc[0]["Feature"]
top_correlation_value = correlation_comparison.iloc[0][
    "Correlation with Class"
]

print("Highest Correlation Feature:", top_correlation_feature)
print(
    "Correlation with Class:",
    round(top_correlation_value, 6)
)

correlation_comparison.to_csv(
    "step27_correlation_analysis.csv",
    index=False
)

print()
print("STEP 27 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step27_correlation_analysis.csv")
# ------------------------------------------------------------
# STEP 28 - COVARIANCE ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 28 - COVARIANCE ANALYSIS")
print("=" * 60)

covariance_data = df[features + ["Class"]].cov()

class_covariance = covariance_data["Class"].drop("Class")

covariance_comparison = pd.DataFrame({
    "Feature": class_covariance.index,
    "Covariance with Class": class_covariance.values
})

covariance_comparison["Absolute Covariance"] = (
    covariance_comparison["Covariance with Class"].abs()
)

covariance_comparison = covariance_comparison.sort_values(
    by="Absolute Covariance",
    ascending=False
)

print("Feature Covariance with Fraud Class:")
print(covariance_comparison)
print()

top_covariance_feature = covariance_comparison.iloc[0]["Feature"]
top_covariance_value = covariance_comparison.iloc[0][
    "Covariance with Class"
]

print("Highest Covariance Feature:", top_covariance_feature)
print(
    "Covariance with Class:",
    round(top_covariance_value, 6)
)

covariance_comparison.to_csv(
    "step28_covariance_analysis.csv",
    index=False
)

print()
print("STEP 28 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step28_covariance_analysis.csv")
print()


# ------------------------------------------------------------
# STEP 29 - COEFFICIENT OF VARIATION ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 29 - COEFFICIENT OF VARIATION ANALYSIS")
print("=" * 60)

normal_mean_cv = normal_data[features].mean()
fraud_mean_cv = fraud_data[features].mean()

normal_std_cv = normal_data[features].std()
fraud_std_cv = fraud_data[features].std()

normal_cv = normal_std_cv / normal_mean_cv.abs()
fraud_cv = fraud_std_cv / fraud_mean_cv.abs()

cv_comparison = pd.DataFrame({
    "Normal CV": normal_cv,
    "Fraud CV": fraud_cv
})

cv_comparison["CV Difference"] = (
    cv_comparison["Fraud CV"]
    - cv_comparison["Normal CV"]
)

cv_comparison["Absolute CV Difference"] = (
    cv_comparison["CV Difference"].abs()
)

cv_comparison = cv_comparison.replace(
    [float("inf"), -float("inf")],
    pd.NA
)

print("Coefficient of Variation Comparison:")
print(cv_comparison)
print()

largest_cv_feature = cv_comparison[
    "Absolute CV Difference"
].idxmax()

largest_cv_difference = cv_comparison.loc[
    largest_cv_feature,
    "Absolute CV Difference"
]

print("Largest CV Difference Feature:", largest_cv_feature)
print(
    "Largest Absolute CV Difference:",
    round(largest_cv_difference, 6)
)

cv_comparison.to_csv(
    "step29_coefficient_of_variation_analysis.csv"
)

print()
print("STEP 29 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step29_coefficient_of_variation_analysis.csv")


# ------------------------------------------------------------
# STEP 30 - Z-SCORE ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 30 - Z-SCORE ANALYSIS")
print("=" * 60)

feature_mean = df[features].mean()
feature_std = df[features].std()

normal_zscores = (
    normal_data[features] - feature_mean
) / feature_std

fraud_zscores = (
    fraud_data[features] - feature_mean
) / feature_std

normal_mean_abs_zscore = normal_zscores.abs().mean()
fraud_mean_abs_zscore = fraud_zscores.abs().mean()

zscore_comparison = pd.DataFrame({
    "Normal Mean Absolute Z-Score": normal_mean_abs_zscore,
    "Fraud Mean Absolute Z-Score": fraud_mean_abs_zscore
})

zscore_comparison["Z-Score Difference"] = (
    zscore_comparison["Fraud Mean Absolute Z-Score"]
    - zscore_comparison["Normal Mean Absolute Z-Score"]
)

zscore_comparison["Absolute Z-Score Difference"] = (
    zscore_comparison["Z-Score Difference"].abs()
)

zscore_comparison = zscore_comparison.sort_values(
    "Absolute Z-Score Difference",
    ascending=False
)

print("Z-Score Comparison:")
print(zscore_comparison)
print()

largest_zscore_feature = zscore_comparison[
    "Absolute Z-Score Difference"
].idxmax()

largest_zscore_difference = zscore_comparison.loc[
    largest_zscore_feature,
    "Absolute Z-Score Difference"
]

print("Largest Z-Score Difference Feature:", largest_zscore_feature)
print(
    "Largest Absolute Z-Score Difference:",
    round(largest_zscore_difference, 6)
)

zscore_comparison.to_csv(
    "step30_zscore_analysis.csv"
)

print()
print("STEP 30 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step30_zscore_analysis.csv")
print()


# ------------------------------------------------------------
# STEP 31 - STATISTICAL SIGNIFICANCE ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 31 - STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 60)

from scipy.stats import ttest_ind

significance_results = []

for feature in features:
    normal_values = normal_data[feature].dropna()
    fraud_values = fraud_data[feature].dropna()

    t_statistic, p_value = ttest_ind(
        normal_values,
        fraud_values,
        equal_var=False
    )

    significance_results.append({
        "Feature": feature,
        "T-Statistic": t_statistic,
        "P-Value": p_value,
        "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
    })

significance_comparison = pd.DataFrame(significance_results)

significance_comparison["Absolute T-Statistic"] = (
    significance_comparison["T-Statistic"].abs()
)

significance_comparison = significance_comparison.sort_values(
    "P-Value",
    ascending=True
)

print("Statistical Significance Comparison:")
print(significance_comparison)
print()

most_significant_feature = significance_comparison.iloc[0]["Feature"]
most_significant_pvalue = significance_comparison.iloc[0]["P-Value"]

print("Smallest P-Value Feature:", most_significant_feature)
print(
    "Smallest P-Value:",
    f"{most_significant_pvalue:.10f}"
)

significant_count = (
    significance_comparison["P-Value"] < 0.05
).sum()

print("Features with p < 0.05:", significant_count)

significance_comparison.to_csv(
    "step31_statistical_significance_analysis.csv",
    index=False
)

print()
print("STEP 31 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step31_statistical_significance_analysis.csv")

# ------------------------------------------------------------
# STEP 32 - FEATURE-WISE FRAUD SIGNAL RANKING
# ------------------------------------------------------------

print("=" * 60)
print("STEP 32 - FEATURE-WISE FRAUD SIGNAL RANKING")
print("=" * 60)

# Create a combined ranking from the statistical comparisons completed
# in the earlier steps. Each metric is converted into a rank so that
# features measured on different numerical scales can be compared fairly.

rank_data = pd.DataFrame(index=features)

# Mean difference
rank_data["Mean Difference"] = (
    fraud_data[features].mean() - normal_data[features].mean()
).abs()

# Median difference
rank_data["Median Difference"] = (
    fraud_data[features].median() - normal_data[features].median()
).abs()

# Variance difference
rank_data["Variance Difference"] = (
    fraud_data[features].var() - normal_data[features].var()
).abs()

# Standard deviation difference
rank_data["Standard Deviation Difference"] = (
    fraud_data[features].std() - normal_data[features].std()
).abs()

# Skewness difference
rank_data["Skewness Difference"] = (
    fraud_data[features].skew() - normal_data[features].skew()
).abs()

# Kurtosis difference
rank_data["Kurtosis Difference"] = (
    fraud_data[features].kurtosis() - normal_data[features].kurtosis()
).abs()

# Minimum difference
rank_data["Minimum Difference"] = (
    fraud_data[features].min() - normal_data[features].min()
).abs()

# Maximum difference
rank_data["Maximum Difference"] = (
    fraud_data[features].max() - normal_data[features].max()
).abs()

# Range difference
normal_range = normal_data[features].max() - normal_data[features].min()
fraud_range = fraud_data[features].max() - fraud_data[features].min()
rank_data["Range Difference"] = (fraud_range - normal_range).abs()

# IQR difference
normal_iqr = normal_data[features].quantile(0.75) - normal_data[features].quantile(0.25)
fraud_iqr = fraud_data[features].quantile(0.75) - fraud_data[features].quantile(0.25)
rank_data["IQR Difference"] = (fraud_iqr - normal_iqr).abs()

# Correlation with fraud class
rank_data["Absolute Correlation"] = (
    df[features + ["Class"]].corr()["Class"].drop("Class").abs()
)

# Z-score difference using the same mean-absolute-z-score approach as Step 30
normal_mean = normal_data[features].mean()
normal_std = normal_data[features].std()
fraud_mean = fraud_data[features].mean()
fraud_std = fraud_data[features].std()

normal_abs_zscore = ((normal_data[features] - normal_mean) / normal_std).abs().mean()
fraud_abs_zscore = ((fraud_data[features] - fraud_mean) / fraud_std).abs().mean()

rank_data["Z-Score Difference"] = (
    fraud_abs_zscore - normal_abs_zscore
).abs()

# Outlier percentage in fraud transactions using the IQR rule
fraud_outlier_percentage = {}
for feature in features:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((fraud_data[feature] < lower) | (fraud_data[feature] > upper)).sum()
    fraud_outlier_percentage[feature] = (outliers / len(fraud_data)) * 100

rank_data["Fraud Outlier Percentage"] = pd.Series(fraud_outlier_percentage)

# Rank each metric: rank 1 means the largest fraud-vs-normal signal.
for column in rank_data.columns:
    rank_data[column + " Rank"] = (
        rank_data[column].rank(method="min", ascending=False)
    )

rank_columns = [
    column + " Rank"
    for column in [
        "Mean Difference",
        "Median Difference",
        "Variance Difference",
        "Standard Deviation Difference",
        "Skewness Difference",
        "Kurtosis Difference",
        "Minimum Difference",
        "Maximum Difference",
        "Range Difference",
        "IQR Difference",
        "Absolute Correlation",
        "Z-Score Difference",
        "Fraud Outlier Percentage"
    ]
]

rank_data["Overall Rank Score"] = rank_data[rank_columns].sum(axis=1)

rank_data = rank_data.sort_values("Overall Rank Score", ascending=True)

print("Feature-wise Fraud Signal Ranking:")
print(rank_data[["Overall Rank Score"] + rank_columns])
print()

top_signal_feature = rank_data.index[0]
top_signal_score = rank_data.iloc[0]["Overall Rank Score"]

print("Top Ranked Fraud Signal Feature:", top_signal_feature)
print("Overall Rank Score:", round(top_signal_score, 6))

rank_data.to_csv(
    "step32_feature_wise_fraud_signal_ranking.csv"
)

print()
print("STEP 32 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step32_feature_wise_fraud_signal_ranking.csv")



# ------------------------------------------------------------
# STEP 33 - TOP FRAUD SIGNAL IDENTIFICATION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 33 - TOP FRAUD SIGNAL IDENTIFICATION")
print("=" * 60)

# Select the top features from the overall fraud signal ranking.
top_signal_results = rank_data.reset_index().rename(
    columns={"index": "Feature"}
)

top_signal_results["Rank"] = range(1, len(top_signal_results) + 1)

# Keep the ranking information and the most important supporting metrics.
top_signal_results = top_signal_results[[
    "Rank",
    "Feature",
    "Overall Rank Score",
    "Mean Difference",
    "Median Difference",
    "Variance Difference",
    "Absolute Correlation",
    "Z-Score Difference",
    "Fraud Outlier Percentage"
]]

print("Top Fraud Signal Features:")
print(top_signal_results.head(10))
print()

identified_top_feature = top_signal_results.iloc[0]["Feature"]
identified_top_score = top_signal_results.iloc[0]["Overall Rank Score"]

print("Identified Top Fraud Signal Feature:", identified_top_feature)
print("Overall Rank Score:", round(identified_top_score, 6))

top_signal_results.to_csv(
    "step33_top_fraud_signal_identification.csv",
    index=False
)

print()
print("STEP 33 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step33_top_fraud_signal_identification.csv")

# ------------------------------------------------------------
# STEP 34 - TRANSACTION AMOUNT ANALYSIS
# ------------------------------------------------------------

print("=" * 60)
print("STEP 34 - TRANSACTION AMOUNT ANALYSIS")
print("=" * 60)

normal_amount = normal_data["Amount"]
fraud_amount = fraud_data["Amount"]

amount_comparison = pd.DataFrame({
    "Normal Mean": [normal_amount.mean()],
    "Fraud Mean": [fraud_amount.mean()],
    "Normal Median": [normal_amount.median()],
    "Fraud Median": [fraud_amount.median()],
    "Normal Standard Deviation": [normal_amount.std()],
    "Fraud Standard Deviation": [fraud_amount.std()],
    "Normal Minimum": [normal_amount.min()],
    "Fraud Minimum": [fraud_amount.min()],
    "Normal Maximum": [normal_amount.max()],
    "Fraud Maximum": [fraud_amount.max()],
})

amount_comparison["Mean Difference"] = (
    amount_comparison["Fraud Mean"]
    - amount_comparison["Normal Mean"]
)

amount_comparison["Median Difference"] = (
    amount_comparison["Fraud Median"]
    - amount_comparison["Normal Median"]
)

amount_comparison["Standard Deviation Difference"] = (
    amount_comparison["Fraud Standard Deviation"]
    - amount_comparison["Normal Standard Deviation"]
)

amount_comparison["Minimum Difference"] = (
    amount_comparison["Fraud Minimum"]
    - amount_comparison["Normal Minimum"]
)

amount_comparison["Maximum Difference"] = (
    amount_comparison["Fraud Maximum"]
    - amount_comparison["Normal Maximum"]
)

print("Transaction Amount Comparison:")
print(amount_comparison.T)
print()

print("Normal Amount Mean:", round(normal_amount.mean(), 6))
print("Fraud Amount Mean:", round(fraud_amount.mean(), 6))
print("Normal Amount Median:", round(normal_amount.median(), 6))
print("Fraud Amount Median:", round(fraud_amount.median(), 6))
print("Normal Amount Minimum:", round(normal_amount.min(), 6))
print("Fraud Amount Minimum:", round(fraud_amount.min(), 6))
print("Normal Amount Maximum:", round(normal_amount.max(), 6))
print("Fraud Amount Maximum:", round(fraud_amount.max(), 6))

amount_comparison.to_csv(
    "step34_transaction_amount_analysis.csv",
    index=False
)

print()
print("STEP 34 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step34_transaction_amount_analysis.csv")

# ------------------------------------------------------------
# STEP 35 - FRAUD VS NORMAL AMOUNT COMPARISON
# ------------------------------------------------------------

print("=" * 60)
print("STEP 35 - FRAUD VS NORMAL AMOUNT COMPARISON")
print("=" * 60)

normal_amount = normal_data["Amount"]
fraud_amount = fraud_data["Amount"]

normal_amount_q1 = normal_amount.quantile(0.25)
normal_amount_q3 = normal_amount.quantile(0.75)
fraud_amount_q1 = fraud_amount.quantile(0.25)
fraud_amount_q3 = fraud_amount.quantile(0.75)

normal_amount_iqr = normal_amount_q3 - normal_amount_q1
fraud_amount_iqr = fraud_amount_q3 - fraud_amount_q1

amount_comparison = pd.DataFrame({
    "Normal Mean": [normal_amount.mean()],
    "Fraud Mean": [fraud_amount.mean()],
    "Normal Median": [normal_amount.median()],
    "Fraud Median": [fraud_amount.median()],
    "Normal Q1": [normal_amount_q1],
    "Fraud Q1": [fraud_amount_q1],
    "Normal Q3": [normal_amount_q3],
    "Fraud Q3": [fraud_amount_q3],
    "Normal IQR": [normal_amount_iqr],
    "Fraud IQR": [fraud_amount_iqr],
    "Normal Standard Deviation": [normal_amount.std()],
    "Fraud Standard Deviation": [fraud_amount.std()],
    "Normal Minimum": [normal_amount.min()],
    "Fraud Minimum": [fraud_amount.min()],
    "Normal Maximum": [normal_amount.max()],
    "Fraud Maximum": [fraud_amount.max()]
})

amount_comparison["Mean Difference"] = (
    amount_comparison["Fraud Mean"]
    - amount_comparison["Normal Mean"]
)

amount_comparison["Median Difference"] = (
    amount_comparison["Fraud Median"]
    - amount_comparison["Normal Median"]
)

amount_comparison["Q1 Difference"] = (
    amount_comparison["Fraud Q1"]
    - amount_comparison["Normal Q1"]
)

amount_comparison["Q3 Difference"] = (
    amount_comparison["Fraud Q3"]
    - amount_comparison["Normal Q3"]
)

amount_comparison["IQR Difference"] = (
    amount_comparison["Fraud IQR"]
    - amount_comparison["Normal IQR"]
)

amount_comparison["Standard Deviation Difference"] = (
    amount_comparison["Fraud Standard Deviation"]
    - amount_comparison["Normal Standard Deviation"]
)

amount_comparison["Minimum Difference"] = (
    amount_comparison["Fraud Minimum"]
    - amount_comparison["Normal Minimum"]
)

amount_comparison["Maximum Difference"] = (
    amount_comparison["Fraud Maximum"]
    - amount_comparison["Normal Maximum"]
)

print("Fraud vs Normal Amount Comparison:")
print(amount_comparison.T)
print()

print("Amount Comparison Summary:")
print("Mean Difference:", round(amount_comparison.loc[0, "Mean Difference"], 6))
print("Median Difference:", round(amount_comparison.loc[0, "Median Difference"], 6))
print("Q1 Difference:", round(amount_comparison.loc[0, "Q1 Difference"], 6))
print("Q3 Difference:", round(amount_comparison.loc[0, "Q3 Difference"], 6))
print("IQR Difference:", round(amount_comparison.loc[0, "IQR Difference"], 6))
print("Standard Deviation Difference:", round(amount_comparison.loc[0, "Standard Deviation Difference"], 6))
print("Maximum Difference:", round(amount_comparison.loc[0, "Maximum Difference"], 6))

amount_comparison.to_csv(
    "step35_fraud_vs_normal_amount_comparison.csv",
    index=False
)

print()
print("STEP 35 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step35_fraud_vs_normal_amount_comparison.csv")


# ------------------------------------------------------------
# STEP 36 - FINAL STATISTICAL SUMMARY
# ------------------------------------------------------------

print("=" * 60)
print("STEP 36 - FINAL STATISTICAL SUMMARY")
print("=" * 60)

summary_results = [
    {
        "Analysis": "Highest Correlation",
        "Feature": top_correlation_feature,
        "Value": top_correlation_value
    },
    {
        "Analysis": "Highest Covariance",
        "Feature": top_covariance_feature,
        "Value": top_covariance_value
    },
    {
        "Analysis": "Largest Variance Difference",
        "Feature": largest_variance_feature,
        "Value": largest_variance_difference
    },
    {
        "Analysis": "Largest Skewness Difference",
        "Feature": largest_skewness_feature,
        "Value": largest_skewness_difference
    },
    {
        "Analysis": "Largest Kurtosis Difference",
        "Feature": largest_kurtosis_feature,
        "Value": largest_kurtosis_difference
    },
    {
        "Analysis": "Largest Median Difference",
        "Feature": largest_median_feature,
        "Value": largest_median_difference
    },
    {
        "Analysis": "Largest IQR Difference",
        "Feature": largest_IQR_feature,
        "Value": largest_IQR_difference
    },
    {
        "Analysis": "Largest Mean Difference",
        "Feature": largest_mean_feature,
        "Value": largest_mean_difference
    },
    {
        "Analysis": "Largest Minimum Difference",
        "Feature": largest_minimum_feature,
        "Value": largest_minimum_difference
    },
    {
        "Analysis": "Largest Maximum Difference",
        "Feature": largest_maximum_feature,
        "Value": largest_maximum_difference
    },
    {
        "Analysis": "Largest Range Difference",
        "Feature": largest_range_feature,
        "Value": largest_range_difference
    },
    {
        "Analysis": "Largest Standard Deviation Difference",
        "Feature": largest_std_feature,
        "Value": largest_std_difference
    },
    {
        "Analysis": "Largest Z-Score Difference",
        "Feature": largest_zscore_feature,
        "Value": largest_zscore_difference
    },
    {
        "Analysis": "Top Feature-wise Fraud Signal",
        "Feature": identified_top_feature,
        "Value": identified_top_score
    },
    {
        "Analysis": "Top Transaction Outlier Feature",
        "Feature": top_outlier_feature,
        "Value": top_outlier_percentage
    },
    {
        "Analysis": "Smallest Statistical Significance P-Value",
        "Feature": most_significant_feature,
        "Value": most_significant_pvalue
    },
    {
        "Analysis": "Normal Amount Mean",
        "Feature": "Amount",
        "Value": normal_amount.mean()
    },
    {
        "Analysis": "Fraud Amount Mean",
        "Feature": "Amount",
        "Value": fraud_amount.mean()
    },
    {
        "Analysis": "Amount Mean Difference",
        "Feature": "Amount",
        "Value": amount_comparison.loc[0, "Mean Difference"]
    }
]

final_statistical_summary = pd.DataFrame(summary_results)

print("Final Statistical Summary:")
print(final_statistical_summary)
print()

final_statistical_summary.to_csv(
    "step36_final_statistical_summary.csv",
    index=False
)

print("STEP 36 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step36_final_statistical_summary.csv")


# ------------------------------------------------------------
# STEP 37 - FINAL FRAUD-SIGNAL REPORT
# ------------------------------------------------------------

print("=" * 60)
print("STEP 37 - FINAL FRAUD-SIGNAL REPORT")
print("=" * 60)

# Create a final feature-level report by combining the main
# statistical signals calculated throughout the project.

final_fraud_signal_report = pd.DataFrame(index=features)

final_fraud_signal_report["Overall Rank Score"] = rank_data.loc[
    features, "Overall Rank Score"
]

final_fraud_signal_report["Correlation with Class"] = (
    df[features + ["Class"]].corr()["Class"].drop("Class")
)

final_fraud_signal_report["Absolute Correlation"] = (
    final_fraud_signal_report["Correlation with Class"].abs()
)

final_fraud_signal_report["Mean Difference"] = (
    fraud_data[features].mean() - normal_data[features].mean()
).abs()

final_fraud_signal_report["Median Difference"] = (
    fraud_data[features].median() - normal_data[features].median()
).abs()

final_fraud_signal_report["Variance Difference"] = (
    fraud_data[features].var() - normal_data[features].var()
).abs()

final_fraud_signal_report["Kurtosis Difference"] = (
    fraud_data[features].kurtosis() - normal_data[features].kurtosis()
).abs()

final_fraud_signal_report["Z-Score Difference"] = (
    fraud_abs_zscore - normal_abs_zscore
).abs()

final_fraud_signal_report["Fraud Outlier Percentage"] = (
    rank_data.loc[features, "Fraud Outlier Percentage"]
)

final_fraud_signal_report["T-Statistic"] = (
    significance_comparison.set_index("Feature")
    .loc[features, "T-Statistic"]
)

final_fraud_signal_report["P-Value"] = (
    significance_comparison.set_index("Feature")
    .loc[features, "P-Value"]
)

final_fraud_signal_report["Significant (p < 0.05)"] = (
    significance_comparison.set_index("Feature")
    .loc[features, "Significant (p < 0.05)"]
)

final_fraud_signal_report = final_fraud_signal_report.sort_values(
    "Overall Rank Score",
    ascending=True
)

final_fraud_signal_report.insert(
    0,
    "Final Rank",
    range(1, len(final_fraud_signal_report) + 1)
)

print("Final Fraud-Signal Report:")
print(final_fraud_signal_report.head(10))
print()

print("Total Features Analyzed:", len(final_fraud_signal_report))
print(
    "Features Significant at p < 0.05:",
    int((final_fraud_signal_report["P-Value"] < 0.05).sum())
)
print(
    "Top Feature from Combined Report:",
    final_fraud_signal_report.index[0]
)
print(
    "Top Feature Overall Rank Score:",
    round(final_fraud_signal_report.iloc[0]["Overall Rank Score"], 6)
)

final_fraud_signal_report.to_csv(
    "step37_final_fraud_signal_report.csv"
)

print()
print("STEP 37 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step37_final_fraud_signal_report.csv")


# ------------------------------------------------------------
# STEP 38 - FINAL PROJECT CONCLUSION
# ------------------------------------------------------------

print("=" * 60)
print("STEP 38 - FINAL PROJECT CONCLUSION")
print("=" * 60)

# Collect the main results from the completed statistical analyses.
# This conclusion summarizes statistical signals only and does not
# claim that any single feature independently causes fraud.

significant_feature_count = int(
    (final_fraud_signal_report["P-Value"] < 0.05).sum()
)

top_signal_feature = final_fraud_signal_report.index[0]
top_signal_score = float(
    final_fraud_signal_report.iloc[0]["Overall Rank Score"]
)

highest_correlation_feature = (
    final_fraud_signal_report["Absolute Correlation"].idxmax()
)
highest_correlation_value = float(
    final_fraud_signal_report.loc[
        highest_correlation_feature, "Correlation with Class"
    ]
)

highest_mean_difference_feature = (
    final_fraud_signal_report["Mean Difference"].idxmax()
)

highest_median_difference_feature = (
    final_fraud_signal_report["Median Difference"].idxmax()
)

highest_variance_difference_feature = (
    final_fraud_signal_report["Variance Difference"].idxmax()
)

highest_kurtosis_difference_feature = (
    final_fraud_signal_report["Kurtosis Difference"].idxmax()
)

highest_zscore_difference_feature = (
    final_fraud_signal_report["Z-Score Difference"].idxmax()
)

highest_outlier_feature = (
    final_fraud_signal_report["Fraud Outlier Percentage"].idxmax()
)

highest_outlier_percentage = float(
    final_fraud_signal_report.loc[
        highest_outlier_feature, "Fraud Outlier Percentage"
    ]
)

normal_amount_mean = float(normal_data["Amount"].mean())
fraud_amount_mean = float(fraud_data["Amount"].mean())
amount_mean_difference = fraud_amount_mean - normal_amount_mean

conclusion_data = [
    [
        "Project Objective",
        "Statistical comparison of normal and fraudulent transactions to identify measurable fraud-signal differences."
    ],
    [
        "Dataset",
        "Credit Card Fraud Detection Dataset (creditcard.csv)"
    ],
    
    [
        "Features Analyzed",
        len(features)
    ],
    [
        "Statistically Significant Features (p < 0.05)",
        significant_feature_count
    ],
    [
        "Top Combined Fraud Signal Feature",
        top_signal_feature
    ],
    [
        "Top Combined Fraud Signal Score",
        round(top_signal_score, 6)
    ],
    [
        "Highest Absolute Correlation Feature",
        highest_correlation_feature
    ],
    [
        "Correlation with Class",
        round(highest_correlation_value, 6)
    ],
    [
        "Largest Mean Difference Feature",
        highest_mean_difference_feature
    ],
    [
        "Largest Median Difference Feature",
        highest_median_difference_feature
    ],
    [
        "Largest Variance Difference Feature",
        highest_variance_difference_feature
    ],
    [
        "Largest Kurtosis Difference Feature",
        highest_kurtosis_difference_feature
    ],
    [
        "Largest Z-Score Difference Feature",
        highest_zscore_difference_feature
    ],
    [
        "Top Fraud Outlier Feature",
        highest_outlier_feature
    ],
    [
        "Top Fraud Outlier Percentage",
        round(highest_outlier_percentage, 6)
    ],
    [
        "Normal Transaction Amount Mean",
        round(normal_amount_mean, 6)
    ],
    [
        "Fraud Transaction Amount Mean",
        round(fraud_amount_mean, 6)
    ],
    [
        "Transaction Amount Mean Difference (Fraud - Normal)",
        round(amount_mean_difference, 6)
    ],
    [
        "Final Conclusion",
        "The completed statistical analysis identified multiple measurable differences between normal and fraudulent transactions. V7 is the top feature in the combined fraud-signal ranking, while V17 shows the highest absolute correlation with the fraud class and the largest z-score difference. V28 shows the largest skewness and kurtosis differences, V14 shows the largest median difference, V5 shows the largest mean/range-related differences, and V8 has the highest fraud outlier percentage. The findings provide statistical signals for fraud analysis, but they should not be interpreted as proof that any single feature independently causes or determines fraud."
    ]
]

final_project_conclusion = pd.DataFrame(
    conclusion_data,
    columns=["Conclusion Item", "Value"]
)

print("Final Project Conclusion:")
print(final_project_conclusion.to_string(index=False))
print()

final_project_conclusion.to_csv(
    "step38_final_project_conclusion.csv",
    index=False
)

print("STEP 38 COMPLETED SUCCESSFULLY")
print()
print("Result File:")
print("step38_final_project_conclusion.csv")
