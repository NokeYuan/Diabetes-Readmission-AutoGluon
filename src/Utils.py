import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample


def analyze_missing_values(X, variables_df):
    """
    Analyzes missing values in the dataset by calculating the number and percentage of missing values.
    Merges the results with variable descriptions for better understanding.

    Parameters:
    X (pd.DataFrame): Feature dataset.
    variables_df (pd.DataFrame): DataFrame containing variable descriptions.

    Returns:
    pd.DataFrame: DataFrame containing missing values analysis sorted in descending order.
    """
    # Check for missing values
    missing_values = X.isnull().sum()

    # Convert missing values to percentage
    missing_percentage = (missing_values / len(X)) * 100

    # Create a DataFrame for missing values analysis
    missing_df = (
        pd.DataFrame({'Feature': X.columns, 'Missing Values': missing_values, 'Missing Percentage': missing_percentage})
        .merge(variables_df, left_on='Feature', right_on='name', how='left')
        .drop(columns='name')
        .sort_values(by='Missing Percentage', ascending=False)
        .reset_index(drop=True)
    )

    # Select relevant columns to display
    return missing_df[
        ['Feature', 'Missing Values', 'Missing Percentage', 'type', 'missing_values', 'description', 'role']]


def compute_metric_with_ci(y_true, y_pred, metric='auroc', n_bootstrap=100, confidence=0.95):
    """
    Compute AUROC, AUPRC, or event rate with 95% confidence interval using bootstrapping.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.
        metric (str): Metric to compute ('auroc', 'auprc', 'event_rate').
        n_bootstrap (int): Number of bootstrap samples for CI estimation.
        confidence (float): Confidence level (default 0.95).

    Returns:
        tuple: (Metric Value, lower_bound_CI, upper_bound_CI)
    """
    metric_scores = []
    for _ in range(n_bootstrap):
        indices = resample(range(len(y_true)), replace=True)
        if len(set(y_true.iloc[indices])) > 1:  # Ensure both classes exist in bootstrap sample
            if metric == 'auroc':
                metric_scores.append(roc_auc_score(y_true.iloc[indices], y_pred.iloc[indices]))
            elif metric == 'auprc':
                metric_scores.append(average_precision_score(y_true.iloc[indices], y_pred.iloc[indices]))
            elif metric == 'readmission_rate':
                metric_scores.append(y_true.iloc[indices].mean())

    metric_mean = np.mean(metric_scores)
    lower_bound = np.percentile(metric_scores, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(metric_scores, (1 + confidence) / 2 * 100)

    return metric_mean, lower_bound, upper_bound







