# === Data Manipulation ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# === Visualization ===
import seaborn as sns
# === Model Evaluation & SHAP ===
import shap
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

# === Custom Utilities ===
from Utils import compute_metric_with_ci


# shap.initjs()
def plot_feature_distributions(
        df,
        numerical_columns=[],
        numerical_plot_type="histogram",
        target_col=None
):
    """
    Plots feature distributions for specified numerical and categorical variables, with optional hue based on target column.

    Parameters:
        df (pd.DataFrame): The dataset.
        numerical_columns (list): List of numerical variables to include.
        numerical_plot_type (str): Type of plot for numerical features ("violin", "histogram", or "boxplot").
        categorical_columns (list): List of categorical variables to include.
        categorical_plot_type (str): Type of plot for categorical features ("count" or "bar").
        target_col (str, optional): Column name to use for hue-based differentiation.

    Returns:
        None (Displays the plots)
    """
    num_features = len(numerical_columns)


    # --- Plot Numerical Features ---
    if num_features > 0:
        num_cols = min(2, num_features)  # Max 2 plots per row
        num_rows = (num_features // num_cols) + (num_features % num_cols > 0)

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8,8))
        #fig.suptitle(f"Numerical Feature Distributions ({numerical_plot_type.capitalize()} Plots)", fontsize=16)
        axes = axes.flatten()

        for i, col in enumerate(numerical_columns):
            if numerical_plot_type == "violin":
                sns.violinplot(y=df[col], x=df[target_col] if target_col else None, ax=axes[i], inner="quartile")
            elif numerical_plot_type == "histogram":
                sns.histplot(df, x=col, hue=target_col if target_col else None, kde=True, ax=axes[i], bins=8,
                             palette="coolwarm", alpha=0.6)
                axes[i].set_yscale("log")  # Apply log scale to y-axis
            elif numerical_plot_type == "boxplot":
                sns.boxplot(y=df[col], x=df[target_col] if target_col else None, ax=axes[i], palette="coolwarm")
            else:
                print(
                    f"Invalid numerical plot type: {numerical_plot_type}. Choose 'violin', 'histogram', or 'boxplot'.")
                return

            axes[i].set_title(col)
            axes[i].set_xlabel("")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    print("Plotting complete!")


def generate_categorical_summary_table(df, categorical_prefixes, target_column="readmitted"):
    """
    Generates a summary table for categorical variables, showing:
    - Overall count & percentage
    - Count & percentage for `readmitted = 0`
    - Count & percentage for `readmitted = 1`

    Parameters:
    - df (pd.DataFrame): The dataset.
    - categorical_prefixes (list): List of prefixes for categorical variables.
    - target_column (str): The target column indicating readmission status.

    Returns:
    - pd.DataFrame: A formatted summary table.
    """

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    # Extract categorical columns based on prefixes
    categorical_columns = ["gender"]  # Always include gender
    for prefix in categorical_prefixes:
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        categorical_columns.extend(matching_cols)

    # Filter only binary columns (values should be 0 or 1)
    categorical_columns = [
        col for col in categorical_columns
        if col in df.columns and df[col].nunique() <= 2  # Ensure binary (0/1)
    ]

    if not categorical_columns:
        raise ValueError("No valid categorical variables found in dataset.")

    # Convert categorical columns to numeric, handling any non-numeric cases
    df[categorical_columns] = df[categorical_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Compute total counts for each category
    total_counts = df[categorical_columns].sum()
    total_percentage = (total_counts / len(df)) * 100

    # Compute counts and percentages for readmitted=0 and readmitted=1
    readmitted_0_counts = df[df[target_column] == 0][categorical_columns].sum()
    readmitted_1_counts = df[df[target_column] == 1][categorical_columns].sum()

    readmitted_0_percentage = (readmitted_0_counts / len(df[df[target_column] == 0])) * 100
    readmitted_1_percentage = (readmitted_1_counts / len(df[df[target_column] == 1])) * 100

    # Construct final table
    summary_table = pd.DataFrame({
        "Variable": total_counts.index,
        "Overall, n (%)": [f"{int(c)} ({p:.1f}%)" for c, p in zip(total_counts, total_percentage)],
        "Readmitted = 0, n (%)": [f"{int(c)} ({p:.1f}%)" for c, p in zip(readmitted_0_counts, readmitted_0_percentage)],
        "Readmitted = 1, n (%)": [f"{int(c)} ({p:.1f}%)" for c, p in zip(readmitted_1_counts, readmitted_1_percentage)],
    })

    # Sorting: First by prefix order, then by variable
    def get_sort_key(col_name):
        for prefix in sorted(categorical_prefixes, key=len, reverse=True):  # Longest match first
            if col_name.startswith(prefix):
                return categorical_prefixes.index(prefix)
        return -1  # Default for variables like "gender"

    summary_table["SortKey"] = summary_table["Variable"].apply(get_sort_key)
    summary_table = summary_table.sort_values(by=["SortKey", "Variable"]).drop(columns=["SortKey"])

    return summary_table


def plot_correlation_heatmap(data, figsize=(12, 10), vmax=0.3, square=True, linewidths=0.5, cbar_shrink=0.5):
    """
    Plots a correlation heatmap with an upper triangle mask.

    Parameters:
        data (pd.DataFrame): The input dataframe.
        figsize (tuple): Figure size (default: (12, 10)).
        vmax (float): Maximum correlation value for color scale (default: 0.3).
        square (bool): Whether to keep the heatmap squares equal (default: True).
        linewidths (float): Width of the grid lines (default: 0.5).
        cbar_shrink (float): Shrink factor for the colorbar (default: 0.5).

    Returns:
        None (Displays the heatmap)
    """
    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=vmax, center=0,
                square=square, linewidths=linewidths, cbar_kws={"shrink": cbar_shrink})

    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()


def subgroup_analysis_with_plot(trainer, group_sets, metric='auroc', n_bootstrap=100, confidence=0.95):
    """
    Perform subgroup analysis by calculating AUROC, AUPRC, or event rate with 95% CI using bootstrapping
    and visualize results using bar plots with manually centered error bars.

    Parameters:
        trainer (autogluon.tabular.Trainer): Trained AutoGluon model.
        group_sets (dict): Dictionary where keys are group names and values are lists of subgroup column names.
        metric (str): Metric to compute ('auroc', 'auprc', 'event_rate').
        n_bootstrap (int): Number of bootstrap samples for CI estimation.
        confidence (float): Confidence level (default 0.95).

    Returns:
        pd.DataFrame: Table of computed metric with 95% CI for each subgroup.
    """
    test_data = trainer.test_data.copy()
    results = []
    overall_y_true = test_data[trainer.label]
    overall_y_pred_proba = trainer.predictor.predict_proba(test_data, as_pandas=True, as_multiclass=True)

    if isinstance(overall_y_pred_proba, pd.DataFrame):
        positive_class = trainer.predictor.positive_class  # Get positive class label
        overall_y_pred = overall_y_pred_proba[positive_class]
    else:
        overall_y_pred = overall_y_pred_proba[:, 1]

    overall_metric, overall_ci_lower, overall_ci_upper = compute_metric_with_ci(
        overall_y_true, overall_y_pred, metric, n_bootstrap, confidence
    )

    # Assign colors to each group
    group_colors = sns.color_palette("tab10", len(group_sets))

    # Create a mapping of group names to colors
    group_color_map = {group: group_colors[i] for i, group in enumerate(group_sets.keys())}

    # Collect results
    for group_name, columns_to_investigate in group_sets.items():
        for col in columns_to_investigate:
            subgroup = test_data[test_data[col] == 1]  # Filter subgroup where column == 1

            if len(subgroup) < 20:  # Avoid very small subgroups
                continue

            y_true_sub = subgroup[trainer.label]
            y_pred_proba = trainer.predictor.predict_proba(subgroup, as_pandas=True, as_multiclass=True)

            if isinstance(y_pred_proba, pd.DataFrame):
                y_pred_sub = y_pred_proba[positive_class]
            else:
                y_pred_sub = y_pred_proba[:, 1]

            metric_mean, lower_bound, upper_bound = compute_metric_with_ci(
                y_true_sub, y_pred_sub, metric, n_bootstrap, confidence
            )

            results.append({
                "Group": group_name,
                "Subgroup": col,
                "Metric": round(metric_mean, 3),
                "95% CI Lower": round(lower_bound, 3),
                "95% CI Upper": round(upper_bound, 3),
                "Sample Size": len(subgroup)
            })

    results_df = pd.DataFrame(results)

    # Ensure bars within the same group stay close, but different groups have a gap
    group_spacing = 2.0  # Increase this for larger gaps between groups
    current_x = 0
    x_positions = []
    colors = []
    group_order = []
    last_group = None

    for row in results_df.itertuples():
        if row.Group != last_group:
            current_x += group_spacing  # Add extra space for new group
        x_positions.append(current_x)
        colors.append(group_color_map[row.Group])  # Assign group-specific color
        group_order.append(row.Group)
        current_x += 1  # Normal spacing for subgroups
        last_group = row.Group

    # Convert group_order to unique order for legend
    unique_groups = list(dict.fromkeys(group_order))

    # Plot the results
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    ax = plt.gca()

    # Create bars and set group-specific colors
    bars = ax.bar(x_positions, results_df["Metric"], width=0.6, color=colors)

    # Add error bars manually
    for i, row in enumerate(results_df.itertuples()):
        plt.errorbar(
            x=x_positions[i], y=row.Metric,
            yerr=[[row.Metric - row._4], [row._5 - row.Metric]],
            fmt='o', color='black', capsize=5
        )

    # Draw reference line for overall model performance with label including the point estimate
    overall_label = f"Overall:{overall_metric:.3f}"
    overall_line = plt.axhline(y=overall_metric, linestyle='dashed', color='red', label=overall_label)

    # Set custom x-axis labels
    plt.xticks(x_positions, results_df["Subgroup"], rotation=45, ha='right')
    plt.xlabel("Subgroups")
    plt.ylabel(metric.upper())
    plt.ylim(0, 1.0)  # Ensure Y-axis max is set to 1.0
    plt.title(f"Subgroup {metric.upper()} Analysis with 95% CI")

    # Add legend **inside** the plot at the top-right
    handles = [plt.Rectangle((0, 0), 1, 1, color=group_color_map[g]) for g in unique_groups]
    handles.append(overall_line)  # Add overall performance line
    labels = unique_groups + [overall_label]  # Include point estimate in legend
    ax.legend(handles, labels, loc="upper right", fontsize=10, frameon=True)

    plt.show()

    return results_df

def plot_feature_importance(trainer, top_n=20):
    """
    Plots feature importance as a bar plot.

    Parameters:
        trainer (autogluon.tabular.Trainer): Trained AutoGluon model.
        top_n (int): Number of top features to display (default: 20).

    Returns:
        None (Displays the bar plot)
    """
    # Compute feature importance
    feature_importance_df = trainer.predictor.feature_importance(trainer.test_data)

    # Reset index and sort by importance
    feature_importance_df = feature_importance_df.reset_index().rename(columns={'index': 'Feature'})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(top_n)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=feature_importance_df, y="Feature", x="importance",
        palette="viridis", edgecolor="black"
    )

    plt.xlabel("RF Feature Importance Score")
    plt.ylabel("Features")
    # plt.title("Top Feature Importance (AutoGluon)")
    plt.gca() # Invert axis for better visualization
    plt.show()


class AutogluonWrapper:
    """
    Wrapper to use AutoGluon with SHAP for feature importance explanations.
    """

    def __init__(self, predictor, feature_names, target_class=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_class = target_class

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        preds = self.ag_model.predict_proba(X)

        # Ensure we select the correct column for binary classification
        if self.target_class is not None:
            if self.target_class in preds.columns:  # If column names exist
                return preds[self.target_class]
            else:
                return preds.iloc[:, 1]  # Default to positive class

        return preds


def shap_summary_plot(trainer, target_class=1, nsamples=100):
    """
    Generate a SHAP summary plot for feature importance using AutoGluon.

    Parameters:
        trainer (TrainAutoGluon): Custom trainer class containing AutoGluon model and data.
        target_class (int): Target class label to explain (default 1 for binary classification).
        nsamples (int): Number of samples to approximate SHAP values.

    Returns:
        shap_values (np.ndarray or list): Computed SHAP values for the selected test sample.
        X_test_sample (pd.DataFrame): The test sample used for SHAP explanation.
    """

    # Ensure `readmitted` is treated as categorical (integer)
    trainer.test_data[trainer.label] = trainer.test_data[trainer.label].astype(int)

    # Extract test data without labels
    X_test = trainer.test_data.drop(columns=[trainer.label])
    y_test = trainer.test_data[trainer.label]

    # Take a sample of X_test for faster SHAP computation
    X_test_sample = X_test.sample(min(200, len(X_test)), random_state=0)

    # Determine baseline class (negative class)
    negative_class = 0 if 0 in y_test.unique() else y_test.value_counts().idxmin()
    baseline = X_test[y_test == negative_class].sample(min(50, len(X_test)), random_state=0)

    # Initialize SHAP Explainer
    ag_wrapper = AutogluonWrapper(trainer.predictor, X_test.columns, target_class)
    explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)

    print("Computing SHAP values for test data...")
    shap_values = explainer.shap_values(X_test_sample, nsamples=nsamples)

    # Generate SHAP Summary Plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test_sample)
    plt.show()

    # Return SHAP values and sampled test data
    return shap_values, X_test_sample

def plot_combined_roc_curves(all_predictors, label_col="readmitted"):
    """
    Plots ROC curves for multiple predictors on the same plot.

    Parameters:
    - all_predictors: Dictionary of label -> trained trainer object
    - label_col: Name of the label column in the dataset
    """
    plt.figure(figsize=(10, 8))

    for label, trainer in all_predictors.items():
        # Extract test data
        X_test = trainer.test_data.drop(columns=[label_col])
        y_test = trainer.test_data[label_col]

        # Get prediction probabilities
        y_proba = trainer.predictor.predict_proba(X_test).iloc[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Plot each ROC curve
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    # Plot reference line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.title("ROC Curves Across Different Feature and Preprocessing strategies")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_predictors_with_calibration(predictors_dict: dict, metric="f1"):
    """
    Calibrate decision thresholds for multiple AutoGluon predictors and compare their performance.

    Parameters:
        predictors_dict (dict): Dictionary mapping model names -> trainer object with `.predictor` and `.test_data`.
        metric (str): Metric to optimize during threshold calibration (default is 'f1').

    Returns:
        pd.DataFrame: Comparison table of performance metrics before and after calibration.
    """
    rows = []

    for name, trainer in predictors_dict.items():
        predictor: TabularPredictor = trainer.predictor
        test_data = trainer.test_data
        label_col = predictor.label
        X_test = test_data.drop(columns=[label_col])
        y_true = test_data[label_col]

        # BEFORE CALIBRATION
        y_pred_default = predictor.predict(X_test)
        y_proba_default = predictor.predict_proba(X_test)
        positive_index = list(predictor.class_labels).index(predictor.positive_class)
        y_proba_pos_default = y_proba_default.iloc[:, positive_index]

        metrics_before = {
            "Accuracy": accuracy_score(y_true, y_pred_default),
            "Precision": precision_score(y_true, y_pred_default),
            "Recall": recall_score(y_true, y_pred_default),
            "F1": f1_score(y_true, y_pred_default),
            "AUROC": roc_auc_score(y_true, y_proba_pos_default)
        }

        # CALIBRATION
        calibrated_threshold = predictor.calibrate_decision_threshold(
            data=test_data, metric=metric, verbose=False
        )
        predictor.set_decision_threshold(calibrated_threshold)

        # AFTER CALIBRATION
        y_pred_calibrated = predictor.predict(X_test)
        y_proba_calibrated = predictor.predict_proba(X_test)
        y_proba_pos_calibrated = y_proba_calibrated.iloc[:, positive_index]

        metrics_after = {
            "Accuracy": accuracy_score(y_true, y_pred_calibrated),
            "Precision": precision_score(y_true, y_pred_calibrated),
            "Recall": recall_score(y_true, y_pred_calibrated),
            "F1": f1_score(y_true, y_pred_calibrated),
            "AUROC": roc_auc_score(y_true, y_proba_pos_calibrated)
        }

        # Combine rows
        rows.append({
            "Model": name,
            "Calibrated": False,
            **metrics_before
        })
        rows.append({
            "Model": name,
            "Calibrated": True,
            **metrics_after
        })

    results_df = pd.DataFrame(rows)
    return results_df

