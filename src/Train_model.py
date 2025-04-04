# Standard Library Imports
import warnings

# Data Manipulation & Processing
import pandas as pd
from autogluon.common import space
# AutoML
from autogluon.tabular import TabularPredictor
# Feature Associations
from dython.nominal import associations
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
# Clustering & Evaluation
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import silhouette_score
# Machine Learning & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Suppress warnings
warnings.filterwarnings('ignore')


class TrainAutoGluon:
    """
    A class to preprocess data and train an AutoGluon model following best ML practices.
    """

    def __init__(self, processed_data, test_size=0.2, random_state=42, corr_threshold=0.9,
                 var_threshold=0.1, max_clusters=10):
        """
        Initialize with preprocessed data, perform data splitting, and store necessary attributes.

        Parameters:
            max_clusters (int): Maximum number of clusters to consider when selecting the best k.
        """
        # Store parameters
        self.test_size = test_size
        self.random_state = random_state
        self.corr_threshold = corr_threshold
        self.var_threshold = var_threshold
        self.max_clusters = max_clusters  # Maximum clusters to evaluate
        self.num_features = None

        # Drop ID columns
        self.data = processed_data.drop(columns=['encounter_id', 'patient_nbr'])
        self.label = 'readmitted'
        self.predictor = None

        # Identify feature types
        self.numeric_features = [
            col for col in self.data.columns if col != self.label and self.data[col].nunique() > 2
        ]
        self.binary_features = [
            col for col in self.data.columns if col != self.label and self.data[col].nunique() == 2
        ]

        # Split data into training and testing sets
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=self.test_size, random_state=self.random_state, stratify=self.data[self.label]
        )

        # Declare other attributes
        self.scaler = StandardScaler()
        self.selected_features = []
        self.best_k = None  # Stores the best number of clusters
        print("Data split complete.")

    def find_best_k(self):
        """
        Determine the best number of clusters using the Silhouette Score.
        Uses selected features if available, otherwise defaults to numeric features.
        """
        print("Finding best number of clusters using Silhouette Score...")

        # Use selected features if available, otherwise use numeric features
        feature_set = self.selected_features if self.selected_features else self.numeric_features

        if not feature_set:
            raise ValueError("No valid features available for clustering.")

        best_silhouette_score = -1
        best_k = 2  # Minimum number of clusters

        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(self.train_data[feature_set])

            score = silhouette_score(self.train_data[feature_set], cluster_labels)
            print(f"Clusters: {k}, Silhouette Score: {score:.4f}")

            if score > best_silhouette_score:
                best_silhouette_score = score
                best_k = k

        self.best_k = best_k
        print(f"Optimal number of clusters determined: {self.best_k}")

    def apply_clustering(self, add_as_feature=True):
        """
        Apply KMeans clustering to the numeric features and either:
        (1) add the cluster assignments as one-hot encoded features, or
        (2) use only cluster features (discard original features).

        Parameters:
            add_as_feature (bool):
                If True, append cluster features to the existing data.
                If False, replace current features with cluster features only.
        """
        if self.best_k is None:
            self.find_best_k()

        print(f"Applying KMeans clustering with {self.best_k} clusters...")

        # Apply clustering
        kmeans = KMeans(n_clusters=self.best_k, random_state=self.random_state, n_init=10)
        self.train_data['cluster'] = kmeans.fit_predict(self.train_data[self.numeric_features])
        self.test_data['cluster'] = kmeans.predict(self.test_data[self.numeric_features])

        # One-hot encode cluster column
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        train_clusters = encoder.fit_transform(self.train_data[['cluster']])
        test_clusters = encoder.transform(self.test_data[['cluster']])

        cluster_feature_names = [f'cluster_{i}' for i in range(1, self.best_k)]
        train_cluster_df = pd.DataFrame(train_clusters, columns=cluster_feature_names, index=self.train_data.index)
        test_cluster_df = pd.DataFrame(test_clusters, columns=cluster_feature_names, index=self.test_data.index)

        if add_as_feature:
            # Add cluster features to original data
            self.train_data = pd.concat([self.train_data.drop(columns=['cluster']), train_cluster_df], axis=1)
            self.test_data = pd.concat([self.test_data.drop(columns=['cluster']), test_cluster_df], axis=1)
            print(f"Added {self.best_k - 1} cluster features to existing data.")
        else:
            # Replace all features with just the cluster features and labels
            label_train = self.train_data[self.label].reset_index(drop=True)
            label_test = self.test_data[self.label].reset_index(drop=True)

            self.train_data = pd.concat([train_cluster_df.reset_index(drop=True), label_train], axis=1)
            self.test_data = pd.concat([test_cluster_df.reset_index(drop=True), label_test], axis=1)

            print(f"Replaced all features with {self.best_k - 1} cluster features.")

    def remove_low_variances(self):
        """
        Remove features with variance below the set threshold (default: 0, meaning only constant features are removed).
        """
        low_variance_features = self.train_data.var()[self.train_data.var() <= self.var_threshold].index.tolist()

        # Do not include the label in the low variance check
        low_variance_features = [col for col in low_variance_features if col != self.label]

        # Drop low-variance features from both train and test sets
        self.train_data.drop(columns=low_variance_features, inplace=True)
        self.test_data.drop(columns=low_variance_features, inplace=True)

        # Update numeric features list
        self.numeric_features = [col for col in self.numeric_features if col not in low_variance_features]

        print(f"Removed {len(low_variance_features)} low-variance features.")


    def normalize_features(self):
        """
        Normalize numerical features (excluding binary variables) in the training set.
        """
        self.train_data[self.numeric_features] = self.scaler.fit_transform(self.train_data[self.numeric_features])
        self.test_data[self.numeric_features] = self.scaler.transform(self.test_data[self.numeric_features])
        print("Normalization complete.")

    def remove_multicollinearity(self, nominal_assoc_method='cramer', numerical_assoc_method='pearson'):
        """
        Remove highly correlated features while keeping the most relevant one based on its correlation with the target variable.

        Parameters:
            nominal_assoc_method (str): The method to measure correlation between categorical variables.
            numerical_assoc_method (str): The method to measure correlation between numerical variables.
        """
        # Compute correlation matrix
        correlation_matrix = associations(
            self.train_data,
            nominal_columns=self.binary_features,
            nom_nom_assoc=nominal_assoc_method,
            num_num_assoc=numerical_assoc_method,
            compute_only=True,
            nan_strategy='replace',
            nan_replace_value=0.0
        )['corr']

        correlated_features = []

        # Identify highly correlated feature pairs
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] > self.corr_threshold:
                    correlated_features.append((correlation_matrix.columns[i], correlation_matrix.index[j]))

        # Keep the feature that is most correlated with the target variable
        lowest_corr_feature = []
        for feature_1, feature_2 in correlated_features:
            compare_features = self.train_data[[self.label, feature_1, feature_2]].copy()

            if feature_1 in self.binary_features or feature_2 in self.binary_features:
                nominal_columns = [col for col in [feature_1, feature_2] if col in self.binary_features]
                nominal_columns.append(self.label)

                compare_correlation_matrix = associations(
                    compare_features,
                    nominal_columns=nominal_columns,
                    nom_nom_assoc='theil'
                )['corr']

                lowest_corr_feature.append(compare_correlation_matrix.iloc[0].idxmin())
            else:
                r_value = [
                    abs(stats.pointbiserialr(compare_features[self.label], compare_features[feature_1])[0]),
                    abs(stats.pointbiserialr(compare_features[self.label], compare_features[feature_2])[0])
                ]
                lowest_corr_feature.append(feature_1 if r_value[0] < r_value[1] else feature_2)

        # Drop the lower-ranked features
        self.train_data.drop(columns=lowest_corr_feature, inplace=True)
        self.test_data.drop(columns=lowest_corr_feature, inplace=True)

        # Update numeric features list
        self.numeric_features = [col for col in self.numeric_features if col not in lowest_corr_feature]

        print(f"Removed {len(lowest_corr_feature)} highly correlated features.")

    def select_features(self):
        """
        Perform feature selection using ANOVA F-test (SelectKBest) on the training set.
        """
        X_train = self.train_data.drop(columns=[self.label])
        y_train = self.train_data[self.label]

        selector = SelectKBest(score_func=f_classif, k=min(self.num_features, len(self.numeric_features)))
        selector.fit(X_train[self.numeric_features], y_train)

        self.selected_features = selector.get_feature_names_out(self.numeric_features).tolist()

        self.train_data = self.train_data[self.selected_features + [self.label]]
        self.test_data = self.test_data[self.selected_features + [self.label]]

        print(f"Selected top {len(self.selected_features)} features using SelectKBest.")

    def apply_smote(self):
        """
        Apply SMOTE to balance the dataset (only for binary classification).
        """
        X_train = self.train_data.drop(columns=[self.label])
        y_train = self.train_data[self.label]

        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        self.train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        self.train_data[self.label] = y_resampled
        print("Applied SMOTE to balance classes.")


    def apply_random_oversampling(self):
        """
        Apply random oversampling using imblearn to balance classes.
        This method duplicates real minority class samples.
        """
        X_train = self.train_data.drop(columns=[self.label])
        y_train = self.train_data[self.label]

        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        self.train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        self.train_data[self.label] = y_resampled
        print("Applied RandomOverSampler to balance classes.")

    def apply_adasyn(self):
        """
        Apply ADASYN (Adaptive Synthetic Sampling) using imblearn.
        This focuses on harder-to-learn samples in the minority class.
        """
        X_train = self.train_data.drop(columns=[self.label])
        y_train = self.train_data[self.label]

        adasyn = ADASYN(random_state=self.random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

        self.train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        self.train_data[self.label] = y_resampled
        print("Applied ADASYN to balance classes.")



    def apply_random_undersampling(self):
        """
        Apply Random Undersampling to balance the dataset (only for binary classification).
        This reduces the majority class by randomly removing samples.
        """
        X_train = self.train_data.drop(columns=[self.label])
        y_train = self.train_data[self.label]

        rus = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

        self.train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        self.train_data[self.label] = y_resampled
        print("Applied Random Undersampling to balance classes.")


    def train_and_tune_model(self, eval_metric='roc_auc', sample_weight='auto_weight',
                             presets='medium_quality', verbosity=0, time_limit=2000):
        """
        Train an AutoGluon model using Bayesian Optimization for HPO.
        """

        print(f"Sample weight: {sample_weight}")
        print(f"Label distribution:\n{self.train_data[self.label].value_counts()}")

        hyperparameters = {
            'GBM': {
                'learning_rate': space.Real(0.01, 0.1, default=0.05),
                'num_leaves': space.Int(16, 64, default=31),
                'min_data_in_leaf': space.Int(5, 50, default=20),
                'feature_fraction': space.Real(0.5, 1.0, default=0.9),
            },
            'CAT': {
                'depth': space.Int(4, 10, default=6),
                'learning_rate': space.Real(0.01, 0.15, default=0.05),
                'l2_leaf_reg': space.Real(1, 10, default=3),
            },
            'XGB': {
                'scale_pos_weight': space.Int(1, 10, default=5),
                'max_depth': space.Int(3, 8, default=6),
                'learning_rate': space.Real(0.01, 0.2, default=0.1),
                'min_child_weight': space.Int(1, 10, default=1),
            },
            'RF': {
                'n_estimators': space.Int(100, 300, default=200),
                'max_depth': space.Int(10, 30, default=20),
            },
            'XT': {
                'n_estimators': space.Int(100, 300, default=200),
                'max_depth': space.Int(10, 30, default=20),
            },
            'LR': {
                'C': space.Real(0.01, 10.0, log=True, default=1.0),
                'solver': space.Categorical('liblinear', 'saga'),
            },
            'NN_TORCH': {
                'num_epochs': 3,
                'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
                'activation': space.Categorical('relu', 'softrelu', 'tanh'),
                'dropout_prob': space.Real(0.0, 0.5, default=0.1)
            },
            'TABPFNMIX': {
                'model_path_classifier': "autogluon/tabpfn-mix-1.0-classifier"
            }
        }

        hyperparameter_tune_kwargs = {
            'num_trials': 5,
            'scheduler': 'local',
            'searcher': 'bayes',  # Uses Bayesian Optimization
        }

        self.predictor = TabularPredictor(
            label=self.label,
            problem_type='binary',
            eval_metric=eval_metric,
            sample_weight=sample_weight
        ).fit(
            train_data=self.train_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            time_limit=time_limit,
            presets=presets,
            verbosity=verbosity,
            keep_only_best=False
        )

    def model_eval(self):
        """
        Evaluate the best model from each model type based on test score.
        Prints a sorted table showing model type, name, and test score.
        """
        # Get leaderboard from self.predictor
        leaderboard = self.predictor.leaderboard(self.test_data,silent=True)

        # Extract model type from model name
        leaderboard['model_type'] = leaderboard['model'].str.extract(r'^([^\\]+)')

        # Select best model from each model type based on score_test
        best_models = leaderboard.loc[leaderboard.groupby('model_type')['score_test'].idxmax()]

        # Sort by score_test in descending order
        best_models = best_models.sort_values(by='score_test', ascending=False).reset_index(drop=True)

        # Display final ranked table
        return best_models[['model_type', 'model', 'score_test']]

    def run_pipeline(self, cluster="No cluster", num_features=0,eval_metric='roc_auc',sample_weight ='auto_weight',presets='medium_quality',imbalanced = None,verbosity = 0,time_limit=3000):
        """
        Executes the full ML pipeline: preprocessing, optional SMOTE, clustering, feature selection, and model training.

        Parameters:
            use_smote (bool): Whether to apply SMOTE for class balancing.
            cluster (str): Clustering strategy. Options:
                           - 'No cluster': Do not apply clustering
                           - 'As features': Add cluster assignments as new features
                           - 'Only': Use only cluster features and drop original ones
            num_features (int): Number of top features to select using feature selection (0 means skip).
        """
        print("Running ML pipeline...")

        # Step 1: Preprocessing
        self.remove_low_variances()
        self.remove_multicollinearity()
        self.normalize_features()

        # Step 2: Feature Selection
        if num_features > 0:
            self.num_features = num_features
            self.select_features()
        else:
            print("Skipping feature selection step...")


        if imbalanced == 'smote':
            self.apply_smote()
        elif imbalanced == 'random':
            self.apply_random_oversampling()
        elif imbalanced == 'adasyn':
            self.apply_adasyn()
        elif imbalanced == 'undersample':
            self.apply_random_undersampling()
        else:
            print("Skipping imbalanced data step...")

        # Step 4: Optional Clustering
        if cluster == "As features":
            self.apply_clustering(add_as_feature=True)
        elif cluster == "Only":
            self.apply_clustering(add_as_feature=False)
        elif cluster == "No cluster":
            print("Skipping clustering step...")
        else:
            raise ValueError("Invalid clustering option. Choose from: 'No cluster', 'As features', 'Only'.")

        self.train_and_tune_model(eval_metric ,sample_weight =sample_weight,presets=presets ,verbosity=verbosity,time_limit=time_limit)

        print("ML Pipeline Completed.")



