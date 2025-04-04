import pandas as pd

from Config import (medical_specialty_mapping, race_mapping, age_mapping, admission_id_mapping,
                    admission_source_mapping, discharge_disposition_mapping, map_diagnosis, drug_mapping, drug_cols)


class DiabetesDataset:
    """
    A class to handle preprocessing of the UCI Diabetes 130-US Hospitals dataset.
    """

    def __init__(self, uci_dataset):
        """
        Initialize the class with the UCI dataset.

        Parameters:
            uci_dataset: The dataset object fetched using `fetch_ucirepo(id=296)`.
        """
        self.uci_dataset = uci_dataset

    def data_preprocessing(self, drop_first=True, remove_duplicates=True, remove_all_high_missing=False):
        """
        Preprocess the dataset by performing:
        - Feature engineering
        - Encoding categorical variables
        - Handling missing values
        - Binarization & transformation of categorical variables

        Returns:
            pandas.DataFrame: The preprocessed dataset ready for modeling.
        """
        # Load the original dataset (IDs already included)
        diabetes_data = self.uci_dataset.data.original

        if remove_all_high_missing == True:
            # Drop 'weight' and 'payer_code' due to high missing values
            diabetes_data.drop(columns=['weight', 'payer_code', 'medical_specialty', 'A1Cresult', 'max_glu_serum'],
                               inplace=True)
            print(
                "Removed columns with high missing values: 'weight', 'payer_code','medical_specialty','A1Cresult', 'max_glu_serum'")

        else:
            ### Step 1: Handle Missing Data ###
            # Drop 'weight' and 'payer_code' due to high missing values
            diabetes_data.drop(columns=['weight', 'payer_code', ], inplace=True)
            print("Removed columns with high missing values: 'weight', 'payer_code'")

            # Fill missing values in 'A1Cresult' and 'max_glu_serum' with 'None'
            diabetes_data[['A1Cresult', 'max_glu_serum']] = diabetes_data[['A1Cresult', 'max_glu_serum']].fillna('None')

            # Recode 'A1Cresult' and 'max_glu_serum' into numerical categories
            diabetes_data['A1Cresult'].replace({'>7': 1, '>8': 1, 'Norm': 0, 'None': -99}, inplace=True)
            diabetes_data['max_glu_serum'].replace({'>200': 1, '>300': 1, 'Norm': 0, 'None': -99}, inplace=True)

            ### Step 2: Process Medical Specialty ###
            # Map 'medical_specialty' and keep only the top 10 categories, grouping the rest as 'Other'
            diabetes_data['medical_specialty_grouped'] = diabetes_data['medical_specialty'].map(
                medical_specialty_mapping)
            top_10 = diabetes_data['medical_specialty_grouped'].value_counts().nlargest(10).index
            diabetes_data['medical_specialty_grouped'] = diabetes_data['medical_specialty_grouped'].apply(
                lambda x: x if x in top_10 else 'Other')

            # Convert 'medical_specialty_grouped' to one-hot encoding
            diabetes_data = pd.get_dummies(diabetes_data, columns=['medical_specialty_grouped'], drop_first=drop_first,
                                           dtype=int)

            # Drop the original 'medical_specialty' column
            diabetes_data.drop(columns=['medical_specialty'], inplace=True)

        ### Step 3: Process Race ###
        # Map 'race' and binarize
        diabetes_data['race_grouped'] = diabetes_data['race'].replace(race_mapping).fillna('Other')
        diabetes_data = pd.get_dummies(diabetes_data, columns=['race_grouped'], drop_first=drop_first, dtype=int)
        diabetes_data.drop(columns=['race'], inplace=True)

        if remove_duplicates == True:
            ## Step 4: Handle Multiple Encounters per Patient ###
            print(f"Total rows before filtering: {diabetes_data.shape[0]}")
            # Keep only the row with the max 'time_in_hospital' per patient_nbr
            diabetes_data = diabetes_data.loc[diabetes_data.groupby('patient_nbr')['time_in_hospital'].idxmax()]
            print(f"Total rows after filtering: {diabetes_data.shape[0]}")

        ### Step 5: Encode Categorical Variables ###
        # Remove rows with 'Unknown/Invalid' gender
        diabetes_data = diabetes_data[diabetes_data['gender'] != 'Unknown/Invalid']

        # Map 'age' column using predefined mapping
        diabetes_data['age'] = diabetes_data['age'].replace(age_mapping)
        diabetes_data = pd.get_dummies(diabetes_data, columns=['age'], drop_first=drop_first, dtype=int)

        # Map & binarize 'admission_type_id'
        diabetes_data['admission_type_id'] = diabetes_data['admission_type_id'].replace(admission_id_mapping)
        diabetes_data = pd.get_dummies(diabetes_data, columns=['admission_type_id'], drop_first=drop_first, dtype=int)

        # Map & binarize 'admission_source_id'
        diabetes_data['admission_source_id'] = diabetes_data['admission_source_id'].replace(admission_source_mapping)
        diabetes_data = pd.get_dummies(diabetes_data, columns=['admission_source_id'], drop_first=drop_first, dtype=int)

        # Map & binarize 'discharge_disposition_id'
        diabetes_data['discharge_disposition_id'] = diabetes_data['discharge_disposition_id'].replace(
            discharge_disposition_mapping)
        diabetes_data = pd.get_dummies(diabetes_data, columns=['discharge_disposition_id'], drop_first=drop_first,
                                       dtype=int)

        ### Step 6: Process Diagnosis Columns ###
        # Apply diagnosis mapping function
        diabetes_data = map_diagnosis(diabetes_data, ['diag_1', 'diag_2', 'diag_3'])

        # Convert diagnosis columns to one-hot encoding
        diabetes_data = pd.get_dummies(diabetes_data, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=drop_first,
                                       dtype=int)

        ### Step 7: Process Drug Columns ###
        # Map drug columns to numeric values
        diabetes_data[drug_cols] = diabetes_data[drug_cols].replace(drug_mapping)

        ### Step 8: Final Binarization of Other Categorical Columns ###
        diabetes_data['change'] = diabetes_data['change'].replace({'Ch': 1, 'No': 0})
        diabetes_data['gender'] = diabetes_data['gender'].replace({'Male': 0, 'Female': 1})
        diabetes_data['diabetesMed'] = diabetes_data['diabetesMed'].replace({'No': 0, 'Yes': 1})

        ### Step 9: Encode Target Variable (readmitted) ###
        # Convert 'readmitted' column into binary (No=0, Yes=1)
        diabetes_data['readmitted'] = diabetes_data['readmitted'].replace({'NO': 0, '>30': 0, '<30': 1})

        ### Step 10: Drop constant columns
        constant_cols = [col for col in diabetes_data.columns if diabetes_data[col].nunique() == 1]
        print(f"Removing constant columns: {constant_cols}")
        diabetes_data.drop(columns=constant_cols, inplace=True)

        ### Step 11: Ensure Consistency in Data Types ###
        for col in diabetes_data.columns:
            if diabetes_data[col].dtype == "object":
                # Convert any remaining object columns to categorical integer encoding
                diabetes_data[col] = diabetes_data[col].astype("category").cat.codes
            else:
                # Force all numerical columns to int64 (prevent automatic downcasting)
                diabetes_data[col] = diabetes_data[col].astype("int64", errors="ignore")

        # Reset index after all transformations
        diabetes_data.reset_index(drop=True, inplace=True)

        return diabetes_data
