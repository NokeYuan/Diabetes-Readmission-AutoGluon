import numpy as np
import pandas as pd

# Dictionary to map medical specialties into broader categories
medical_specialty_mapping = {
    # Surgery Group
    'Surgery-General': 'Surgery',
    'Surgery-Cardiovascular/Thoracic': 'Surgery',
    'Surgery-Neuro': 'Surgery',
    'Surgery-Colon&Rectal': 'Surgery',
    'Surgery-Plastic': 'Surgery',
    'Surgery-Thoracic': 'Surgery',
    'Surgery-PlasticwithinHeadandNeck': 'Surgery',
    'Surgery-Pediatric': 'Surgery',
    'Surgery-Maxillofacial': 'Surgery',
    'Surgery-Vascular': 'Surgery',
    'Surgery-Cardiovascular': 'Surgery',
    'Surgeon': 'Surgery',
    'SurgicalSpecialty': 'Surgery',

    # Internal Medicine
    'InternalMedicine': 'Internal Medicine',
    'Family/GeneralPractice': 'Internal Medicine',
    'Hospitalist': 'Internal Medicine',
    'PhysicianNotFound': 'Internal Medicine',

    # Pediatrics
    'Pediatrics-Endocrinology': 'Pediatrics',
    'Pediatrics': 'Pediatrics',
    'Pediatrics-CriticalCare': 'Pediatrics',
    'Pediatrics-Pulmonology': 'Pediatrics',
    'Pediatrics-Hematology-Oncology': 'Pediatrics',
    'Pediatrics-InfectiousDiseases': 'Pediatrics',
    'Pediatrics-AllergyandImmunology': 'Pediatrics',
    'Pediatrics-Neurology': 'Pediatrics',
    'Pediatrics-EmergencyMedicine': 'Pediatrics',
    'Cardiology-Pediatric': 'Pediatrics',

    # Cardiology
    'Cardiology': 'Cardiology',
    'Cardiology-Pediatric': 'Cardiology',

    # Neurology
    'Neurology': 'Neurology',
    'Neurophysiology': 'Neurology',

    # Psychiatry & Psychology
    'Psychiatry': 'Psychiatry & Psychology',
    'Psychiatry-Child/Adolescent': 'Psychiatry & Psychology',
    'Psychiatry-Addictive': 'Psychiatry & Psychology',
    'Psychology': 'Psychiatry & Psychology',

    # Radiology
    'Radiology': 'Radiology',
    'Radiologist': 'Radiology',

    # Obstetrics & Gynecology
    'Obsterics&Gynecology-GynecologicOnco': 'Obstetrics & Gynecology',
    'ObstetricsandGynecology': 'Obstetrics & Gynecology',
    'Obstetrics': 'Obstetrics & Gynecology',
    'Gynecology': 'Obstetrics & Gynecology',

    # Oncology & Hematology
    'Hematology/Oncology': 'Oncology & Hematology',
    'Oncology': 'Oncology & Hematology',
    'Hematology': 'Oncology & Hematology',

    # Endocrinology
    'Endocrinology': 'Endocrinology',
    'Endocrinology-Metabolism': 'Endocrinology',

    # Orthopedics
    'Orthopedics': 'Orthopedics',
    'Orthopedics-Reconstructive': 'Orthopedics',
    'SportsMedicine': 'Orthopedics',

    # Emergency Medicine
    'Emergency/Trauma': 'Emergency Medicine',

    # Pulmonology
    'Pulmonology': 'Pulmonology',

    # Gastroenterology
    'Gastroenterology': 'Gastroenterology',

    # Infectious Diseases
    'InfectiousDiseases': 'Infectious Diseases',

    # Nephrology
    'Nephrology': 'Nephrology',

    # Urology
    'Urology': 'Urology',

    # Otolaryngology
    'Otolaryngology': 'Otolaryngology',

    # Anesthesiology
    'Anesthesiology': 'Anesthesiology',
    'Anesthesiology-Pediatric': 'Anesthesiology',

    # Rheumatology
    'Rheumatology': 'Rheumatology',

    # Dermatology
    'Dermatology': 'Dermatology',

    # Pathology
    'Pathology': 'Pathology',

    # Allergy & Immunology
    'AllergyandImmunology': 'Allergy & Immunology',

    # Dentistry
    'Dentistry': 'Dentistry',

    # Ophthalmology
    'Ophthalmology': 'Ophthalmology',

    # Speech Therapy
    'Speech': 'Speech Therapy',

    # Physical Medicine & Rehabilitation
    'PhysicalMedicineandRehabilitation': 'Physical Medicine & Rehabilitation',

    # Proctology
    'Proctology': 'Proctology',

    # Osteopathy
    'Osteopath': 'Osteopathy',

    # Perinatology
    'Perinatology': 'Perinatology',

    # Miscellaneous
    'DCPTEAM': 'Other',
    'Resident': 'Other',
    'OutreachServices': 'Community Health'
}


race_mapping = {
"Asian":"Other",
"Hispanic":"Other"
}

age_mapping = {
    "[70-80)": 75,
    "[60-70)": 65,
    "[50-60)": 55,
    "[80-90)": 85,
    "[40-50)": 45,
    "[30-40)": 35,
    "[90-100)": 95,
    "[20-30)": 25,
    "[10-20)": 15,
    "[0-10)": 5
}

admission_id_mapping = {
          1.0:"Emergency",
          2.0:"Emergency",
          3.0:"Elective",
          4.0:"New Born",
          5.0:'Not Available',
          6.0:'Not Available',
          7.0:"Trauma Center",
          8.0:'Not Available'
}

# Define mapping of admission_source_id into 5 categories
admission_source_mapping = {
    # **Referral**
    1: 'Referral',   # Physician Referral
    2: 'Referral',   # Clinic Referral
    3: 'Referral',   # HMO Referral

    # **Transfer**
    4: 'Transfer',   # Transfer from a hospital
    5: 'Transfer',   # Transfer from a Skilled Nursing Facility (SNF)
    6: 'Transfer',   # Transfer from another health care facility
    10: 'Transfer',  # Transfer from critical access hospital
    18: 'Transfer',  # Transfer From Another Home Health Agency
    22: 'Transfer',  # Transfer from hospital inpatient/same facility resulting in separate claim
    25: 'Transfer',  # Transfer from Ambulatory Surgery Center
    26: 'Transfer',  # Transfer from Hospice

    # **Unknown**
    9: 'Unknown',    # Not Available
    15: 'Unknown',   # Not Available
    17: 'Unknown',   # NULL
    20: 'Unknown',   # Not Mapped
    21: 'Unknown',   # Unknown/Invalid

    # **Other**
    7: 'Other',      # Emergency Room
    8: 'Other',      # Court/Law Enforcement
    19: 'Other',     # Readmission to Same Home Health Agency
    14: 'Other',     # Extramural Birth (Doesn't fit birth category as it refers to a different type of admission)

    # **Birth**
    11: 'Birth',     # Normal Delivery
    12: 'Birth',     # Premature Delivery
    13: 'Birth',     # Sick Baby
    23: 'Birth',     # Born inside this hospital
    24: 'Birth'      # Born outside this hospital
}

# Define mapping for discharge disposition into categories
discharge_disposition_mapping = {
    # **Home**
    1: 'Home',  # Discharged to home
    6: 'Home',  # Discharged to home with home health service
    8: 'Home',  # Discharged to home under care of Home IV provider

    # **Transfer**
    2: 'Transfer',  # Discharged/transferred to another short-term hospital
    3: 'Transfer',  # Discharged/transferred to SNF
    4: 'Transfer',  # Discharged/transferred to ICF
    5: 'Transfer',  # Discharged/transferred to another type of inpatient care institution
    10: 'Transfer',  # Neonate discharged to another hospital for neonatal aftercare
    15: 'Transfer',  # Transferred within this institution to Medicare approved swing bed
    16: 'Transfer',  # Transferred/referred another institution for outpatient services
    17: 'Transfer',  # Transferred/referred to this institution for outpatient services
    22: 'Transfer',  # Transferred to another rehab facility
    23: 'Transfer',  # Transferred to a long-term care hospital
    24: 'Transfer',  # Transferred to a Medicaid nursing facility
    27: 'Transfer',  # Transferred to a federal health care facility
    28: 'Transfer',  # Transferred to a psychiatric hospital
    29: 'Transfer',  # Transferred to a Critical Access Hospital (CAH)
    30: 'Transfer',  # Transferred to another type of health care institution not defined elsewhere

    # **Expired**
    11: 'Expired',  # Expired
    19: 'Expired',  # Expired at home (Medicaid only, hospice)
    20: 'Expired',  # Expired in a medical facility (Medicaid only, hospice)
    21: 'Expired',  # Expired, place unknown (Medicaid only, hospice)

    # **Hospice**
    13: 'Hospice',  # Hospice / home
    14: 'Hospice',  # Hospice / medical facility

    # **Outpatient**
    9: 'Outpatient',  # Admitted as inpatient (likely for readmission tracking)
    12: 'Outpatient',  # Still patient or expected to return for outpatient services

    # **Unknown/Other**
    7: 'Other',  # Left AMA (Against Medical Advice)
    18: 'Unknown',  # NULL
    25: 'Unknown',  # Not Mapped
    26: 'Unknown'  # Unknown/Invalid
}


def map_diagnosis(data, cols):
    """
    Maps ICD-9 diagnosis codes into broader disease categories.

    Parameters:
        data (DataFrame): The dataset containing diagnosis columns.
        cols (list): List of diagnosis columns (e.g., ['diag_1', 'diag_2', 'diag_3']).

    Returns:
        DataFrame: Updated dataset with diagnosis codes replaced by disease categories.
    """

    for col in cols:
        # Convert everything to string first for uniform processing
        data[col] = data[col].astype(str)

        # Assign '-1' (as int) to codes starting with 'V' or 'E'
        data.loc[data[col].str.startswith(('V', 'E')), col] = -1

        # Convert all numeric values to float, keeping decimals (e.g., 250.xx remains as float)
        data[col] = pd.to_numeric(data[col], errors='coerce')

    for col in cols:
        # Create a temporary column for mapping
        data["temp_diag"] = np.nan

        # Apply mapping conditions
        data.loc[((data[col] >= 390) & (data[col] <= 459)) | (data[col] == 785), "temp_diag"] = "Circulatory"
        data.loc[((data[col] >= 460) & (data[col] <= 519)) | (data[col] == 786), "temp_diag"] = "Respiratory"
        data.loc[((data[col] >= 520) & (data[col] <= 579)) | (data[col] == 787), "temp_diag"] = "Digestive"
        data.loc[((data[col] >= 250) & (data[col] < 251)), "temp_diag"] = "Diabetes"
        data.loc[((data[col] >= 800) & (data[col] <= 999)), "temp_diag"] = "Injury"
        data.loc[((data[col] >= 710) & (data[col] <= 739)), "temp_diag"] = "Musculoskeletal"
        data.loc[((data[col] >= 580) & (data[col] <= 629)) | (data[col] == 788), "temp_diag"] = "Genitourinary"
        data.loc[((data[col] >= 140) & (data[col] <= 239)), "temp_diag"] = "Neoplasms"

        # Replace -1 with "Other"
        data.loc[data[col] == -1, "temp_diag"] = "Other"

        # Fill any remaining NaNs with "Other"
        data["temp_diag"] = data["temp_diag"].fillna("Other")

        # Ensure the column is explicitly cast to string before assignment
        data[col] = data["temp_diag"].astype(str)

        # Drop temporary column
        data.drop("temp_diag", axis=1, inplace=True)

    return data


drug_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
             'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
             'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
             'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
             'metformin-rosiglitazone', 'metformin-pioglitazone','examide','citoglipton',
             'glimepiride-pioglitazone']


drug_mapping = {
    "No": 0,       # No medication
    "Down": 1,     # Dose decreased
    "Steady": 2,   # Dose maintained
    "Up": 3        # Dose increased
}


