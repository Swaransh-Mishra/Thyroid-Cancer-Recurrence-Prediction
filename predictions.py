# prediction.py

# Import necessary libraries
import joblib
import pandas as pd
import numpy as np # Used for creating the DataFrame correctly

def make_prediction(input_data):
    """
    Loads the trained model and scaler to make a prediction on new data.

    Args:
        input_data (dict): A dictionary containing the user's input for each feature.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and the probability of recurrence.
    """
    # Load the saved model and the scaler
    # Make sure these .pkl files are in the same directory as this script
    model = joblib.load('tuned_random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Convert the input dictionary to a pandas DataFrame
    df = pd.DataFrame(input_data, index=[0])

    # --- Preprocessing Steps (Must match your notebook) ---

    # 1. Label Encoding for categorical features
    # These mappings are derived directly from your notebook's LabelEncoder.
    categorical_mappings = {
        'Gender': {'F': 0, 'M': 1},
        'Smoking': {'No': 0, 'Yes': 1},
        'Hx Smoking': {'No': 0, 'Yes': 1},
        'Hx Radiothreapy': {'No': 0, 'Yes': 1}, # Note: This typo matches your trained model's column name
        'Thyroid Function': {'Euthyroid': 2, 'Clinical Hyperthyroidism': 0, 'Clinical Hypothyroidism': 1, 'Subclinical Hyperthyroidism': 3, 'Subclinical Hypothyroidism': 4},
        'Physical Examination': {'Single nodular goiter-left': 3, 'Multinodular goiter': 1, 'Single nodular goiter-right': 4, 'Normal': 2, 'Diffuse goiter': 0},
        'Adenopathy': {'No': 3, 'Right': 4, 'Extensive': 2, 'Bilateral': 0, 'Left': 3, 'Central': 1},
        'Pathology': {'Micropapillary': 2, 'Papillary': 3, 'Follicular': 1, 'Hurthel cell': 0},
        'Focality': {'Uni-Focal': 1, 'Multi-Focal': 0},
        'Risk': {'Low': 2, 'Intermediate': 1, 'High': 0},
        'T': {'T1a': 0, 'T1b': 1, 'T2': 2, 'T3a': 3, 'T3b': 4, 'T4a': 5, 'T4b': 6},
        'N': {'N0': 0, 'N1b': 2, 'N1a': 1},
        'M': {'M0': 0, 'M1': 1},
        'Stage': {'I': 0, 'II': 1, 'IVB': 4, 'III': 2, 'IVA': 3},
        'Response': {'Indeterminate': 2, 'Excellent': 1, 'Structural Incomplete': 3, 'Biochemical Incomplete': 0}
    }

    # Apply the mappings to the input DataFrame
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # 2. Scaling the 'Age' feature
    if 'Age' in df.columns:
        df[['Age']] = scaler.transform(df[['Age']])

    # Ensure the order of columns matches the model's training order
    model_features = model.feature_names_in_
    df = df[model_features]

    # --- Make Prediction ---
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability

