import pandas as pd
import numpy as np
import scipy.constants as const
import os
import re

# Paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_PATH, 'data', 'raw')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'data', 'processed')

# Files
input_file = 'skempi_v2.csv'
output_file = 'skempi_v2_processed.csv'

# File paths
input_file_path = os.path.join(INPUT_PATH, input_file)
output_file_path = os.path.join(OUTPUT_PATH, output_file)

def data_preprocessing_skempi(path):
    """
    Reads the Skempi dataset and performs data preprocessing.
    Parameters:
        path: str
            Path to the input file
        
        Returns:
        df: pandas.DataFrame
            Processed dataframe with the following new columns:
                - Temp_K: Temperature in Kelvin
                - ddG_kcal_mol: Free energy change in kcal/mol       
    """
    print("Initializing data preprocessing...")
    
    # Read the dataset
    df = pd.read_csv(path, sep=';')
    
    # Clean the column names
    df.columns = df.columns.str.strip()
    
    # Extract only the numbers from the Temperature column
    # We have to do this because the Temperature column is a string with the format "298(assumed)" (sometimes)
    df['Temp_K'] = df['Temperature'].astype(str).str.extract(r'(\d+)').astype(float) # Convert to float
    df['Temp_K'] = df['Temp_K'].fillna(298.0) # Fill the missing values with 298.0 K (standard temperature)

        
    def affinity_columns_match(raw_col, parsed_col):
        def has_letters(value):
            text = re.sub(r'[eE][+-]?\d+', '', str(value).strip())
            return bool(re.search(r'[a-zA-Z]', text))
        
        parsed = pd.to_numeric(df[parsed_col], errors='coerce')
        letters = df[raw_col].apply(has_letters)
        letters_ok = (~letters) | parsed.isna()
        
        raw_numeric = pd.to_numeric(
            df[raw_col].astype(str).str.replace(r'^[<>~]\s*', '', regex=True),
            errors='coerce',
        )
        raw_numeric = raw_numeric.where(~letters, np.nan)
        numbers_ok = letters | np.isclose(raw_numeric, parsed, rtol=0, atol=0, equal_nan=True)
        
        return (letters_ok & numbers_ok).all()
    
    if (
        affinity_columns_match('Affinity_mut (M)', 'Affinity_mut_parsed')
        and affinity_columns_match('Affinity_wt (M)', 'Affinity_wt_parsed')
    ):
        col_affinity_mut = "Affinity_mut_parsed"
        col_affinity_wt = "Affinity_wt_parsed"
        print(f"~~~ Affinity columns match their parsed counterparts. We will use {col_affinity_mut} and {col_affinity_wt} as columns.")
    else:
        col_affinity_mut = "Affinity_mut (M)"
        col_affinity_wt = "Affinity_wt (M)"
        print("~~~ Warning: Affinity columns do NOT match their parsed counterparts. We will use {col_affinity_mut} and {col_affinity_wt} as columns.")
    
    # Check if the columns Affinity_mut (M) and Affinity_wt (M) have missing values
    if df[col_affinity_mut].isna().any() or df[col_affinity_wt].isna().any():
        print(f"~~~ Warning: Missing values found in the {col_affinity_mut} and {col_affinity_wt} columns.")
        df = df[df[col_affinity_mut].notna() & df[col_affinity_wt].notna()]
        print("~~~~ Removed missing values.")
    else:
        print(f"~~~ No missing values detected in {col_affinity_mut} or {col_affinity_wt}.")

    # Ensure the affinity columns are numeric
    affinity_mut = pd.to_numeric(df[col_affinity_mut], errors='coerce')
    affinity_wt = pd.to_numeric(df[col_affinity_wt], errors='coerce')
    
    # Ideal gas constant
    R_j = const.R # In J/(mol*K)
    # Convert to kcal/(mol*K)
    R_kcal = R_j / 4184 # 1 kcal = 4184 J
    print("~~~ Ideal gas constant: ", R_kcal, "kcal/(mol*K)")

    # Apply the thermodynamic equation to calculate the free energy change (R*T*ln(affinity_mut / affinity_wt))
    df['ddG_kcal_mol'] = R_kcal * df['Temp_K'] * np.log(affinity_mut / affinity_wt)
    print("~~~ Finished calculating the free energy change.")
    
    # Check if the ddG column has missing values
    if df['ddG_kcal_mol'].isna().any():
        print("~~~ Warning: Missing values found in the ddG column.")
        print(f"~~~ Removing missing values. Original shape: {df.shape}")
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ddG_kcal_mol']).copy()
        print(f"~~~ New shape: {df_clean.shape}")
    else:
        print("~~~ No missing values detected in ddG column.")
        df_clean = df.copy()
    
    return df_clean

# Main execution
if __name__ == "__main__":
    
    df = data_preprocessing_skempi(input_file_path)
    
    # Show the first 5 rows of the dataframe
    print(df[['#Pdb', 'Mutation(s)_PDB', 'Temp_K', 'ddG_kcal_mol']].head())
    
    # Save the dataframe to a csv file
    df.to_csv(output_file_path, index=False)