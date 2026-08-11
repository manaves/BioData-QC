import pandas as pd
import numpy as np
import scipy.constants as const
import os

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
    
    # Check if the columns Affinity_mut (M) and Affinity_wt (M) have missing values
    if df['Affinity_mut (M)'].isna().any() or df['Affinity_wt (M)'].isna().any():
        print("~~~ Warning: Missing values found in the Affinity_mut (M) and Affinity_wt (M) columns.")
        df = df[df['Affinity_mut (M)'].notna() & df['Affinity_wt (M)'].notna()]
    else:
        print("~~~ No missing values detected in Affinity_mut (M) or Affinity_wt (M).")
    
    # Ensure the affinity columns are numeric
    affinity_mut = pd.to_numeric(df['Affinity_mut (M)'], errors='coerce')
    affinity_wt = pd.to_numeric(df['Affinity_wt (M)'], errors='coerce')
    
    # Ideal gas constant
    R_j = const.R # In J/(mol*K)
    # Convert to kcal/(mol*K)
    R_kcal = R_j / 4184 # 1 kcal = 4184 J
    print("~~~ Ideal gas constant: ", R_kcal, "kcal/(mol*K)")

    # Apply the thermodynamic equation to calculate the free energy change (R*T*ln(affinity_mut / affinity_wt))
    df['ddG_kcal_mol'] = R_kcal * df['Temp_K'] * np.log(affinity_mut / affinity_wt)
    print("~~~ Finished calculating the free energy change.")
    
    return df

# Main execution
if __name__ == "__main__":
    
    df = data_preprocessing_skempi(input_file_path)
    
    # Show the first 5 rows of the dataframe
    print(df[['#Pdb', 'Mutation(s)_PDB', 'Temp_K', 'ddG_kcal_mol']].head())
    
    # Save the dataframe to a csv file
    df.to_csv(output_file_path, index=False)