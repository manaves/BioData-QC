import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

# Define project root and directory paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_PATH, 'data', 'processed')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'data', 'processed')

# File names
input_file = 'skempi_v2_processed.csv'
output_file = 'skempi_v2_qc.csv'

# Absolute file paths
input_file_path = os.path.join(INPUT_PATH, input_file)
output_file_path = os.path.join(OUTPUT_PATH, output_file)

def quality_control(df):
    """
    Executes statistical anomaly detection and quality control (QC) flagging.
    
    This module performs data auditing without blind deletion. It labels 
    records based on metadata assumptions, replica variability, and 
    statistical deviations to preserve biological ground truth (e.g., hot-spots).

    Parameters:
        df (pandas.DataFrame): Preprocessed dataset containing thermodynamic calculations.

    Returns:
        df (pandas.DataFrame): Dataset augmented with QC flags and audit trails.
    """
    
    print("Initializing quality control...")
    
    # 1. Data cleaning and type enforcement
    # Drop records missing target values before calculating statistical distributions
    if df['ddG_kcal_mol'].isna().any():
        print("~~~ Warning: NaN values found in the ddG_kcal_mol column. Dropping missing entries...")
        df = df[df['ddG_kcal_mol'].notna()]
        
    df['ddG_kcal_mol'] = pd.to_numeric(df['ddG_kcal_mol'], errors='coerce')
    
    # 2. Contextual robust outlier detection (Grouped by Protein/PDB)
    # Computes MAD-based Z-scores per macromolecular complex to respect protein-specific baselines
    def calc_robust_z_score(group):
        median = group.median()
        mad = np.median(np.abs(group - median))
        
        # Prevent division by zero or NaN propagation
        if pd.isna(mad) or mad < 1e-4:
            mad = 1e-4
            
        return 0.6745 * (group - median) / mad
    
    df['z_score'] = df.groupby('#Pdb')['ddG_kcal_mol'].transform(calc_robust_z_score)
    
    # 3. Unsupervised multidimensional anomaly detection
    # Uses Isolation Forest to isolate structural or numerical outliers
    global_median = df['ddG_kcal_mol'].median()
    ddg_array = df['ddG_kcal_mol'].fillna(global_median).values.reshape(-1, 1)
    
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    df['iso_forest_outlier'] = iso_forest.fit_predict(ddg_array)  # -1 for outliers, 1 for inliers    

    # 4. Experimental variability analysis (Inter-laboratory replicates)
    # Calculates standard deviation for identical mutations across distinct publications
    variability = df.groupby(['#Pdb', 'Mutation(s)_PDB'])['ddG_kcal_mol'].transform('std')
    df['std_replicates'] = variability.fillna(0.0)  # Single measurements default to 0.0 standard deviation

    # 5. Informative flagging engine
    # Assign default PASS status to all records
    df['QC_Flag'] = 'PASS'
    df['QC_Reason'] = 'Clean data with metrics within normal ranges'
    
    # Rule 1: Metadata Caution (Assumed temperature)
    mask_assumed_temp = df['Temperature'].astype(str).str.contains('assumed', case=False, na=False)
    df.loc[mask_assumed_temp, 'QC_Flag'] = 'REVIEW'
    df.loc[mask_assumed_temp, 'QC_Reason'] = 'Default temperature (298K)'
    
    # Rule 2: Experimental Discrepancy (Replicate standard deviation > 1.5 kcal/mol)
    mask_high_var = df['std_replicates'] > 1.5
    df.loc[mask_high_var, 'QC_Flag'] = 'REVIEW'
    df.loc[mask_high_var, 'QC_Reason'] = 'High inconsistency between replicates (std > 1.5 kcal/mol)'
    
    # Rule 3: Statistical Anomaly (Potential biological Hot-Spot or assay artifact)
    mask_xtreme_zscore = np.abs(df['z_score']) > 3.5
    mask_iso_outlier = df['iso_forest_outlier'] == -1
    mask_anomaly = mask_xtreme_zscore | mask_iso_outlier
    
    df.loc[mask_anomaly, 'QC_Flag'] = 'ANOMALY'
    df.loc[mask_anomaly, 'QC_Reason'] = 'High statistical deviation: Possible biological hotspot or artifact'

    # Rule 4: Mathematical Invalidation (Missing or non-physical Kd inputs)
    mask_reject = df['ddG_kcal_mol'].isna()
    df.loc[mask_reject, 'QC_Flag'] = 'REJECT'
    df.loc[mask_reject, 'QC_Reason'] = 'Inability to calculate ddG (missing data or affinity <= 0)'
    
    # 6. Audit Trail Summary
    print("\n~~~ Quality control summary")
    print(df['QC_Flag'].value_counts())
    
    mask_z_outlier = np.abs(df['z_score']) > 3.5
    mask_iso_outlier = df['iso_forest_outlier'] == -1

    n_entries_z_outlier = mask_z_outlier.sum()
    n_entries_iso_outlier = mask_iso_outlier.sum()

    print(f"~~~ Entries flagged due to extreme statistical outlier (Z-score): {n_entries_z_outlier}")
    print(f"~~~ Entries flagged due to Isolation Forest outlier: {n_entries_iso_outlier}")
    
    return df

# Main execution
if __name__ == "__main__":
    try:
        df = pd.read_csv(input_file_path)
        df_qc = quality_control(df)
        
        # Persist processed data
        df_qc.to_csv(output_file_path, index=False)
        print(f"\nFile saved successfully: {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path}.")
        print("Please check directory path and input file name.")