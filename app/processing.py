"""
Core data-processing and quality-control logic for BioData-QC.

Exposes the two stages used by the pipeline:

- ``data_preprocessing``: cleans the raw dataframe, parses the temperature
  column (K, °C or °F, falling back to a default that is tracked in the
  ``temp_assumed`` flag) and computes the mutation-induced binding free-energy
  change ``ddG_kcal_mol``.
- ``quality_control``: flags records using a robust per-complex Z-score, an
  Isolation Forest outlier model and replicate variability, producing the
  ``QC_Flag`` and ``QC_Reason`` columns.
"""

import numpy as np
import pandas as pd
import scipy.constants as const
from sklearn.ensemble import IsolationForest

def data_preprocessing(
    check_temp,
    df_raw: pd.DataFrame,
    col_aff_mut: str,
    col_aff_wt: str,
    col_temp: str = 'Temperature',
    default_temp: float = 298.0
) -> pd.DataFrame:
    """
    Cleans raw uploaded dataframe, parses temperature, ensures affinity columns 
    are float values, and calculates thermodynamic ddG (kcal/mol).
    Rows with no parseable temperature get the default value and are marked
    in a 'temp_assumed' boolean column (read later by quality_control).
    
    Parameters:
        df_raw: Raw input DataFrame.
        col_aff_mut: Mandatory numeric/float column for mutant affinity (e.g., Kd_mut in M).
        col_aff_wt: Mandatory numeric/float column for wild-type affinity (e.g., Kd_wt in M).
        col_temp: Column name for temperature.
        check_temp: Selectbox value that indicates the temperature scale used.
        default_temp: Fallback temperature in Kelvin if missing or unparseable.
    
    Returns:
        pd.DataFrame
            DataFrame with all the preprocessed data (new temperature column, ddG calculated, etc.)
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # Temperature treatment
    temp_str = df[col_temp].astype(str)
    temp_extracted = temp_str.str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
    df['Temperature'] = pd.to_numeric(temp_extracted, errors='coerce')

    df['temp_assumed'] = (
        df['Temperature'].isna()
        | temp_str.str.contains('assumed', case=False, na=False)
    )

    df['Temperature'] = df['Temperature'].fillna(default_temp)
    
    if check_temp == "Kelvin (K)":
        df['Temp_K'] = df['Temperature']
    elif check_temp == "Celsius (C)":
        df['Temp_K'] = df['Temperature'] + 273.15
    elif (check_temp == "Fahrenheit (F)"):
        df['Temp_K'] = (df['Temperature'] - 32) * 5/9 + 273.15

    # Enforce float numeric data type for chosen affinity columns
    # Check if columns exist before attempting to convert to numeric
    if col_aff_mut in df.columns:
        df[col_aff_mut] = pd.to_numeric(df[col_aff_mut], errors='coerce')
    else:
        raise ValueError(f"Mutant affinity column '{col_aff_mut}' not found in dataframe")
    
    if col_aff_wt in df.columns:
        df[col_aff_wt] = pd.to_numeric(df[col_aff_wt], errors='coerce')
    else:
        raise ValueError(f"Wild-type affinity column '{col_aff_wt}' not found in dataframe")
    
    # Drop rows with non-numeric, missing, or non-positive (<= 0) affinity values
    df = df.dropna(subset=[col_aff_mut, col_aff_wt]).copy()
    df = df[(df[col_aff_mut] > 0) & (df[col_aff_wt] > 0)].copy()

    # Calculate Free Energy Change (ddG = R * T * ln(Kd_mut / Kd_wt))
    R_kcal = const.R / 4184.0  # Convert J/(mol*K) to kcal/(mol*K)
    df['ddG_kcal_mol'] = R_kcal * df['Temp_K'] * np.log(df[col_aff_mut] / df[col_aff_wt])
    
    # Clean infinite/NaN calculations
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ddG_kcal_mol']).copy()  # pyright: ignore[reportCallIssue]
    
    return df_clean


def quality_control(
    df_preprocessed: pd.DataFrame,
    col_pdb: str = '#Pdb',
    col_mutation: str = 'Mutation(s)_PDB',
    z_threshold: float = 3.5,
    contamination_rate: float = 0.03,
    replicate_std_threshold: float = 1.5,
    mad_floor: float = 1e-4,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Executes statistical anomaly detection and quality control (QC) flagging.
    
    Parameters:
        df_preprocessed : pd.DataFrame
            Input dataframe with calculated thermodynamic metrics, as returned
            by data_preprocessing (which adds the 'temp_assumed' flag column).
        col_pdb : str
            Column name for PDB structure IDs.
        col_mutation : str
            Column name for mutation strings.
        z_threshold : float
            Cutoff for absolute robust Z-score flagging.
        contamination_rate : float
            Contamination rate for Isolation Forest algorithm.
        replicate_std_threshold : float
            Maximum allowed standard deviation across replicates (kcal/mol).
        mad_floor : float
            Minimum allowable MAD value to prevent division by zero or inflated Z-scores.
        random_state : int
            Seed for reproducibility in Isolation Forest algorithm.  
    
    Returns:
        pd.DataFrame
            Dataframe with added 'z_score', 'iso_forest_outlier', 'std_replicates', 
            'QC_Flag', and 'QC_Reason' columns.
    """
    df = df_preprocessed.copy()
    df['ddG_kcal_mol'] = pd.to_numeric(df['ddG_kcal_mol'], errors='coerce')
    df = df.dropna(subset=['ddG_kcal_mol']).copy()

    # Robust Z-score (grouped by PDB/Protein complex) [cite: 2]
    def calc_robust_z_score(group):
        median = group.median()
        mad = np.median(np.abs(group - median))
        if pd.isna(mad) or mad < mad_floor:
            mad = mad_floor
        return 0.6745 * (group - median) / mad

    if col_pdb in df.columns:
        df['z_score'] = df.groupby(col_pdb)['ddG_kcal_mol'].transform(calc_robust_z_score)
    else:
        df['z_score'] = calc_robust_z_score(df['ddG_kcal_mol'])

    # Isolation Forest outlier detection [cite: 2]
    global_median = df['ddG_kcal_mol'].median()
    ddg_array = df['ddG_kcal_mol'].fillna(global_median).values.reshape(-1, 1)  # pyright: ignore[reportAttributeAccessIssue]
    
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=random_state)  # pyright: ignore[reportArgumentType]
    df['iso_forest_outlier'] = iso_forest.fit_predict(ddg_array)

    # Inter-laboratory replicate variability [cite: 2]
    if col_pdb in df.columns and col_mutation in df.columns:
        variability = df.groupby([col_pdb, col_mutation])['ddG_kcal_mol'].transform('std')
        df['std_replicates'] = variability.fillna(0.0)
    else:
        df['std_replicates'] = 0.0

    # QC Flagging Rules [cite: 2]
    df['QC_Flag'] = 'PASS'
    df['QC_Reason'] = 'Clean data with metrics within normal ranges'

    if 'temp_assumed' in df.columns:
        mask_assumed_temp = df['temp_assumed']
        df.loc[mask_assumed_temp, 'QC_Flag'] = 'REVIEW'
        df.loc[mask_assumed_temp, 'QC_Reason'] = 'Default temperature used'

    mask_high_var = df['std_replicates'] > replicate_std_threshold
    df.loc[mask_high_var, 'QC_Flag'] = 'REVIEW'
    df.loc[mask_high_var, 'QC_Reason'] = f'High inconsistency between replicates (std > {replicate_std_threshold} kcal/mol)'

    mask_xtreme_zscore = np.abs(df['z_score']) > z_threshold
    mask_iso_outlier = df['iso_forest_outlier'] == -1
    mask_anomaly = mask_xtreme_zscore | mask_iso_outlier

    df.loc[mask_anomaly, 'QC_Flag'] = 'ANOMALY'
    df.loc[mask_anomaly, 'QC_Reason'] = 'High statistical deviation: Possible biological hotspot or artifact'

    mask_reject = df['ddG_kcal_mol'].isna()
    df.loc[mask_reject, 'QC_Flag'] = 'REJECT'
    df.loc[mask_reject, 'QC_Reason'] = 'Inability to calculate ddG'

    return df