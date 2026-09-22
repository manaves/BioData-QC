# BioData-QC

Interactive quality control and 3D structural analysis for protein&ndash;protein binding data.

BioData-QC is a Streamlit application that ingests thermodynamic protein binding measurements (for example, the [SKEMPI&nbsp;2.0](https://life.bsc.es/pid/skempi2) benchmark), computes mutation-induced binding free-energy changes (&Delta;&Delta;G), flags low-quality or anomalous records, and renders the corresponding wild-type structures in an interactive 3D viewer.

## Features

- **Flexible data loading** &mdash; fetch the SKEMPI&nbsp;2.0 dataset straight from the web, or upload any CSV with automatic delimiter detection.
- **Configurable column mapping** &mdash; map PDB, mutation, wild-type/mutant affinity and temperature columns through the sidebar; QC thresholds are tunable with live sliders.
- **Thermodynamic calculation** &mdash; computes `ddG = R * T * ln(Kd_mut / Kd_wt)` in kcal/mol with automatic temperature parsing and unit conversion (K, &deg;C, &deg;F).
- **Multi-signal quality control** &mdash; robust per-complex Z-scores, Isolation Forest anomaly detection, and replicate variability analysis, combined into explainable PASS / REVIEW / ANOMALY / REJECT flags.
- **Interactive dashboard** &mdash; &Delta;&Delta;G distributions, WT-vs-mutant affinity scatter, amino-acid substitution matrices, PDB coverage and QC diagnostic plots (Plotly).
- **3D structural viewer** &mdash; fetches structures from the RCSB PDB API and displays them with py3Dmol/stmol, with mutation-site highlighting, distance-based neighbour selection, multiple representations and color schemes.
- **Executive summary export** &mdash; generates a self-contained HTML report with structure metadata, &Delta;&Delta;G metrics, mutation tables and key plots.
- **Dataset export** &mdash; download the original or fully processed dataset as CSV.

## Pipeline

1. **Preprocessing** (`app/processing.py::data_preprocessing`)
   - Normalises column names and parses the temperature column (defaulting to 298 K when missing, tracked in a `temp_assumed` flag).
   - Converts affinity columns to numeric, dropping non-numeric, missing or non-positive values.
   - Computes `ddG_kcal_mol` and removes infinite/NaN results.

2. **Quality control** (`app/processing.py::quality_control`)
   - **Robust Z-score** per PDB complex using median/MAD (with a configurable MAD floor to avoid inflated scores on low-variance groups).
   - **Isolation Forest** for unsupervised multidimensional outlier detection.
   - **Replicate spread** as the standard deviation of &Delta;&Delta;G for identical mutations across publications.
   - Assigns a `QC_Flag` and human-readable `QC_Reason` per record.

3. **Structural enrichment** &mdash; the 4-character `PDB_ID` is derived from the PDB column and used to fetch and render structures.

### QC flags

| Flag | Meaning |
| --- | --- |
| `PASS` | Metrics within normal ranges. |
| `REVIEW` | Assumed/default temperature or high replicate inconsistency (std &gt; 1.5 kcal/mol). |
| `ANOMALY` | Extreme robust Z-score or Isolation Forest outlier (possible biological hotspot or assay artifact). |
| `REJECT` | &Delta;&Delta;G could not be computed (missing data or non-physical affinity). |

## Getting started

### Prerequisites

- Python 3.11+ (project tested on a conda environment named `biodata_env`)
- Internet access for the SKEMPI&nbsp;2.0 download and RCSB PDB structure fetching

### Installation

```bash
git clone git@github.com:manaves/BioData-QC.git
cd BioData-QC

conda create -n biodata_env python=3.11 -y
conda activate biodata_env
pip install -r requirements.txt
```

### Run the app

Run from the repository root so the bundled logo path resolves correctly:

```bash
streamlit run app/main.py
```

Then open the local URL printed by Streamlit (default
`http://localhost:8501`). You can also find the Streamlit app in https://biodata-qc.streamlit.app/.

## Usage

1. On the welcome screen, click **Load example dataset (SKEMPI v2)** to download the benchmark and auto-run the pipeline with default parameters, or **Upload local CSV** to bring your own data.
2. In the sidebar, verify or adjust the column mappings and QC parameters (Robust Z-Score cutoff, Isolation Forest contamination, MAD floor).
3. Click **Run Pipeline**.
4. Explore results in **Dataset Metrics & Table** and **3D Structural Viewer**, then export the dataset or generate an executive summary report.

### Expected input columns

| Column | Description | Example |
| --- | --- | --- |
| PDB | PDB entry plus the two chain identifiers. | `1JTG_A_B` |
| Mutation | `<WT_AA><Chain><ResNum><Mut_AA>`; multiple mutations comma-separated. | `EA104A` |
| Wild-type affinity | Equilibrium dissociation constant (Kd) of the wild-type, in molar. | `1.20e-09` |
| Mutant affinity | Equilibrium dissociation constant (Kd) of the mutant, in molar. | `4.50e-08` |
| Temperature | Experimental temperature (K, &deg;C or &deg;F). | `298.15` |

## Project structure

```
BioData-QC/
├── app/
│   ├── main.py            # Streamlit UI, session state and dashboard
│   ├── processing.py      # Preprocessing and QC logic
│   ├── utils.py           # 3D rendering, plots and report generation
│   └── logo/logo.png      # Application logo
├── requirements.txt
└── LICENSE
```

## Tech stack

| Area | Libraries |
| --- | --- |
| App / UI | Streamlit |
| Data | pandas, NumPy, SciPy |
| ML / QC | scikit-learn (Isolation Forest) |
| Visualisation | Plotly |
| 3D structures | py3Dmol, stmol |
| Structures source | RCSB PDB Web API |

## Citation

If you use the SKEMPI&nbsp;2.0 dataset, please cite:

> Jankauskaitė J, Jiménez-García B, Dapkūnas J, Fernández-Recio J, Moal IH.
> **SKEMPI 2.0: an updated benchmark of changes in protein&ndash;protein binding
> energy, kinetics and thermodynamics upon mutation.** *Bioinformatics*,
> 35(3):462&ndash;469, 2019. https://doi.org/10.1093/bioinformatics/bty635

## License

Released under the [MIT License](LICENSE).

## Contact

- GitHub: [@manaves](https://github.com/manaves)
- LinkedIn: [Maria Navarro Paredes](https://www.linkedin.com/in/maria-navarro-paredes/)
- Email: navarroparedesmaria@gmail.com
