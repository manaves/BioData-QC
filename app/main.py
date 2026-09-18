import io
import math
import re
import requests
import pandas as pd
import streamlit as st

# Custom module imports
from processing import data_preprocessing, quality_control
from utils import (
    render_wt_structure_highlight, 
    parse_mutation_info,
    build_executive_summary_report,
    plot_z_score,
    plot_iso_forest_outlier,
    plot_std_replicates,
    plot_qc_flag,
    plot_qc_reason,
    plot_aa_transition_matrix,
    plot_affinity_scatter,
    plot_ddg_distribution,
    plot_pdb_counts
)

# SKEMPI url (example)
SKEMPI_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"

# Hardcoded SKEMPI v2 column mapping.
# The keys match the `key=` of the column selectboxes in the sidebar form.
SKEMPI_COLUMNS = {
    "sel_pdb": "#Pdb",
    "sel_mut": "Mutation(s)_PDB",
    "sel_aff_wt": "Affinity_wt_parsed",
    "sel_aff_mut": "Affinity_mut_parsed",
    "sel_temp": "Temperature",
}

# Default QC parameters: used both as slider defaults and for the example auto-run.
DEFAULT_QC_PARAMS = {
    "z_thresh": 3.5,
    "iso_contam": 0.03,
    "mad_floor": 0.000001,
}

# Rows per page in the processed-dataset table
PAGE_SIZE = 10

# CSS for the first-page upload control. It turns the file uploader into a
# button that matches the primary "Load example dataset" button, renames its
# default "Upload" label to "Upload local CSV", and keeps the uploader's own
# size hint ("200MB per file • CSV") underneath.
START_UPLOADER_STYLE = """
<style>
.st-key-start_uploader [data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 0.25rem !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    min-height: 0 !important;
}
.st-key-start_uploader [data-testid="stFileUploaderDropzone"] > div {
    width: 100% !important;
}
.st-key-start_uploader [data-testid="stFileUploaderDropzone"] button {
    width: 100% !important;
    min-height: 2.5rem !important;
    background-color: #FF4B4B !important;
    border: 1px solid #FF4B4B !important;
    color: #FFFFFF !important;
}
/* Hide Streamlit's built-in upload icon + "Upload" text ... */
.st-key-start_uploader [data-testid="stFileUploaderDropzone"] button * {
    font-size: 0 !important;
}
/* ... and replace it with our own label (14px = the primary button label size). */
.st-key-start_uploader [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after {
    content: "Upload local CSV";
    font-size: 0.875rem !important;
}
.st-key-start_uploader [data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #E63E3E !important;
    border-color: #E63E3E !important;
    color: #FFFFFF !important;
}
.st-key-start_uploader [data-testid="stFileUploaderDropzoneInstructions"] {
    align-self: center !important;
    text-align: center !important;
}
</style>
"""

# Columns created by the pipeline (preprocessing + QC). Shown next to the 5
# sidebar-selected columns in the processed-dataset table.
CALCULATED_COLS = [
    "Temperature",          # Numeric temperature parsed from the selected Temperature column
    "Temp_K",               # Temperature converted to Kelvin
    "ddG_kcal_mol",         # Calculated binding free-energy change
    "PDB_ID",               # 4-character PDB code extracted from the PDB column
    "z_score",              # Robust Z-score (grouped by PDB)
    "iso_forest_outlier",   # Isolation Forest outlier flag
    "std_replicates",       # Std-dev of ddG across replicates
    "QC_Flag",              # QC pass/review/anomaly/reject flag
    "QC_Reason",            # Human-readable QC reason
]

# Cache the web download so it doesn't re-fetch on every button click
@st.cache_data(show_spinner="Downloading SKEMPI v2 dataset from URL...")
def load_skempi_from_url(url: str) -> pd.DataFrame:
  # Using a custom User-Agent header helps avoid server blocks
  headers = {"User-Agent": "Mozilla/5.0"}
  response = requests.get(url, headers=headers, timeout=15)
  response.raise_for_status()

  # Read CSV directly from memory buffer
  return pd.read_csv(io.StringIO(response.text), sep=None, engine="python")

# Config
st.set_page_config(page_title="BioData Pipeline & 3D Viewer", page_icon="🧬", layout="wide")

LOGO_PATH = "app/logo/logo.png"

st.logo(
    LOGO_PATH,
    size="large",
    icon_image=LOGO_PATH
)

# --- Session state initialization ---
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = None  # None | "example" | "upload"
if "auto_run" not in st.session_state:
    st.session_state["auto_run"] = False


# --- Helpers ---
def clear_results():
    """Remove previous pipeline results (and table pagination) from session state."""
    for key in ("df_qc", "col_pdb", "col_mut", "col_aff_wt", "col_aff_mut",
                "col_temp", "dataset_page", "exec_report"):
        st.session_state.pop(key, None)


def read_csv_file(file) -> pd.DataFrame:
    """Read an uploaded CSV with automatic separator detection (with fallback)."""
    try:
        return pd.read_csv(file, sep=None, engine="python")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=",")


def load_example_dataset():
    """Fetch SKEMPI v2, auto-map its columns and trigger the automatic run."""
    try:
        df = load_skempi_from_url(SKEMPI_URL)
    except Exception as e:
        st.error(f"Failed to fetch dataset from URL: {e}")
        return

    st.session_state["df_raw"] = df
    st.session_state["data_mode"] = "example"

    for sel_key, col_name in SKEMPI_COLUMNS.items():
        st.session_state.pop(sel_key, None)  # clear any stale mapping
        if col_name in df.columns:
            st.session_state[sel_key] = col_name

    clear_results()
    st.session_state["auto_run"] = True
    st.rerun()


def run_pipeline(df_raw, col_pdb, col_mut, col_aff_wt, col_aff_mut, col_temp,
                 check_temp, z_thresh, iso_contam, mad_floor):
    """Preprocessing + QC + storing results in session state."""
    try:
        with st.spinner("Processing thermodynamic calculations and running QC..."):
            # Preprocessing data
            df_preprocessed = data_preprocessing(
                df_raw=df_raw,
                col_aff_wt=col_aff_wt,
                col_aff_mut=col_aff_mut,
                col_temp=col_temp,
                check_temp=check_temp
            )

            # QC analysis
            df_qc = quality_control(
                df_preprocessed,
                col_pdb=col_pdb,
                col_mutation=col_mut,
                z_threshold=z_thresh,
                contamination_rate=iso_contam,
                mad_floor=mad_floor
            )
            
            df_qc = df_qc.reset_index(drop=True)
            df_qc['PDB_ID'] = df_qc[col_pdb].astype(str).str[:4].str.upper()

            # Store all required column variables in session state
            st.session_state['df_qc'] = df_qc
            st.session_state['col_pdb'] = col_pdb
            st.session_state['col_mut'] = col_mut
            st.session_state['col_aff_wt'] = col_aff_wt
            st.session_state['col_aff_mut'] = col_aff_mut
            st.session_state['col_temp'] = col_temp

            # New results -> jump back to the first page of the table
            st.session_state.pop("dataset_page", None)

        st.success(f"Successfully processed {len(df_raw):,} records!")

    except KeyError as ke:
        st.error(f"Column error: Missing column {ke}. Please check your sidebar dropdown selections.")
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")


# --- "GET STARTED" SCREEN (only while no dataset is loaded) ---
if st.session_state["df_raw"] is None:
    st.markdown(START_UPLOADER_STYLE, unsafe_allow_html=True)
    st.title("👋 Welcome to BioData")

    with st.container(border=True):
        st.subheader("Get started")
        st.markdown("""
            Welcome to the **BioData Quality Control & 3D Viewer** application!

            This application processes thermodynamic protein binding data, executes outlier detection pipelines, 
            and renders 3D macromolecular structures.
            
            **Quick Instructions:**
            - Load the **SKEMPI v2** example dataset or upload your own CSV to get started.
            - Select column mappings and QC parameter thresholds in the sidebar.
            - Click **Run Pipeline** to compute thermodynamic properties and generate interactive 3D visualizations.

            **Important - Expected Column Meanings**
            1. **PDB Column**: The PDB entry for the complex, followed by the chain identifiers for the two subunits (e.g., `1JTG_A_B`).
            2. **Mutation Column**: Contains mutation designations. Format: `<WT_AA><Chain><ResNum><Mut_AA>` (e.g., `EA104A`). Multiple mutations are comma-separated.
            3. **Wild-Type Affinity Column**: Equilibrium dissociation constant ($K_d$) of the wild-type protein.
            4. **Mutant Affinity Column**: Equilibrium dissociation constant ($K_d$) of the mutant protein.
            5. **Temperature Column**: Experimental temperature ($K$, $^\circ C$, or $^\circ F$).
        """)

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Load example dataset (SKEMPI v2)", type="primary", width="stretch"):
                load_example_dataset()

        with col_btn2:
            # The uploader *is* the action: clicking it opens the file picker
            # directly. START_UPLOADER_STYLE styles it like the primary button
            # on the left (that style also provides the visible label text).
            uploaded_file = st.file_uploader(
                "Upload local CSV",
                type=["csv"],
                key="start_uploader",
                label_visibility="collapsed",
            )

    if uploaded_file is not None:
        try:
            st.session_state["df_raw"] = read_csv_file(uploaded_file)
            st.session_state["data_mode"] = "upload"

            for sel_key in SKEMPI_COLUMNS:
                st.session_state.pop(sel_key, None)
            clear_results()

            st.rerun()
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")

    st.stop()


# --- DATASET LOADED ---
df_raw = st.session_state["df_raw"]
st.write("Data loaded successfully! Total rows (before processing):", len(df_raw))

# --- SIDEBAR: DATA SOURCE ---
with st.sidebar:
    st.title("Data Source")

    new_file = st.file_uploader(
        f"Upload a new CSV (max 200 MB) (replaces the current dataset)",
        type=["csv"],
        key="sidebar_uploader",
    )

    if new_file is None:
        # Uploader emptied: forget the previous file so a re-upload is detected
        st.session_state.pop("uploaded_sig", None)
    else:
        # A file_uploader keeps its file across reruns, so we only process it
        # when it differs from the one already loaded.
        file_sig = (getattr(new_file, "file_id", None), new_file.name, new_file.size)
        if st.session_state.get("uploaded_sig") != file_sig:
            try:
                df = read_csv_file(new_file)
                st.session_state["df_raw"] = df
                st.session_state["data_mode"] = "upload"
                st.session_state["uploaded_sig"] = file_sig

                # Reset the column mapping 
                for sel_key in SKEMPI_COLUMNS:
                    st.session_state[sel_key] = None

                clear_results()
                st.rerun()
            except Exception as e:
                st.error(f"Could not read the CSV file: {e}")

    if st.button("Load SKEMPI v2 example", width="stretch"):
        load_example_dataset()

    # Instructions & column meanings, available at any time
    with st.expander("Instructions"):
        st.markdown("""
            - Load the **SKEMPI v2** example dataset or upload your own CSV to get started.
            - Select column mappings and QC parameter thresholds in the sidebar.
            - Click **Run Pipeline** to compute thermodynamic properties and generate interactive 3D visualizations.                    
        """)

    st.title("Pipeline Configuration")

# Form to add the information needed for the execution
with st.sidebar.form(key="pipeline_config_form"):
    # Column information
    st.info("Select or verify the column mappings for execution.")

    column_options = df_raw.columns.tolist()

    col_pdb = st.selectbox("PDB Column:", options=column_options, key="sel_pdb",
                            index=None, placeholder="Select a column...",
                            help="The PDB entry for the complex, followed by the chain identifiers for the two subunits (e.g., `1JTG_A_B`).")
    col_mut = st.selectbox("Mutation Column:", options=column_options, key="sel_mut",
                            index=None, placeholder="Select a column...",
                            help="Contains mutation designations. Format: `<WT_AA><Chain><ResNum><Mut_AA>` (e.g., `EA104A`). Multiple mutations are comma-separated.")
    col_aff_wt = st.selectbox("Wild-Type Affinity Column (Numeric Float):", options=column_options,
                                key="sel_aff_wt", index=None, placeholder="Select a column...",
                                help="Equilibrium dissociation constant ($K_d$) of the wild-type protein.")
    col_aff_mut = st.selectbox("Mutant Affinity Column (Numeric Float):", options=column_options,
                                key="sel_aff_mut", index=None, placeholder="Select a column...",
                                help="Equilibrium dissociation constant ($K_d$) of the mutant protein.")
    col_temp = st.selectbox("Temperature Column:", options=column_options, key="sel_temp",
                                index=None, placeholder="Select a column...",
                                help="Experimental temperature ($K$, $^\circ C$, or $^\circ F$).")
    check_temp = st.selectbox("Temperature scale used:", options=["Kelvin (K)", "Celsius (C)", "Fahrenheit (F)"])

    st.divider()
    
    # QC information
    st.subheader("QC Parameters")
    z_thresh = st.slider("Robust Z-Score Cutoff:", 1.5, 5.0, DEFAULT_QC_PARAMS["z_thresh"], 0.1)
    iso_contam = st.slider("Isolation Forest Contamination:", 0.01, 0.15, DEFAULT_QC_PARAMS["iso_contam"], 0.01)

    mad_floor = st.slider(
        "MAD Floor (Min Variance Cutoff):",
        min_value=0.000001,
        max_value=0.1,
        value=DEFAULT_QC_PARAMS["mad_floor"],
        step=0.0005,
        format="scientific",
        help="Minimum allowable Median Absolute Deviation (MAD) value. Prevents division by zero and extreme Z-scores for groups with near-zero variance."
    )

    submit_button = st.form_submit_button(label="Run Pipeline", width='content')

if submit_button:
    # Validate that every required column has been selected (selectboxes start
    # empty for uploaded CSVs, so a None here means the user skipped one).
    _missing = [name for name, val in (
        ("PDB Column", col_pdb),
        ("Mutation Column", col_mut),
        ("Wild-Type Affinity Column", col_aff_wt),
        ("Mutant Affinity Column", col_aff_mut),
        ("Temperature Column", col_temp),
    ) if val is None]
    
    if _missing:
        st.error(f"Please select a value for: {', '.join(_missing)}")
    else:
        run_pipeline(df_raw, col_pdb, col_mut, col_aff_wt, col_aff_mut,
                     col_temp, check_temp, z_thresh, iso_contam, mad_floor)

# --- Automatic run for the example dataset (default QC parameters) ---
if st.session_state.get("auto_run"):
    st.session_state["auto_run"] = False  # Consume the flag: run only once

    status_ph = st.empty()
    status_ph.info("SKEMPI v2 example loaded: running the pipeline with the default QC parameters...")

    run_pipeline(
        df_raw,
        col_pdb=col_pdb,
        col_mut=col_mut,
        col_aff_wt=col_aff_wt,
        col_aff_mut=col_aff_mut,
        col_temp=col_temp,
        check_temp="Kelvin (K)",
        z_thresh=DEFAULT_QC_PARAMS["z_thresh"],
        iso_contam=DEFAULT_QC_PARAMS["iso_contam"],
        mad_floor=DEFAULT_QC_PARAMS["mad_floor"],
    )

    status_ph.empty()  # The message disappears as soon as the run finishes

if "df_qc" not in st.session_state:
    st.info("Select the column mappings and QC parameters in the sidebar, then press **Run Pipeline**.")

# --- DISPLAY DASHBOARD ---
if 'df_qc' in st.session_state:
    # Important variables
    df_qc = st.session_state['df_qc']
    col_pdb = st.session_state.get('sel_pdb') # st.session_state.get('col_pdb') or
    col_mut = st.session_state['col_mut']
    col_aff_wt = st.session_state.get('col_aff_wt')
    col_aff_mut = st.session_state.get('col_aff_mut')
    col_temp = st.session_state.get('col_temp')

    # `key` persists the active tab in session state, so widget interactions
    # inside a tab (e.g. changing the mutation) no longer snap back to tab 1.
    tab1, tab2 = st.tabs(["Dataset Metrics & Table", "3D Structural Viewer"], key="main_tabs")

    # Processed dataset
    with tab1:
        st.subheader("Processed Dataset")
        
        # Build the column list shown in the table:
        #   - The 5 columns selected in the sidebar (PDB, Mutation, WT/Mut affinity, Temperature)
        #   - The new columns calculated during preprocessing & QC (see CALCULATED_COLS)
        # `temp_assumed` is intentionally excluded from both the table and the processed-dataset download below.
        display_cols = []
        for c in [col_pdb, col_mut, col_aff_wt, col_aff_mut, col_temp] + CALCULATED_COLS:
            if c and c in df_qc.columns and c not in display_cols:
                display_cols.append(c)
        df_qc_display = df_qc[display_cols] if display_cols else df_qc

        # --- Paginated table ---
        total_rows = len(df_qc_display)
        page_count = max(1, math.ceil(total_rows / PAGE_SIZE))
        page = st.pagination(page_count, key="dataset_page")  # Pages are 1-indexed
        start = (page - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total_rows)
        st.caption(f"Showing rows {start + 1:,}–{end:,} of {total_rows:,} · page {page} of {page_count}")
        st.dataframe(df_qc_display.iloc[start:end], width='stretch')
        
        # --- Download buttons ---
        st.caption("Download datasets:")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇ Download original dataset",
                data=df_raw.to_csv(index=False).encode('utf-8'),
                file_name="original_dataset.csv",
                mime="text/csv",
                width='stretch',
            )
        with dl_col2:
            # Full processed dataset (all columns) but without `temp_assumed`
            df_qc_download = df_qc.drop(columns=['temp_assumed'], errors='ignore')
            st.download_button(
                label="⬇ Download processed dataset (all columns)",
                data=df_qc_download.to_csv(index=False).encode('utf-8'),
                file_name="processed_dataset.csv",
                mime="text/csv",
                width='stretch',
            )

        st.divider()

        # Organized 2-Column Grid Layout for Charts
        st.subheader("Dataset Thermodynamic Summary")
        c1, c2 = st.columns(2)
        
        # --- THERMODYNAMIC CHARTS --- 
        with c1:
            if 'ddG_kcal_mol' in df_qc.columns:
                st.plotly_chart(plot_ddg_distribution(df_qc), width='content')
            if col_aff_wt and col_aff_mut:
                st.plotly_chart(plot_affinity_scatter(df_qc, col_aff_wt, col_aff_mut), width='content')

        with c2:
            if 'PDB_ID' in df_qc.columns:
                st.plotly_chart(plot_pdb_counts(df_qc), width='content')
            if col_mut in df_qc.columns:
                st.plotly_chart(plot_aa_transition_matrix(df_qc, col_mut), width='content')

        st.divider()

        # --- QUALITY CONTROL & OUTLIER CHARTS ---
        st.subheader("Quality Control & Outlier Diagnostics")
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            if 'z_score' in df_qc.columns:
                st.plotly_chart(plot_z_score(df_qc), width='content')
            st.plotly_chart(plot_qc_flag(df_qc), width='content')

        with g_col2:
            if 'iso_forest_outlier' in df_qc.columns:
                st.plotly_chart(plot_iso_forest_outlier(df_qc), width='content')
            if 'std_replicates' in df_qc.columns:
                st.plotly_chart(plot_std_replicates(df_qc), width='content')
            st.plotly_chart(plot_qc_reason(df_qc), width='content')
    
    # 3D viewer
    with tab2:
        st.header("3D Structure & Mutation Analysis")

        available_pdbs = sorted(df_qc['PDB_ID'].dropna().unique())

        if available_pdbs:
            # --- SELECTION ---
            st.subheader("Selection")
            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                selected_pdb = st.selectbox("1. Select PDB Complex:", available_pdbs, key="tab2_pdb")

            pdb_subset = df_qc[df_qc['PDB_ID'] == selected_pdb]
            available_muts = sorted(pdb_subset[col_mut].dropna().unique())

            with sel_col2:
                selected_mutation = st.selectbox("2. Select Mutation:", available_muts, key="tab2_mut")

            # Metadata info bar
            mut_row = pdb_subset[pdb_subset[col_mut] == selected_mutation].iloc[0]
            st.info(
                f"**PDB ID:** `{selected_pdb}` | "
                f"**Mutation:** `{selected_mutation}` | "
                f"**ΔΔG:** `{mut_row['ddG_kcal_mol']:.2f} kcal/mol` | "
                f"**QC Flag:** `{mut_row['QC_Flag']}` | "
                f"**QC Reason:** `{mut_row['QC_Reason']}`"
            )

            # --- MUTATION PROPERTIES TABLE ---
            st.subheader("Mutation Properties")
            parsed_muts = parse_mutation_info(selected_mutation)  # pyright: ignore[reportArgumentType]
            focus_residue = None
            if parsed_muts:
                mut_df = pd.DataFrame(parsed_muts)
                mut_event = st.dataframe(
                    mut_df[['chain', 'resnum', 'wt_name', 'mut_name', 'wt_class', 'mut_class', 'mw_change']],
                    hide_index=True,
                    width='content',
                    on_select="rerun",
                    selection_mode="single-row",
                    # Key includes the mutation so the selection resets when the
                    # user switches mutation (row indices are positional).
                    key=f"mut_props_table::{selected_mutation}",
                )
                # The returned DataframeState is a TypedDict; use .get so the
                # optional keys don't trip the type checker.
                selection = mut_event.get("selection")
                selected_rows = list(selection.get("rows", [])) if selection else []
                if selected_rows and 0 <= selected_rows[0] < len(parsed_muts):
                    focus_residue = parsed_muts[selected_rows[0]]
                st.caption("Click a row to center, zoom and flash that residue in the 3D view.")
            else:
                st.write("No standard mutation details could be parsed from string.")

            st.divider()

            # --- CANVAS (LEFT) + APPEARANCE CONTROLS (RIGHT) ---
            viewer_col, controls_col = st.columns([1.2, 1])

            with controls_col:
                st.subheader("Appearance & Styling Controls")
                opt_col1, opt_col2 = st.columns(2)

                with opt_col1:
                    st.caption("Structure")
                    selected_style = st.selectbox(
                        "Main representation:",
                        ["Cartoon", "Spheres", "Sticks", "Ribbon Trace"],
                        key="tab2_style"
                    )
                    color_scheme = st.selectbox(
                        "Color scheme:",
                        ["Chain ID", "Secondary Structure", "N-to-C Spectrum", "B-Factor"],
                        key="tab2_color"
                    )
                    bg_color = st.selectbox("Canvas background:", ["White", "Dark"], key="tab2_bg")
                    show_surface = st.checkbox("Surface overlay:", value=False, key="tab2_surface")

                with opt_col2:
                    st.caption("Mutation & neighbors")
                    mut_repr = st.selectbox(
                        "Site representation:",
                        ["Sticks & Spheres", "Spheres Only", "Sticks Only"],
                        key="tab2_mut_repr"
                    )
                    highlight_color = st.color_picker("Highlight color:", "#FF007F", key="tab2_hl_col")
                    show_neighbors = st.checkbox("Highlight neighbors:", value=False, key="tab2_neigh")
                    neighbor_radius = st.slider("Radius (Å):", 3.0, 10.0, 5.0, 0.5, key="tab2_radius")
                    neighbor_color = st.color_picker("Neighbor color:", "#00E5FF", key="tab2_neigh_col")

            with viewer_col:
                st.subheader("Wild-Type Structure")

                success_wt, wt_data = render_wt_structure_highlight(
                    pdb_id=selected_pdb,  # pyright: ignore[reportArgumentType]
                    mutation_str=selected_mutation,  # pyright: ignore[reportArgumentType]
                    style_type=selected_style,
                    color_scheme=color_scheme,
                    highlight_color=highlight_color,
                    mut_repr=mut_repr,
                    bg_color=bg_color,
                    show_surface=show_surface,
                    show_neighbors=show_neighbors,
                    neighbor_radius=neighbor_radius,
                    neighbor_color=neighbor_color,
                    width=560,
                    height=460,
                    focus_residue=focus_residue
                )

                if success_wt:
                    st.download_button(
                        label="Download WT PDB",
                        data=wt_data,
                        file_name=f"{selected_pdb}_wt.pdb",
                        mime="chemical/x-pdb",
                        key="dl_wt"
                    )
                else:
                    st.error(wt_data)

                # --- EXECUTIVE SUMMARY EXPORT ---
                st.divider()
                report_key = f"{selected_pdb}::{selected_mutation}"
                if st.button("Generate executive summary report", key="gen_report", width="stretch"):
                    report_html = build_executive_summary_report(
                        df_qc=df_qc,
                        pdb_id=selected_pdb,
                        mutation=selected_mutation,
                        col_mut=col_mut,
                        col_aff_wt=col_aff_wt,
                        col_aff_mut=col_aff_mut,
                        col_temp=col_temp,
                        pdb_block=wt_data if success_wt else None,
                        pdb_source=f"RCSB PDB Web API ({selected_pdb})" if success_wt else None,
                    )
                    st.session_state["exec_report"] = {"key": report_key, "html": report_html}

                report = st.session_state.get("exec_report")
                if report and report.get("key") == report_key:
                    safe_mut = re.sub(r"[^A-Za-z0-9_-]+", "_", str(selected_mutation)).strip("_") or "mutation"
                    st.download_button(
                        label="⬇ Download executive summary report",
                        data=report["html"].encode("utf-8"),
                        file_name=f"{selected_pdb}_{safe_mut}_executive_summary.html",
                        mime="text/html",
                        key="dl_report",
                    )
                else:
                    st.caption("Generate the report to enable the download button.")

        else:
            st.warning("No valid 4-character PDB IDs found in the dataset.")