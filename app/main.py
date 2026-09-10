import os
import pandas as pd
import streamlit as st

# Custom module imports
from processing import data_preprocessing, quality_control
from utils import (
    render_wt_structure_highlight, 
    parse_mutation_info,
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

# Config
st.set_page_config(page_title="BioData Pipeline & 3D Viewer", page_icon="🧬", layout="wide")

# Pop-up dialog
@st.dialog("👋 Welcome to BioData-QC")
def show_welcome_popup():
    st.write("""
        Welcome to the **BioData Quality Control & 3D Viewer** application!
        
        **Quick Instructions:**
        - Upload your CSV dataset using the sidebar menu.
        - Select column mappings and QC parameter thresholds.
        - Click **Run Pipeline** to compute thermodynamic properties and generate interactive 3D visualizations.
        
        **IMPORTANT - COLUMNS MEANING**
        1. **PDB Column**: The PDB entry for the complex, followed by the chain identifiers for the two subunits.
        2. **Mutation Column**: Column with the mutations. The format must be: The first character is the one letter amino acid code for the original residue, the second character is the chain identifier, the third to penultimate characters indicate the residue number, followed by the residue insertion code where applicable, and the final character indicates the mutant amino acid. Where multiple mutations are present, they are separated by commas.
        3. **Wild-Type Affinity Column**: The affinity of the wild-type form. It must be a numeric float.
        4. **Mutant Affinity Column**: The affinity of the mutant form. It must be a numeric float.
        5. **Temperature Column**: The temperature column of the experiment. You must select in the select box the scale used.
    """)
    
    if st.button("Get Started", type="primary", use_container_width=True):
        st.session_state['welcome_seen'] = True
        st.rerun()

# Trigger pop-up on first load
if 'welcome_seen' not in st.session_state:
    show_welcome_popup()


# --- SIDEBAR ---
st.sidebar.title("Data & Column Configuration")

# Section to upload a file (max 200 MB)
uploaded_file = st.sidebar.file_uploader("Upload Dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception:
        # Fallback to comma if auto-sniffing fails
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, sep=',')
    
    # Form to add the information needed for the execution
    with st.sidebar.form(key="pipeline_config_form"):
        # Column information
        st.info("Enter the column names for the needed columns.")

        col_pdb = st.text_input("PDB Column:")
        col_mut = st.text_input("Mutation Column:")
        col_aff_wt = st.text_input("Wild-Type Affinity Column (Numeric Float):")
        col_aff_mut = st.text_input("Mutant Affinity Column (Numeric Float):")
        col_temp = st.text_input("Temperature Column:")
        check_temp = st.selectbox("Temperature scale used:", options=["Kelvin (K)", "Celsius (C)", "Fahrenheit (F)"])
        
        st.divider()
        # QC information
        st.subheader("QC Parameters")
        z_thresh = st.slider("Robust Z-Score Cutoff:", 1.5, 5.0, 3.5, 0.1)
        iso_contam = st.slider("Isolation Forest Contamination:", 0.01, 0.15, 0.03, 0.01)
        
        mad_floor = st.slider(
            "MAD Floor (Min Variance Cutoff):",
            min_value=0.000001,
            max_value=0.1,
            value=0.000001,
            step=0.0005,
            format="scientific",
            help="Minimum allowable Median Absolute Deviation (MAD) value. Prevents division by zero and extreme Z-scores for groups with near-zero variance."
        )

        submit_button = st.form_submit_button(label="Run Pipeline", width='content')

    if submit_button:
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
                    contamination_rate=iso_contam
                )

                df_qc['PDB_ID'] = df_qc[col_pdb].astype(str).str[:4].str.upper()

                # Store all required column variables in session state
                st.session_state['df_qc'] = df_qc
                st.session_state['col_mut'] = col_mut
                st.session_state['col_aff_wt'] = col_aff_wt
                st.session_state['col_aff_mut'] = col_aff_mut

            st.success(f"Successfully processed {len(df_qc):,} records!")

        except KeyError as ke:
            st.error(f"Column error: Missing column {ke}. Please check your sidebar dropdown selections.")
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")


# --- DISPLAY DASHBOARD ---
if 'df_qc' in st.session_state:
    # Important variables
    df_qc = st.session_state['df_qc']
    col_mut = st.session_state['col_mut']
    col_aff_wt = st.session_state.get('col_aff_wt')
    col_aff_mut = st.session_state.get('col_aff_mut')

    tab1, tab2 = st.tabs(["Dataset Metrics & Table", "3D Structural Viewer"])

    # Processed dataset
    with tab1:
        st.subheader("Processed Dataset")
        st.dataframe(df_qc, width='stretch')
        
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
        st.header("3D Macromolecular Structure Viewer")

        available_pdbs = sorted(df_qc['PDB_ID'].dropna().unique())

        if available_pdbs:
            # PDB & Mutation Selectors
            c1, c2 = st.columns(2)
            with c1:
                selected_pdb = st.selectbox("1. Select PDB Complex:", available_pdbs)

            pdb_subset = df_qc[df_qc['PDB_ID'] == selected_pdb]
            available_muts = sorted(pdb_subset[col_mut].dropna().unique())

            with c2:
                selected_mutation = st.selectbox("2. Select Mutation:", available_muts)

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
            if parsed_muts:
                mut_df = pd.DataFrame(parsed_muts)
                st.dataframe(
                    mut_df[['chain', 'resnum', 'wt_name', 'mut_name', 'wt_class', 'mut_class', 'mw_change']],
                    hide_index=True,
                    width='content'
                )
            else:
                st.write("No standard mutation details could be parsed from string.")

            st.divider()

            # --- 3D VIEWER (LEFT) AND APPEARANCE CONTROLS (RIGHT) ---
            col_viewer, col_options = st.columns([3, 2])
            
            # Options
            with col_options:
                st.subheader("Appearance & Styling")
                
                selected_style = st.selectbox(
                    "Main representation:",
                    ["Cartoon", "Spheres", "Sticks", "Ribbon Trace"]
                )
                bg_color = st.selectbox("Canvas background:", ["White", "Dark"])

                color_scheme = st.selectbox(
                    "Protein color scheme:",
                    ["Chain ID", "Secondary Structure", "N-to-C Spectrum", "B-Factor"]
                )
                show_surface = st.checkbox("Overlay molecular surface", value=False)

                highlight_color = st.color_picker("Mutation highlight color:", "#FF007F")
                mut_repr = st.selectbox(
                    "Mutation site representation:",
                    ["Sticks & Spheres", "Spheres Only", "Sticks Only"]
                )

                st.markdown("---")
                show_neighbors = st.checkbox("Highlight 5Å radius neighbors", value=False)
                neighbor_radius = st.slider("Radius (Å):", 3.0, 10.0, 5.0, 0.5)
                neighbor_color = st.color_picker("Neighbor residue color:", "#00E5FF")

            # 3D viewer
            with col_viewer:
                with st.spinner(f"Fetching and rendering 3D structure for {selected_pdb}..."):
                    success, pdb_block_or_error = render_wt_structure_highlight(
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
                        width=650,
                        height=520
                    )
                
                if success:
                    st.caption(f"Fetched successfully from RCSB PDB API ({selected_pdb})")
                    st.download_button(
                        label=f"Download {selected_pdb} PDB file",
                        data=pdb_block_or_error,
                        file_name=f"{selected_pdb}.pdb",
                        mime="chemical/x-pdb",
                        width='content'
                    )
                else:
                    st.error(pdb_block_or_error)

        else:
            st.warning("No valid 4-character PDB IDs found in the dataset.")
else:
    if not uploaded_file:
        st.info("Please upload a CSV dataset in the sidebar to get started.")