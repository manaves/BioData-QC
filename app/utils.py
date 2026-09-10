import re
import py3Dmol
import pandas as pd
import urllib.request
from stmol import showmol
import plotly.express as px
import plotly.graph_objects as go

AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'class': 'Nonpolar', 'mw': 89.09},
    'R': {'name': 'Arginine', 'class': 'Basic (+)', 'mw': 174.20},
    'N': {'name': 'Asparagine', 'class': 'Polar', 'mw': 132.12},
    'D': {'name': 'Aspartic Acid', 'class': 'Acidic (-)', 'mw': 133.10},
    'C': {'name': 'Cysteine', 'class': 'Polar', 'mw': 121.16},
    'E': {'name': 'Glutamic Acid', 'class': 'Acidic (-)', 'mw': 147.13},
    'Q': {'name': 'Glutamine', 'class': 'Polar', 'mw': 146.15},
    'G': {'name': 'Glycine', 'class': 'Nonpolar', 'mw': 75.07},
    'H': {'name': 'Histidine', 'class': 'Basic (+)', 'mw': 155.16},
    'I': {'name': 'Isoleucine', 'class': 'Nonpolar', 'mw': 131.17},
    'L': {'name': 'Leucine', 'class': 'Nonpolar', 'mw': 131.17},
    'K': {'name': 'Lysine', 'class': 'Basic (+)', 'mw': 146.19},
    'M': {'name': 'Methionine', 'class': 'Nonpolar', 'mw': 149.21},
    'F': {'name': 'Phenylalanine', 'class': 'Aromatic', 'mw': 165.19},
    'P': {'name': 'Proline', 'class': 'Nonpolar', 'mw': 115.13},
    'S': {'name': 'Serine', 'class': 'Polar', 'mw': 105.09},
    'T': {'name': 'Threonine', 'class': 'Polar', 'mw': 119.12},
    'W': {'name': 'Tryptophan', 'class': 'Aromatic', 'mw': 204.23},
    'Y': {'name': 'Tyrosine', 'class': 'Aromatic', 'mw': 181.19},
    'V': {'name': 'Valine', 'class': 'Nonpolar', 'mw': 117.15}
}

def fetch_pdb_from_web(pdb_id: str):
    clean_id = str(pdb_id).strip().upper()[:4]
    rcsb_url = f"https://files.rcsb.org/download/{clean_id}.pdb"
    try:
        req = urllib.request.Request(rcsb_url, headers={'User-Agent': 'StreamlitBioApp/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode('utf-8'), f"RCSB PDB Web API ({clean_id})"
    except Exception as e:
        return None, f"Could not retrieve PDB '{clean_id}' from RCSB PDB API. Error: {str(e)}"

def parse_mutation_info(mutation_str: str):
    if not isinstance(mutation_str, str) or not mutation_str.strip():
        return []
    
    parsed = []
    mut_list = [m.strip() for m in re.split(r'[,;]+', mutation_str) if m.strip()]
    skempi_pattern = re.compile(r'^([A-Z])([A-Za-z0-9])(\d+)([A-Za-z])?([A-Z])$')
    
    for m in mut_list:
        match = skempi_pattern.match(m)
        if match:
            wt_code, chain, resnum, inscode, mut_code = match.groups()
            wt_info = AMINO_ACIDS.get(wt_code, {'name': wt_code, 'class': 'Unknown', 'mw': 0})
            mut_info = AMINO_ACIDS.get(mut_code, {'name': mut_code, 'class': 'Unknown', 'mw': 0})
            
            parsed.append({
                'raw': m,
                'chain': chain,
                'resnum': int(resnum),
                'inscode': inscode if inscode else '',
                'wt_code': wt_code,
                'mut_code': mut_code,
                'wt_name': wt_info['name'],
                'mut_name': mut_info['name'],
                'wt_class': wt_info['class'],
                'mut_class': mut_info['class'],
                'mw_change': round(mut_info['mw'] - wt_info['mw'], 2)
            })
    return parsed

def render_wt_structure_highlight(
    pdb_id: str, 
    mutation_str: str, 
    style_type="Cartoon", 
    color_scheme="Chain ID",
    highlight_color="#FF007F",
    mut_repr="Sticks & Spheres",
    bg_color="White",
    show_surface=False,
    show_neighbors=False,
    neighbor_radius=5.0,
    neighbor_color="#00E5FF",
    width=680, 
    height=520
):
    """
    Renders 3D protein structure with optional 5 Ångström neighborhood highlighting.
    """
    pdb_block, source = fetch_pdb_from_web(pdb_id)
    if not pdb_block:
        return False, source

    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_block, 'pdb')
        
        # Canvas background
        canvas_bg = '#ffffff' if bg_color == "White" else '#111111'
        view.setBackgroundColor(canvas_bg)

        # Main color scheme
        color_map = {
            "Chain ID": "chain",
            "Secondary Structure": "ssPyMOL",
            "N-to-C Spectrum": "spectrum",
            "B-Factor": "bfactor"
        }
        selected_color = color_map.get(color_scheme, "chain")

        # Global representation
        if style_type == "Cartoon":
            view.setStyle({}, {'cartoon': {'color': selected_color, 'opacity': 0.85}})
        elif style_type == "Spheres":
            view.setStyle({}, {'sphere': {'color': selected_color, 'scale': 0.5}})
        elif style_type == "Sticks":
            view.setStyle({}, {'stick': {'colorscheme': 'chainCarbon', 'radius': 0.2}})
        elif style_type == "Ribbon Trace":
            view.setStyle({}, {'line': {'color': selected_color, 'linewidth': 3}})

        if show_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.35, 'color': 'white' if bg_color == "Dark" else 'gray'})

        view.addStyle({'hetflag': True}, {'stick': {'radius': 0.15}})

        # Process mutation target & spatial neighborhood
        parsed_muts = parse_mutation_info(mutation_str)
        has_valid_target = False

        if parsed_muts:
            for m in parsed_muts:
                chain = m['chain']
                resnum = m['resnum']
                
                res_spec = {'chain': chain, 'resi': str(resnum)}
                if m['inscode']:
                    res_spec['icode'] = m['inscode']

                # --- NEIGHBORHOOD SELECTION ---
                if show_neighbors:
                    neighbor_spec = {
                        'within': {
                            'distance': neighbor_radius,
                            'sel': res_spec,
                            'byres': True  # Select whole residues, not just individual atoms
                        }
                    }
                    # Render neighbors as semi-transparent (sticks & spheres)
                    view.addStyle(neighbor_spec, {
                        'stick': {
                            'color': neighbor_color,
                            'radius': 0.25,
                            'opacity': 0.9
                        },
                        'sphere': {
                            'color': neighbor_color, 
                            'radius': 1.3,
                            'opacity': 0.85
                        }
                    })

                # --- MUTATED RESIDUE HIGHLIGHT (Overrides neighbor style) ---
                if mut_repr in ["Sticks & Spheres", "Sticks only"]:
                    view.addStyle(res_spec, {'stick': {'color': highlight_color, 'radius': 0.45}})
                if mut_repr in ["Sticks & Spheres", "Spheres only"]:
                    view.addStyle(res_spec, {'sphere': {'color': highlight_color, 'opacity': 0.85, 'radius': 1.3}})

                # Residue label
                label_bg = '#222222' if bg_color == "White" else '#ffffff'
                label_fg = '#ffffff' if bg_color == "White" else '#000000'
                
                label_txt = f"{m['wt_code']}({chain}:{resnum}) -> {m['mut_code']}"
                view.addLabel(
                    label_txt, 
                    {'fontSize': 12, 'fontColor': label_fg, 'backgroundColor': label_bg}, 
                    res_spec
                )
                
                if not has_valid_target:
                    # Zoom in closely to show local contacts if neighbors are enabled
                    zoom_spec = neighbor_spec if show_neighbors else res_spec  # pyright: ignore[reportPossiblyUnboundVariable]
                    view.zoomTo(zoom_spec)
                    has_valid_target = True

        if not has_valid_target:
            view.zoomTo()

        showmol(view, height=height, width=width)
        return True, pdb_block

    except Exception as e:
        return False, f"3D rendering error: {str(e)}"

def plot_z_score(df_qc: pd.DataFrame) -> go.Figure:
    """
    Generates a scatter plot of Robust Z-Score versus calculated ΔΔG values.

    Parameters
        df_qc : pd.DataFrame
            Dataframe containing processed QC metrics and thermodynamic data.
            Expected columns: 'z_score', 'ddG_kcal_mol', and optionally 'QC_Flag'.

    Returns
        plotly.graph_objects.Figure
            A Plotly scatter plot mapping z-scores against binding free energy changes.
    """
    fig = px.scatter(
        df_qc,
        x='z_score',
        y='ddG_kcal_mol',
        color='QC_Flag' if 'QC_Flag' in df_qc.columns else None,
        title="Robust Z-Score vs ΔΔG",
        labels={'z_score': 'Z-Score', 'ddG_kcal_mol': 'ΔΔG (kcal/mol)'}
    )
    return fig


def plot_iso_forest_outlier(df_qc: pd.DataFrame) -> go.Figure:
    """
    Generates a scatter plot of Isolation Forest anomaly scores versus ΔΔG.

    Parameters
        df_qc : pd.DataFrame
            Dataframe containing QC metrics and thermodynamic data.
            Expected columns: 'iso_forest_outlier', 'ddG_kcal_mol', and optionally 'QC_Flag'.

    Returns
        plotly.graph_objects.Figure
            A Plotly scatter plot showing multidimensional isolation forest scores vs ΔΔG.
    """
    fig = px.scatter(
        df_qc,
        x='iso_forest_outlier',
        y='ddG_kcal_mol',
        color='QC_Flag' if 'QC_Flag' in df_qc.columns else None,
        title="Isolation Forest Score vs ΔΔG",
        labels={'iso_forest_outlier': 'Isolation Forest Score', 'ddG_kcal_mol': 'ΔΔG (kcal/mol)'}
    )
    return fig


def plot_std_replicates(df_qc: pd.DataFrame) -> go.Figure:
    """
    Generates a scatter plot of experimental replicate standard deviations versus ΔΔG.

    Parameters
        df_qc : pd.DataFrame
            Dataframe containing quality control metrics and thermodynamic data.
            Expected columns: 'std_replicates', 'ddG_kcal_mol', and optionally 'QC_Flag'.

    Returns
        plotly.graph_objects.Figure
            A Plotly scatter plot evaluating experimental variance relative to ΔΔG.
    """
    fig = px.scatter(
        df_qc,
        x='std_replicates',
        y='ddG_kcal_mol',
        color='QC_Flag' if 'QC_Flag' in df_qc.columns else None,
        title="Standard Deviation (Replicates) vs ΔΔG",
        labels={'std_replicates': 'Std Dev', 'ddG_kcal_mol': 'ΔΔG (kcal/mol)'}
    )
    return fig


def plot_qc_flag(df_qc: pd.DataFrame) -> go.Figure:
    """
    Generates a histogram illustrating the overall distribution of QC flags.

    Parameters
        df_qc : pd.DataFrame
            Dataframe containing QC evaluation outputs.
            Expected columns: 'QC_Flag'.

    Returns
        plotly.graph_objects.Figure
            A Plotly histogram summarizing record counts grouped by QC pass/fail flags.
    """
    fig = px.histogram(
        df_qc,
        x='QC_Flag',
        color='QC_Flag',
        title="Quality Control Flag Distribution"
    )
    return fig


def plot_qc_reason(df_qc: pd.DataFrame) -> go.Figure:
    """
    Generates a horizontal bar chart summarizing primary reasons for QC flag exclusions.

    Parameters
        df_qc : pd.DataFrame
            Dataframe containing quality control metrics and exclusion reasons.
            Expected columns: 'QC_Reason', and optionally 'QC_Flag'.

    Returns
        plotly.graph_objects.Figure
            A Plotly horizontal histogram depicting primary quality failure causes.
    """
    if 'QC_Reason' in df_qc.columns:
        fig = px.histogram(
            df_qc,
            y='QC_Reason',
            color='QC_Flag' if 'QC_Flag' in df_qc.columns else None,
            title="Quality Control Exclusion Reasons",
            orientation='h'
        )
        return fig
    return px.scatter(title="QC_Reason column missing")


def plot_ddg_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plots the frequency distribution of binding free energy changes (ΔΔG) with a marginal rug plot.

    Includes a red reference line at ΔΔG = 0 kcal/mol separating stabilizing (< 0) 
        from destabilizing (> 0) mutations.

    Parameters
        df : pd.DataFrame
            Dataframe containing thermodynamic calculation results.
            Expected columns: 'ddG_kcal_mol'.

    Returns
        plotly.graph_objects.Figure
            A Plotly histogram with marginal rug distribution for ΔΔG values.
    """
    if 'ddG_kcal_mol' not in df.columns:
        return px.scatter(title="ddG_kcal_mol column missing")
        
    fig = px.histogram(
        df,
        x='ddG_kcal_mol',
        nbins=30,
        marginal='rug',
        title="Distribution of Binding Free Energy Changes (ΔΔG)",
        labels={'ddG_kcal_mol': 'ΔΔG (kcal/mol)'},
        color_discrete_sequence=['#6366F1']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral (0 kcal/mol)")
    return fig


def plot_affinity_scatter(df: pd.DataFrame, col_aff_wt: str, col_aff_mut: str) -> go.Figure:
    """
    Plots Wild-Type versus Mutant affinities on logarithmic scales with a y = x reference line.

    Points above the y = x line indicate decreased binding affinity upon mutation, 
    whereas points below indicate increased binding affinity.

    Parameters
        df : pd.DataFrame
            Dataframe containing experimental wild-type and mutant affinity measurements.
        col_aff_wt : str
            Column name for Wild-Type binding affinity (e.g., Kd in Molar).
        col_aff_mut : str
            Column name for Mutant binding affinity (e.g., Kd in Molar).

    Returns
        plotly.graph_objects.Figure
            A Plotly log-log scatter plot comparing WT vs Mutant affinities with an equivalence diagonal.
    """
    fig = px.scatter(
        df,
        x=col_aff_wt,
        y=col_aff_mut,
        hover_data=['PDB_ID'] if 'PDB_ID' in df.columns else None,
        title="Wild-Type vs. Mutant Affinity",
        labels={col_aff_wt: "Wild-Type Affinity (M)", col_aff_mut: "Mutant Affinity (M)"},
        log_x=True,
        log_y=True
    )
    
    min_val = min(df[col_aff_wt].min(), df[col_aff_mut].min())
    max_val = max(df[col_aff_wt].max(), df[col_aff_mut].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='No Change Line (y=x)',
        line=dict(color='gray', dash='dash')
    ))
    return fig


def plot_pdb_counts(df: pd.DataFrame) -> go.Figure:
    """
    Generates a bar chart showing the total number of mutation records per PDB complex.

    Parameters
        df : pd.DataFrame
            Dataframe containing dataset entries with macromolecular structure IDs.
            Expected columns: 'PDB_ID'.

    Returns
        plotly.graph_objects.Figure
            A Plotly bar chart depicting entry counts per PDB structure ID.
    """
    if 'PDB_ID' not in df.columns:
        return px.scatter(title="PDB_ID column missing")
        
    pdb_counts = df['PDB_ID'].value_counts().reset_index()
    pdb_counts.columns = ['PDB_ID', 'Count']
    
    fig = px.bar(
        pdb_counts,
        x='PDB_ID',
        y='Count',
        title="Mutations per PDB Structure",
        color='Count',
        color_continuous_scale='Viridis'
    )
    return fig


def plot_aa_transition_matrix(df: pd.DataFrame, col_mut: str) -> go.Figure:
    """
    Generates a 2D cross-tabulation heatmap of Wild-Type to Mutant amino acid substitutions.
    Parses single-letter residue codes from the first and last characters of the mutation string.

    Parameters:
        df : pd.DataFrame
            Dataframe containing mutation annotation strings (e.g., 'A38K' or 'AD35A').
        col_mut : str
            Column name containing mutation string representations.

    Returns
        plotly.graph_objects.Figure
            A Plotly heatmap showing frequency of residue conversions from WT to Mutant.
    """
    wt_aa = df[col_mut].astype(str).str[0]
    mut_aa = df[col_mut].astype(str).str[-1]
    
    ct = pd.crosstab(wt_aa, mut_aa)
    
    fig = px.imshow(
        ct,
        labels=dict(x="Mutant Amino Acid", y="Wild-Type Amino Acid", color="Count"),
        title="Amino Acid Substitution Matrix",
        color_continuous_scale="Blues",
        text_auto=True
    )
    return fig