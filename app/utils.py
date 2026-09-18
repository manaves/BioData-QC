import re
import html
import math
import py3Dmol
import datetime
import pandas as pd
import urllib.request
from typing import Any
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

def _pdb_atom_stats(pdb_block: str):
    """
    Scan ATOM/HETATM records for the ranges needed by gradient colorschemes.

    3Dmol does not infer gradient domains, so we compute them from the file:
    B-factor = columns 61-66, residue number = columns 23-26 (PDB format).
    Returns (bf_min, bf_max, resi_min, resi_max).
    """
    bfactors, resnums = [], []
    for line in pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            try:
                bfactors.append(float(line[60:66]))
            except ValueError:
                pass
        if line.startswith("ATOM") and len(line) >= 26:
            try:
                resnums.append(int(line[22:26]))
            except ValueError:
                pass
            
    if not bfactors:
        bf_min, bf_max = 0.0, 100.0
    else:
        bf_min, bf_max = min(bfactors), max(bfactors)
        if bf_min == bf_max:  # Avoid a zero-width gradient domain
            bf_min, bf_max = bf_min - 1.0, bf_max + 1.0
    if not resnums:
        resi_min, resi_max = 1, 100
    else:
        resi_min, resi_max = min(resnums), max(resnums)
        if resi_min == resi_max:
            resi_min, resi_max = resi_min - 1, resi_max + 1
    
    return bf_min, bf_max, resi_min, resi_max

def _build_scheme_spec(color_scheme: str, pdb_block: str):
    """
    Return the 3Dmol colorscheme spec for the scheme chosen in the UI.

    Gradient schemes use the object form {prop, gradient, min, max}, which has
    been supported by 3Dmol.js for many years ('b' = B-factor, 'resi' =
    residue number). 'chain' and 'ssJmol' are plain scheme-name strings.
    """
    if color_scheme == "B-Factor":
        bf_min, bf_max, _, _ = _pdb_atom_stats(pdb_block)
        return {'prop': 'b', 'gradient': 'roygb', 'min': bf_min, 'max': bf_max}
    if color_scheme == "N-to-C Spectrum":
        # 'resi' restarts on every chain, so each chain sweeps the gradient
        # from its own N-terminus to its own C-terminus.
        _, _, resi_min, resi_max = _pdb_atom_stats(pdb_block)
        return {'prop': 'resi', 'gradient': 'roygb', 'min': resi_min, 'max': resi_max}
    if color_scheme == "Secondary Structure":
        return 'ssJmol' #'ssPyMOL'
    return 'chain'

# In-process cache of successfully downloaded PDB blocks, so Streamlit reruns
# caused by viewer interactions do not re-download the same file. Bounded FIFO;
# failures are intentionally not cached so a transient error can be retried.
_PDB_CACHE: dict = {}
_PDB_CACHE_MAX = 32


def _cache_pdb_block(clean_id: str, block: str) -> None:
    if clean_id not in _PDB_CACHE and len(_PDB_CACHE) >= _PDB_CACHE_MAX:
        _PDB_CACHE.pop(next(iter(_PDB_CACHE)))
    _PDB_CACHE[clean_id] = block


def fetch_pdb_from_web(pdb_id: str):
    """
    Fetch PDB structure from RCSB DB, reusing an in-process cache when possible.
    """
    clean_id = str(pdb_id).strip().upper()[:4]
    cached = _PDB_CACHE.get(clean_id)
    if cached is not None:
        return cached, f"RCSB PDB Web API ({clean_id}, cached)"
    rcsb_url = f"https://files.rcsb.org/download/{clean_id}.pdb"
    try:
        req = urllib.request.Request(rcsb_url, headers={"User-Agent": "StreamlitBioApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            block = response.read().decode("utf-8")
        _cache_pdb_block(clean_id, block)
        return block, f"RCSB PDB Web API ({clean_id})"
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

def _residue_center(pdb_block: str, chain, resnum, icode: str = ""):
    """
    Return the (x, y, z) centroid of the atoms of one residue in a PDB block.

    Coordinates come from columns 31-54 of ATOM/HETATM records. Returns None
    when the residue is not found or has no parseable coordinates, so callers
    can skip the flash animation instead of guessing a position.
    """
    xs, ys, zs = [], [], []
    for line in pdb_block.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        if (line[21].strip() or "A") != str(chain).strip():
            continue
        try:
            if int(line[22:26]) != int(resnum):
                continue
        except (TypeError, ValueError):
            continue
        if line[26].strip() != (icode or ""):
            continue
        try:
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
        except ValueError:
            continue

    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)


def _neighborhood_spec(res_spec: dict, radius: float) -> dict:
    """Selection spec for residues within `radius` Å of `res_spec` (by residue)."""
    return {
        'within': {
            'distance': radius,
            'sel': res_spec,
            'byres': True
        }
    }


def _add_flash_animation(
    view,
    pdb_block: str,
    res_spec: dict,
    center=None,
    flash_color: str = "#FFD400",
    flash_radius: float = 1.8,
    flashes: int = 6,
    interval_ms: int = 320
):
    """
    Inject JavaScript that pulses a translucent sphere over `res_spec`.

    3Dmol.js has no built-in flash method (checked against 2.5.5), so the
    animation alternates addSphere/removeShape with setInterval and ends with
    the sphere removed. It runs after the model styles are applied because the
    code is prepended to `view.endjs`, i.e. inside the load promise.

    `center` may be supplied by the caller to avoid re-scanning the PDB block;
    when omitted it is derived from `res_spec`. If the residue cannot be found,
    nothing is injected.
    """
    if center is None:
        center = _residue_center(
            pdb_block,
            res_spec.get("chain"),
            res_spec.get("resi"),
            res_spec.get("icode", ""),
        )
    if center is None:
        return

    cx, cy, cz = center
    js = (
        "(function() {\n"
        f"  var center = {{x: {cx}, y: {cy}, z: {cz}}};\n"
        f"  var opts = {{center: center, radius: {flash_radius}, "
        f"color: '{flash_color}', opacity: 0.55}};\n"
        "  var handle = null, ticks = 0;\n"
        "  var timer = setInterval(function() {\n"
        "    if (handle) { viewer_UNIQUEID.removeShape(handle); handle = null; }\n"
        "    else { handle = viewer_UNIQUEID.addSphere(opts); }\n"
        "    viewer_UNIQUEID.render();\n"
        "    ticks += 1;\n"
        f"    if (ticks >= {flashes}) {{\n"
        "      clearInterval(timer);\n"
        "      if (handle) { viewer_UNIQUEID.removeShape(handle); }\n"
        "      viewer_UNIQUEID.render();\n"
        "    }\n"
        f"  }}, {interval_ms});\n"
        "})();\n"
    )
    view.endjs = js + view.endjs


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
    width: int | str | None = 680, 
    height=520,
    focus_residue=None
):
    """
    Renders 3D protein structure with optional 5 Ångström neighborhood highlighting.

    Parameters
        width : int | str | None
            Canvas width. An int fixes the width in pixels; None (or a CSS
            string such as "100%") makes the viewer fill its parent column.
        focus_residue : dict, optional
            Residue to center/zoom and flash, e.g. one entry from
            parse_mutation_info ({'chain', 'resnum', 'inscode', ...}). When
            given, it overrides the default zoom-to-first-mutation behaviour.
    """
    pdb_block, source = fetch_pdb_from_web(pdb_id)
    if not pdb_block:
        return False, source

    # py3Dmol turns int widths into "Npx" and passes strings through verbatim,
    # so "100%" lets the inner viewer div track the column width. The library
    # source is untyped (no py.typed), so Any avoids false int-only errors.
    view_width: Any = "100%" if width is None else width
    try:
        view = py3Dmol.view(width=view_width, height=height)
        view.addModel(pdb_block, 'pdb')
        
        # Canvas background
        canvas_bg = '#ffffff' if bg_color == "White" else '#111111'
        view.setBackgroundColor(canvas_bg)

        scheme_spec = _build_scheme_spec(color_scheme, pdb_block)

        if style_type == "Cartoon":
            view.setStyle({}, {'cartoon': {'colorscheme': scheme_spec, 'opacity': 0.85}})
        elif style_type == "Spheres":
            view.setStyle({}, {'sphere': {'colorscheme': scheme_spec, 'scale': 0.5}})
        elif style_type == "Sticks":
            view.setStyle({}, {'stick': {'colorscheme': scheme_spec, 'radius': 0.2}})
        elif style_type == "Ribbon Trace":
            view.setStyle({}, {'line': {'colorscheme': scheme_spec, 'linewidth': 3}})

        if show_surface:
            surface_color = '#909090' if bg_color == "White" else '#3d3d3d'
            view.addSurface(py3Dmol.VDW, {'opacity': 0.45, 'color': surface_color})

        view.addStyle({'hetflag': True}, {'stick': {'radius': 0.15}})

        # Residue clicked in the Mutation Properties table, if any. Only focus
        # it when the residue exists in this structure
        focus_spec = None
        focus_center = None
        if focus_residue and focus_residue.get('chain') is not None and focus_residue.get('resnum') is not None:
            candidate = {'chain': focus_residue['chain'], 'resi': str(focus_residue['resnum'])}
            if focus_residue.get('inscode'):
                candidate['icode'] = focus_residue['inscode']
            focus_center = _residue_center(
                pdb_block,
                candidate.get('chain'),
                candidate.get('resi'),
                candidate.get('icode', ''),
            )
            if focus_center is not None:
                focus_spec = candidate

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
                neighbor_spec = None
                if show_neighbors:
                    neighbor_spec = _neighborhood_spec(res_spec, neighbor_radius)
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

                # --- MUTATED RESIDUE HIGHLIGHT ---
                mut_repr_l = str(mut_repr).lower()
                if mut_repr_l in ("sticks & spheres", "sticks only"):
                    view.addStyle(res_spec, {'stick': {'color': highlight_color, 'radius': 0.45}})
                if mut_repr_l in ("sticks & spheres", "spheres only"):
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
                
                # Default framing: only when no table row asked for a focus.
                if not has_valid_target and focus_spec is None:
                    zoom_spec = neighbor_spec if (show_neighbors and neighbor_spec is not None) else res_spec
                    view.zoomTo(zoom_spec)
                    has_valid_target = True

        # --- TABLE-DRIVEN FOCUS: center, zoom and flash the clicked residue ---
        if focus_spec is not None:
            view.addStyle(focus_spec, {'stick': {'color': '#FFD400', 'radius': 0.5}})
            zoom_spec = _neighborhood_spec(focus_spec, neighbor_radius) if show_neighbors else focus_spec
            view.zoomTo(zoom_spec)
            _add_flash_animation(view, pdb_block, focus_spec, center=focus_center)
            has_valid_target = True

        if not has_valid_target:
            view.zoomTo()

        # Only a pixel width is passed to the iframe; None stretches it to the
        # column. The inner viewer div (view_width) carries the CSS sizing.
        frame_width: Any = width if isinstance(width, int) else None
        showmol(view, height=height, width=frame_width)
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
    
    pairs = []
    for s in df[col_mut].dropna().astype(str):
        for m in parse_mutation_info(s):
            pairs.append((m['wt_code'], m['mut_code']))

    if not pairs:
        return px.scatter(title="No parseable mutations found")

    wt_series = pd.Series([p[0] for p in pairs], name="Wild-Type Amino Acid")
    mut_series = pd.Series([p[1] for p in pairs], name="Mutant Amino Acid")
    ct = pd.crosstab(wt_series, mut_series)

    return px.imshow(
        ct,
        labels=dict(x="Mutant Amino Acid", y="Wild-Type Amino Acid", color="Count"),
        title="Amino Acid Substitution Matrix",
        color_continuous_scale="Blues",
        text_auto=True
    )


# ---------------------------------------------------------------------------
# Executive summary report export engine
# ---------------------------------------------------------------------------

def pdb_block_metadata(pdb_block: str) -> dict:
    """
    Summarize a PDB block for the report's structure metadata section.

    Returns chain identifiers, residue/atom counts and the non-water ligand
    residue names found in HETATM records. Missing/empty blocks yield zeros so
    the report can still be generated when the structure was not fetched.
    """
    if not pdb_block:
        return {"chains": [], "residue_count": 0, "atom_count": 0,
                "hetatm_count": 0, "ligands": []}

    chains, residues, ligands = set(), set(), set()
    atom_count = hetatm_count = 0
    for line in pdb_block.splitlines():
        record = line[:6].strip()
        if record == "ATOM" and len(line) >= 27:
            atom_count += 1
            chain = line[21].strip() or "A"
            chains.add(chain)
            residues.add((chain, line[22:27].strip()))
        elif record == "HETATM":
            hetatm_count += 1
            ligand = line[17:20].strip()
            if ligand and ligand not in {"HOH", "DOD", "WAT"}:
                ligands.add(ligand)

    return {
        "chains": sorted(chains),
        "residue_count": len(residues),
        "atom_count": atom_count,
        "hetatm_count": hetatm_count,
        "ligands": sorted(ligands),
    }


def _report_escape(value) -> str:
    return html.escape("" if value is None else str(value))


def _report_number(value, digits: int = 3) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(num) or math.isinf(num):
        return "—"
    return f"{num:,.{digits}f}"


def _report_definition_table(items) -> str:
    rows = "".join(
        f'<tr><th scope="row">{_report_escape(label)}</th>'
        f'<td>{_report_escape(value) if value not in (None, "") else "—"}</td></tr>'
        for label, value in items
    )
    return f'<table class="report-kv"><tbody>{rows}</tbody></table>'


def _report_dataframe_table(df: Any, max_rows=None) -> str:
    if df is None or df.empty:
        return '<p class="report-empty">No data available.</p>'
    view = df.head(max_rows) if max_rows else df
    table = view.to_html(
        index=False,
        border=0,
        classes="report-table",
        escape=True,
        na_rep="—",
        float_format=lambda x: f"{x:,.3f}",
    )
    return f'<div class="report-table-wrap">{table}</div>'


def _report_format_affinity(df: Any, cols) -> Any:
    """Render molar affinity columns in scientific notation (3-decimal floats show as 0.000)."""
    df = df.copy()
    for col in cols:
        if col and col in df.columns:
            numeric: Any = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.map(lambda v: "—" if pd.isna(v) else f"{v:.2e}")
    return df


def _report_figure(fig, include_plotlyjs: bool = False) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displayModeBar": False, "responsive": True},
    )


def _report_ddg_stats(series: Any) -> dict:
    numeric: Any = pd.to_numeric(series, errors="coerce")
    values: Any = numeric.dropna()
    if values.empty:
        return {}
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return {
        "Records": f"{len(values):,}",
        "Mean ΔΔG (kcal/mol)": _report_number(values.mean()),
        "Median ΔΔG (kcal/mol)": _report_number(values.median()),
        "Std dev (kcal/mol)": _report_number(std),
        "Min ΔΔG (kcal/mol)": _report_number(values.min()),
        "Max ΔΔG (kcal/mol)": _report_number(values.max()),
    }


_REPORT_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    margin: 0; padding: 0; background: #f4f6fb; color: #1f2937;
}
.report { max-width: 1080px; margin: 0 auto; padding: 2.5rem 1.5rem 3rem; }
.report-header {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    color: #fff; border-radius: 14px; padding: 1.75rem 2rem; margin-bottom: 1.75rem;
}
.report-header h1 { margin: 0 0 0.35rem; font-size: 1.65rem; }
.report-header .subtitle { margin: 0; font-size: 1rem; opacity: 0.92; }
.report-header .timestamp { margin: 0.6rem 0 0; font-size: 0.8rem; opacity: 0.8; }
section.report-section {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
}
section.report-section > h2 {
    margin: 0 0 0.9rem; font-size: 1.1rem; color: #312e81;
    border-bottom: 2px solid #eef2ff; padding-bottom: 0.5rem;
}
.report-kv { border-collapse: collapse; width: 100%; }
.report-kv th, .report-kv td {
    text-align: left; padding: 0.42rem 0.6rem; border-bottom: 1px solid #f1f3f9;
    font-size: 0.88rem; vertical-align: top;
}
.report-kv th { width: 38%; color: #4b5563; font-weight: 600; }
.report-kv td { color: #111827; }
.report-table-wrap { width: 100%; max-width: 100%; overflow-x: auto; }
table.report-table {
    border-collapse: collapse; width: 100%; max-width: 100%;
    table-layout: fixed; font-size: 0.8rem;
}
table.report-table thead th {
    background: #eef2ff; color: #312e81; text-align: left;
    padding: 0.45rem 0.5rem; border-bottom: 2px solid #c7d2fe;
    white-space: normal; overflow-wrap: anywhere; word-break: break-word;
    vertical-align: bottom;
}
table.report-table tbody td {
    padding: 0.4rem 0.5rem; border-bottom: 1px solid #f1f3f9;
    overflow-wrap: anywhere; word-break: break-word;
    vertical-align: top;
}
table.report-table tbody tr:nth-child(even) { background: #fafbff; }
.report-empty { color: #6b7280; font-size: 0.88rem; }
.report-figure { margin: 0 0 1.1rem; }
.report-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
@media (max-width: 820px) { .report-grid { grid-template-columns: 1fr; } }
.report-footer {
    font-size: 0.75rem; color: #6b7280; text-align: center;
    border-top: 1px solid #e5e7eb; padding-top: 1rem;
}
"""


def build_executive_summary_report(
    df_qc: pd.DataFrame,
    pdb_id,
    mutation,
    col_mut: str,
    col_aff_wt: str | None = None,
    col_aff_mut: str | None = None,
    col_temp: str | None = None,
    pdb_block: str | None = None,
    pdb_source: str | None = None,
    generated_at=None,
) -> str:
    """
    Bundle the current PDB metadata, ΔΔG metrics, mutation tables and key
    plots into a single self-contained HTML executive summary.

    The report is scoped to the PDB complex currently selected in the 3D
    viewer (`pdb_id`), while still including dataset-wide QC context. It has no
    external assets: Plotly.js is embedded once so the downloaded file renders
    offline.

    Parameters
        df_qc : pd.DataFrame
            Processed + QC dataframe stored in session state.
        pdb_id : str
            4-character PDB code of the selected complex.
        mutation : str
            Currently selected mutation string.
        col_mut : str
            Name of the mutation column.
        col_aff_wt, col_aff_mut, col_temp : str, optional
            Selected affinity/temperature column names, included when present.
        pdb_block : str, optional
            Raw PDB text used for structure metadata (chains, atoms, ligands).
        pdb_source : str, optional
            Human-readable provenance of the structure.
        generated_at : datetime, optional
            Report timestamp; defaults to the current UTC time.

    Returns
        str
            A complete HTML document ready for `st.download_button`.
    """
    if generated_at is None:
        generated_at = datetime.datetime.now(datetime.timezone.utc)
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M UTC")
    complex_label = str(pdb_id)

    df: Any = df_qc.copy()
    if "PDB_ID" in df.columns:
        complex_df: Any = df[df["PDB_ID"].astype(str) == str(pdb_id)]
    else:
        complex_df = df
    if complex_df.empty:
        complex_df = df

    mut_rows: Any = pd.DataFrame()
    if col_mut and col_mut in complex_df.columns:
        mut_rows = complex_df[complex_df[col_mut].astype(str) == str(mutation)]
    mut_row = mut_rows.iloc[0] if not mut_rows.empty else None

    def row_value(column):
        if mut_row is None or not column or column not in mut_row:
            return None
        return mut_row[column]

    # --- Structure metadata ---
    structure = pdb_block_metadata(pdb_block) if pdb_block else None
    metadata_items = [
        ("PDB complex", complex_label),
        ("Selected mutation", mutation),
        ("Structure source", pdb_source),
        ("ΔΔG (kcal/mol)", _report_number(row_value("ddG_kcal_mol")) if mut_row is not None else None),
        ("QC flag", row_value("QC_Flag")),
        ("QC reason", row_value("QC_Reason")),
        ("Robust Z-score", _report_number(row_value("z_score")) if mut_row is not None else None),
        ("Replicate std dev (kcal/mol)", _report_number(row_value("std_replicates")) if mut_row is not None else None),
        ("Wild-type affinity", row_value(col_aff_wt)),
        ("Mutant affinity", row_value(col_aff_mut)),
        ("Temperature (raw)", row_value(col_temp)),
        ("Temperature (K)", _report_number(row_value("Temp_K"), 2) if mut_row is not None else None),
    ]
    if structure is not None:
        metadata_items.extend([
            ("Chains", ", ".join(structure["chains"]) or None),
            ("Residues", f"{structure['residue_count']:,}"),
            ("Atoms (ATOM records)", f"{structure['atom_count']:,}"),
            ("Ligands (HETATM)", ", ".join(structure["ligands"]) or "none"),
        ])

    # --- ΔΔG metrics ---
    all_stats = _report_ddg_stats(df["ddG_kcal_mol"]) if "ddG_kcal_mol" in df.columns else {}
    complex_stats = (
        _report_ddg_stats(complex_df["ddG_kcal_mol"])
        if "ddG_kcal_mol" in complex_df.columns else {}
    )
    metric_names = list(all_stats.keys()) or list(complex_stats.keys())
    metrics_rows = "".join(
        f"<tr><th scope=\"row\">{_report_escape(name)}</th>"
        f"<td>{_report_escape(all_stats.get(name, '—'))}</td>"
        f"<td>{_report_escape(complex_stats.get(name, '—'))}</td></tr>"
        for name in metric_names
    )
    metrics_table = (
        '<table class="report-table"><thead><tr><th>Metric</th>'
        '<th>All records</th><th>Selected complex</th></tr></thead>'
        f'<tbody>{metrics_rows}</tbody></table>'
        if metrics_rows else '<p class="report-empty">No ΔΔG metrics available.</p>'
    )

    if "QC_Flag" in df.columns:
        flag_counts = (
            df["QC_Flag"].value_counts().rename_axis("QC flag").reset_index(name="Records")
        )
        qc_breakdown = _report_dataframe_table(flag_counts)
    else:
        qc_breakdown = '<p class="report-empty">No QC flags available.</p>'

    # --- Mutation tables ---
    complex_cols = [c for c in [
        col_mut, col_aff_wt, col_aff_mut, "Temperature", "Temp_K",
        "ddG_kcal_mol", "z_score", "std_replicates", "QC_Flag", "QC_Reason",
    ] if c and c in complex_df.columns]
    complex_mut_table = _report_dataframe_table(
        _report_format_affinity(
            complex_df[complex_cols].reset_index(drop=True),
            [col_aff_wt, col_aff_mut],
        )
    ) if complex_cols else '<p class="report-empty">No mutation columns available.</p>'

    parsed = pd.DataFrame(parse_mutation_info(str(mutation)))
    parsed_cols = [c for c in [
        "chain", "resnum", "wt_name", "mut_name", "wt_class",
        "mut_class", "mw_change",
    ] if c in parsed.columns]
    parsed_table = _report_dataframe_table(parsed[parsed_cols]) \
        if parsed_cols else '<p class="report-empty">No standard mutation details could be parsed.</p>'

    # --- Key plots ---
    figures = []
    if "ddG_kcal_mol" in complex_df.columns:
        figures.append((
            "ΔΔG distribution (selected complex)",
            plot_ddg_distribution(complex_df),
        ))
    if "z_score" in complex_df.columns:
        figures.append((
            "Robust Z-score vs ΔΔG (selected complex)",
            plot_z_score(complex_df),
        ))
    if "iso_forest_outlier" in complex_df.columns:
        figures.append((
            "Isolation Forest vs ΔΔG (selected complex)",
            plot_iso_forest_outlier(complex_df),
        ))
    if "QC_Flag" in df.columns:
        figures.append(("QC flag distribution (dataset)", plot_qc_flag(df)))

    figure_html_parts = []
    for index, (title, fig) in enumerate(figures):
        figure_html_parts.append(
            '<div class="report-figure">'
            f'<h3>{_report_escape(title)}</h3>'
            f'{_report_figure(fig, include_plotlyjs=(index == 0))}'
            "</div>"
        )
    figures_html = "".join(figure_html_parts) or \
        '<p class="report-empty">No plots available.</p>'

    report_title = f"BioData-QC Executive Summary — {_report_escape(complex_label)}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{report_title}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<div class="report">
    <header class="report-header">
        <h1>BioData-QC Executive Summary</h1>
        <p class="subtitle">Complex {_report_escape(complex_label)} · Mutation {_report_escape(mutation)}</p>
        <p class="timestamp">Generated {_report_escape(timestamp)}</p>
    </header>

    <section class="report-section">
        <h2>Structure &amp; Assay Metadata</h2>
        {_report_definition_table(metadata_items)}
    </section>

    <section class="report-section">
        <h2>Calculated ΔΔG Metrics</h2>
        {metrics_table}
        <h3>QC Flag Breakdown</h3>
        {qc_breakdown}
    </section>

    <section class="report-section">
        <h2>Mutation Tables</h2>
        <h3>All mutations in {_report_escape(complex_label)}</h3>
        {complex_mut_table}
        <h3>Selected mutation events</h3>
        {parsed_table}
    </section>

    <section class="report-section">
        <h2>Key Plots</h2>
        {figures_html}
    </section>

    <footer class="report-footer">
        Generated by BioData-QC · Quality control methodology based on the
        SKEMPI 2.0 benchmark (Jankauskaitė et al., 2019).
    </footer>
</div>
</body>
</html>"""