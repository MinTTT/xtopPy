"""
xTop Algorithm for Proteome Data Analysis
Converted from MATLAB to Python

This function loads one or more files with peptide precursor intensities,
and generates one or more files with protein data, including the protein
intensity (TopPepN, TopPepAll and/or xTop), peptide ranking and detection
efficiencies, as well as other useful quantities.

Original MATLAB implementation: https://gitlab.com/mm87/xtop
Python conversion: 20250920
@author: Pan Chu
@data: 2025-09-20
@mail: pan_chu@outlook.com
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

# from scipy.optimize import lsq_linear
from datetime import datetime
import warnings
from typing import List, Union, Dict, Any, Optional, Tuple
from numba import jit, prange

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Falling back to sequential processing.")

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Global settings (similar to MATLAB's global variables)
rank_settings = {
    'rankFiles': 'all',  # 'individual' or 'all'
    'filterThres': 0.5,         # Filter strength (0-1)
    'export_ranking': False     # Export peptide ranking files
}

top_pep_settings = {
    'TopPepVal': [],           # TopPepN values to compute (e.g., [1, 2, 3] or [1, np.inf])
    'export_Npep': False       # Export number of peptides detected
}

x_top_settings = {
    'xTopPepVal': [1],                    # This array sets the overall normalization of the xTop intensities, exp features after xTop v2.0
    'thres_xTop': 300,                    # Intensity threshold
    'N_c_thres': 3,                       # Minimum samples for peptide
    's2_min': 0.01,                       # Variance regularization
    'absolute_filter_ratio': 0.1,         # Outlier filter ratio
    'error_type': 'relative (std-log)',   # Error type for output
    'plot_xTop_convergence': False,       # Debug plots
    'accuracy_max': 1e-10,                # Convergence criterion
    'N_step_max': 1000,                   # Maximum optimization steps
    'N_optimizations': 8,                # Number of optimization runs
    'rng_seed': 'shuffle',                # Random seed
    'export_level': 'full',               # 'basic' or 'full' output
    'parallelize_flag': True,             # Use parallel processing
    'max_parallel_workers': -1,           # Max workers (-1 = all cores)
    'backend': 'loky'                # 'threading' or 'loky' (multiprocessing)
}

verbose = 1  # Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)

def xTop(source_filename_arg: Union[str, List[str]] = None,
         verbose=verbose, file_output: bool = False) -> Tuple[Any, Any]:
    """
    This function loads one or more files with peptide precursor intensities,
    and generates one or more files with protein data.
    
    Args:
        source_filename_arg: Filenames (relative or absolute paths) to the sources with peptide data.
                           If None, uses default test files.
    
    Returns:
        
    """
    global rank_settings, top_pep_settings, x_top_settings

    verbose = verbose  
    # Default source files if none provided
    if source_filename_arg is not None:
        if isinstance(source_filename_arg, str):
            source_filename = [source_filename_arg]
        elif isinstance(source_filename_arg, pd.DataFrame):
            source_filename = [source_filename_arg]
        else:
            source_filename = list(source_filename_arg)

    else:
        source_filename = ['test_file1.csv', 'test_file2.csv']
    
    # Check if files exist (skip check for DataFrames)
    for f in source_filename:
        if isinstance(f, pd.DataFrame):
            continue
        if not os.path.exists(f):
            raise FileNotFoundError(f"Could not locate {f}")
    
    # Create output folder
    if file_output:
        output_folder_name = f"xTop_output_{datetime.now().strftime('%y-%m-%d-%H%M')}"
        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)
        elif verbose:
            print("** Warning: destination folder already exists. **")
            print("** Content will be overwritten.                **\n")
    else:
        output_folder_name = ''
        if verbose:
            print("** Warning: file_output is set to False. No output files will be created. **\n")

    
    # Create output filenames
    if file_output:
        fname_output = []
        for k, f in enumerate(source_filename):
            # Strip extension
            if isinstance(f, pd.DataFrame):
                fname_output.append(os.path.join(output_folder_name, f"[DataFrame-{k+1}]"))
                continue
            fname_base = os.path.splitext(os.path.basename(f))[0]
            fname_output.append(os.path.join(output_folder_name, f"[{fname_base}]"))
    else:
        fname_output = [None for _ in source_filename]
    
    # Print settings
    if verbose:
        print_settings(source_filename)
    
    # Main processing
    if rank_settings['rankFiles'] == 'all':
        if verbose:
            print("\n+ GLOBAL PEPTIDE RANKING")
        
        pep_data = rank_peptide_levels(source_filename)
        pep_data = remove_problematic_peptides(pep_data, [])  # Empty list for now
        
        for k in range(len(source_filename)):
            results = export_counts_from_pep(pep_data, source_filename[k], 
                                   fname_output[k] if fname_output else None)
        
        if rank_settings['export_ranking']:
            fname_pep_ranking = os.path.join(output_folder_name, "[GLOBAL] PepRank.csv") if file_output else None
            pep_rank_df = export_pep_ranking(pep_data, fname_pep_ranking)
            results['Peptide_Ranking'] = {'Peptide_Ranking': pep_rank_df}
    
    elif rank_settings['rankFiles'] == 'individual':
        pep_data = []
        results = []
        for k in range(len(source_filename)):
            if verbose:
                if isinstance(source_filename[k], pd.DataFrame):
                    print(f"\n\n:::::::: SOURCE: DataFrame-{k+1} (shape: {source_filename[k].shape})")
                else:
                    print(f"\n\n:::::::: SOURCE: {source_filename[k]}")
                print("\n+ + + Peptide ranking")
            
            current_pep_data = rank_peptide_levels([source_filename[k]])
            current_pep_data = remove_problematic_peptides(current_pep_data, [])
            pep_data.append(current_pep_data)

            result_temp = export_counts_from_pep(current_pep_data, source_filename[k], fname_output[k] if fname_output else None)
            

            if rank_settings['export_ranking']:
                fname_pep_ranking = f"{fname_output[k]} PepRank.csv" if file_output else None
                pep_rank_df = export_pep_ranking(current_pep_data, fname_pep_ranking)
                result_temp['Peptide_Ranking'] = {'Peptide_Ranking': pep_rank_df}
            results.append(result_temp)

    else:
        raise ValueError("rankFiles must be either 'all' or 'individual'")
    
    return pep_data, results

def print_settings(source_files: List[str]):
    """Print the current settings to console."""
    global rank_settings, top_pep_settings, x_top_settings, verbose
    
    print("\n* * * Main Settings * *")
    print("* Input files:")
    for i, f in enumerate(source_files):
        if isinstance(f, pd.DataFrame):
            print(f"*     DataFrame-{i+1} (shape: {f.shape})")
        else:
            print(f"*     {f}")
    
    if len(source_files) > 1:
        if rank_settings['rankFiles'] == 'all':
            print("*\n* Ranking will be performed on all data sources simultaneously.")
        else:
            print("*\n* Ranking will be performed on each data source individually.")
    
    if rank_settings['filterThres'] == 0:
        print("*   Ranking mode: no filtering.")
    else:
        print(f"*   Ranking mode: filtered (threshold: {rank_settings['filterThres']:.3f})")
    
    print(f"*   Export ranking = {str(rank_settings['export_ranking']).lower()}")
    
    if top_pep_settings['TopPepVal']:
        print("*\n* TopPepN settings\n*   Intensities to be exported:")
        for val in top_pep_settings['TopPepVal']:
            if not np.isinf(val):
                print(f"*     TopPep{int(val)}")
            else:
                print("*     TopPepAll")
        print(f"*   Export Npep = {str(top_pep_settings['export_Npep']).lower()}")
    
    print("*\n* xTop settings.")
    print(f"*            thres_xTop: {x_top_settings['thres_xTop']}")
    print(f"*             N_c_thres: {x_top_settings['N_c_thres']}")
    print(f"*                s2_min: {x_top_settings['s2_min']}")
    print(f"* absolute_filter_ratio: {x_top_settings['absolute_filter_ratio']}")
    print(f"*          accuracy_max: {x_top_settings['accuracy_max']:e}")
    print(f"*            N_step_max: {x_top_settings['N_step_max']}")
    print(f"*       N_optimizations: {x_top_settings['N_optimizations']}")
    
    if x_top_settings['export_level'] == 'basic':
        print("*\n* Only xTop intensities will be exported.")
    else:
        print("*\n* All xTop quantities will be exported.")
    
    print(f"* Random number generator seed: {x_top_settings['rng_seed']}")
    
    if x_top_settings['parallelize_flag']:
        if JOBLIB_AVAILABLE:
            backend_str = f"using joblib with {x_top_settings['backend']} backend"
            if x_top_settings['max_parallel_workers'] == -1:
                print(f"* The xTop calculation will be parallelized {backend_str} (all cores).")
            elif x_top_settings['max_parallel_workers'] > 0:
                print(f"* The xTop calculation will be parallelized {backend_str} (max {x_top_settings['max_parallel_workers']} workers).")
            else:
                print("* The xTop calculation will not be parallelized.")
        else:
            print("* The xTop calculation will be sequential (joblib not available).")
    else:
        print("* The xTop calculation will not be parallelized.")

    print("* * * * * * * * * * * * * * * * * * * *")

# Numba-compiled helper functions for fast peptide ranking
@jit(nopython=True)
def _calculate_peptide_stats_numba(intensities: np.ndarray, peptide_indices: np.ndarray, n_unique_peptides: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast calculation of peptide statistics using numba compilation.
    
    Args:
        intensities: 2D array of intensities (n_peptides x n_samples)
        peptide_indices: Array mapping each peptide to its unique index
        n_unique_peptides: Number of unique peptides
    
    Returns:
        Tuple of (sum_intensities, detection_counts, max_intensities)
    """
    n_peptides, n_samples = intensities.shape
    
    pep_int_sum = np.zeros(n_unique_peptides)
    pep_n = np.zeros(n_unique_peptides)
    pep_max = np.zeros(n_unique_peptides)
    
    for pi in range(n_peptides):
        pu = peptide_indices[pi]
        row_sum = 0.0
        row_count = 0
        row_max = 0.0
        
        for si in range(n_samples):
            intensity = intensities[pi, si]
            row_sum += intensity
            if intensity > 0:
                row_count += 1
            if intensity > row_max:
                row_max = intensity
        
        pep_int_sum[pu] += row_sum
        pep_n[pu] += row_count
        if row_max > pep_max[pu]:
            pep_max[pu] = row_max
    
    return pep_int_sum, pep_n, pep_max

@jit(nopython=True)
def _rank_peptides_for_protein_numba(pep_indices: np.ndarray, pep_int_sum: np.ndarray, pep_n: np.ndarray, 
                                     filter_thres: float) -> Tuple[np.ndarray, int]:
    """
    Fast peptide ranking for a single protein using numba compilation.
    
    Args:
        pep_indices: Indices of peptides belonging to this protein
        pep_int_sum: Total intensities for each peptide
        pep_n: Detection counts for each peptide
        filter_thres: Filtering threshold
    
    Returns:
        Tuple of (ranks_array, num_filtered)
    """
    n_pep = len(pep_indices)
    if n_pep == 0:
        return np.zeros(0, dtype=np.int32), 0
    
    # Extract intensities and counts for this protein's peptides
    top_pep_int = np.zeros(n_pep)
    top_pep_n = np.zeros(n_pep)
    
    for i in range(n_pep):
        idx = pep_indices[i]
        top_pep_int[i] = pep_int_sum[idx]
        top_pep_n[i] = pep_n[idx]
    
    ranks = np.zeros(n_pep, dtype=np.int32)
    num_filtered = 0
    
    if filter_thres == 0:
        # No filtering - just rank by total intensity
        # Sort indices by intensity (descending)
        sorted_idx = np.argsort(-top_pep_int)
        for rank_idx in range(n_pep):
            ranks[sorted_idx[rank_idx]] = rank_idx + 1
    else:
        # Filtering mode
        # Find maximum detection count
        n_max = np.max(top_pep_n) if n_pep > 0 else 0
        
        # Calculate average intensities
        avg_pep_int = np.zeros(n_pep)
        for i in range(n_pep):
            avg_pep_int[i] = top_pep_int[i] / (top_pep_n[i] + 1e-8)
        
        # Find reference average (from peptide with max detections)
        max_n_idx = np.argmax(top_pep_n)
        avg_sign_ref = avg_pep_int[max_n_idx] if n_pep > 0 else 0
        
        # Flag peptides to use
        use_pep = np.ones(n_pep, dtype=np.bool_)
        for i in range(n_pep):
            if top_pep_n[i] < filter_thres * n_max and avg_pep_int[i] > avg_sign_ref:
                use_pep[i] = False
                num_filtered += 1
        
        # Sort by intensity (descending) and assign ranks
        sorted_idx = np.argsort(-top_pep_int)
        rank_counter = 0
        
        for i in range(n_pep):
            idx = sorted_idx[i]
            if use_pep[idx]:
                rank_counter += 1
                ranks[idx] = rank_counter
            else:
                ranks[idx] = 0
    
    return ranks, num_filtered

def rank_peptide_levels(file_list: List[str]) -> Dict:
    """
    Fast peptide ranking using numba compilation and optimized algorithms.
    
    Args:
        file_list: List of CSV files containing peptide intensity data.
                  Format: Column 1 = Peptide ID, Column 2 = Protein ID, 
                  Columns 3+ = Sample intensities
    
    Returns:
        pep_data: Dictionary containing ranked peptide data
    """
    global rank_settings, verbose
    
    if not file_list:
        raise ValueError("Input file list is empty.")
    
    # Step 1: Fast data loading and preprocessing
    all_data = []
    all_peptides = []
    all_proteins = []
    all_conditions = []
    
    for f in file_list:
        if isinstance(f, pd.DataFrame):
            df = f
        else:
            try:
                # Optimized file reading with appropriate separators
                file_ext = f.split('.')[-1].lower()
                if file_ext == 'tsv' or file_ext == 'txt':
                    df = pd.read_csv(f, sep='\t')
                elif file_ext == 'csv':
                    df = pd.read_csv(f)
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading file {f}: {e}")
        
        if df.shape[1] < 3:
            raise ValueError(f"File {f} must have at least 3 columns (peptide, protein, samples)")
        
        # Extract data efficiently
        samples = df.columns[2:].tolist()
        peptides = df.iloc[:, 0].astype(str).values  # Use values for speed
        proteins = df.iloc[:, 1].astype(str).values
        intensities = df.iloc[:, 2:].values.astype(np.float64, copy=False)  # Ensure float64 for numba
        
        n_pep, n_samples = intensities.shape
        
        all_data.append({
            'filename': f,
            'samples': samples,
            'peptides': peptides,
            'proteins': proteins,
            'intensities': intensities,
            'n_pep': n_pep,
            'n_samples': n_samples
        })
        
        all_peptides.extend(peptides.tolist())
        all_proteins.extend(proteins.tolist())
        all_conditions.extend(samples)
        
        if verbose:
            if len(file_list) > 1:
                print(f"+")
                if isinstance(f, pd.DataFrame):
                    print(f"+ + + Source file: DataFrame (shape: {f.shape}) + + +")
                else:
                    print(f"+ + + Source file: {f} + + +")
            print(f"+  Number of samples: {n_samples}")
            print(f"+  Input number of peptides: {n_pep}")
            print(f"+  Input number of protein assignments: {len(np.unique(proteins))}")
    
    # Step 2: Fast unique peptide identification using numpy
    if verbose and len(file_list) > 1:
        print("+")
        print("+ + + Merging datasets + + +")
    
    # Convert to numpy arrays for faster processing
    all_peptides_arr = np.array(all_peptides, dtype='U50')  # Assuming max 50 chars for peptide names
    all_proteins_arr = np.array(all_proteins, dtype='U50')
    
    # Fast unique operation
    unique_peptides, u2apep, a2upep = np.unique(all_peptides_arr, return_index=True, return_inverse=True)
    n_upep = len(unique_peptides)
    
    # Get proteins for unique peptides efficiently
    unique_prot_for_upep = all_proteins_arr[u2apep]
    unique_proteins, _, pvec_all2unique = np.unique(unique_prot_for_upep, return_index=True, return_inverse=True)
    n_prot = len(unique_proteins)
    
    if verbose and len(file_list) > 1:
        print(f"+  Total number of samples: {len(all_conditions)}")
        print(f"+  Unique peptides: {n_upep}")
        print(f"+  Unique proteins: {n_prot}")
    
    # Step 3: Fast peptide statistics calculation using numba
    pep_int_sum = np.zeros(n_upep, dtype=np.float64)
    pep_n = np.zeros(n_upep, dtype=np.float64)
    pep_max = np.zeros(n_upep, dtype=np.float64)
    
    # Create cumulative offsets for efficient indexing
    cumulative_offset = 0
    
    for data_dict in all_data:
        n_pep = data_dict['n_pep']
        intensities = data_dict['intensities']
        
        # Map peptides to unique indices for this file
        peptide_indices = a2upep[cumulative_offset:cumulative_offset + n_pep]
        
        # Fast statistics calculation using numba
        file_int_sum, file_n, file_max = _calculate_peptide_stats_numba(
            intensities, peptide_indices, n_upep
        )
        
        # Accumulate results
        pep_int_sum += file_int_sum
        pep_n += file_n
        pep_max = np.maximum(pep_max, file_max)
        
        cumulative_offset += n_pep
    
    # Step 4: Fast ranking using optimized sparse matrix operations
    if verbose:
        if rank_settings['filterThres'] == 0:
            print("+  Ranking peptides according to total intensity... ", end='')
        else:
            print(f"+  Filtering and ranking (filterThres = {rank_settings['filterThres']:.2f})... ", end='')
    
    # Create efficient protein-to-peptide mapping
    protein_peptide_map = {}
    for pep_idx in range(n_upep):
        prot_idx = pvec_all2unique[pep_idx]
        if prot_idx not in protein_peptide_map:
            protein_peptide_map[prot_idx] = []
        protein_peptide_map[prot_idx].append(pep_idx)
    
    # Pre-allocate sparse matrix data
    row_indices = []
    col_indices = []
    data_values = []
    
    cnt_prot_cut = 0
    cnt_pep_cut = 0
    
    # Process each protein using numba-compiled ranking
    for prot_idx in range(n_prot):
        if prot_idx not in protein_peptide_map:
            continue
            
        pep_indices = np.array(protein_peptide_map[prot_idx], dtype=np.int32)
        
        # Fast ranking using numba
        ranks, num_filtered = _rank_peptides_for_protein_numba(
            pep_indices, pep_int_sum, pep_n, rank_settings['filterThres']
        )
        
        # Store results efficiently
        for i, rank in enumerate(ranks):
            pep_idx = pep_indices[i]
            if rank > 0:  # Only store non-zero ranks
                row_indices.append(pep_idx)
                col_indices.append(prot_idx)
                data_values.append(rank)
        
        if num_filtered > 0:
            cnt_prot_cut += 1
            cnt_pep_cut += num_filtered
    
    # Create sparse matrix efficiently
    pep2prot = sp.csc_matrix((data_values, (row_indices, col_indices)), 
                             shape=(n_upep, n_prot), dtype=np.int32)
    
    # Build optimized result dictionary
    pep_data = {
        'input_file': file_list,
        'unique_pepPrec': unique_peptides.tolist(),
        'unique_protein': unique_proteins.tolist(),
        'pep2prot': pep2prot,
        'pepInt': pep_int_sum,
        'pepN': pep_n,
        'pepMax': pep_max
    }
    
    if verbose:
        print("ok")
        if rank_settings['filterThres'] > 0:
            print(f"+  -> Peptides filtered out: {cnt_pep_cut}")
            print(f"+  -> Proteins affected: {cnt_prot_cut}")
    
    return pep_data

def remove_problematic_peptides(pep_data: Dict, pep_list: List[str]) -> Dict:
    """Remove specified problematic peptides from the analysis."""
    global verbose
    
    cnt_removed = 0
    
    if pep_list:
        if verbose == 1:
            print("\n* Removing problematic peptide(s)...")
        elif verbose > 1:
            print("\n* Removing problematic peptide(s):")
        
        for pep in pep_list:
            try:
                i = pep_data['unique_pepPrec'].index(pep)
                p = pep_data['pep2prot'][i, :].nonzero()[1]
                
                if verbose > 1:
                    print(f"  [{i}] {pep_data['unique_pepPrec'][i]}")
                    if len(p) > 0:
                        print(f"    associated to {pep_data['unique_protein'][p[0]]}")
                    else:
                        print("    not found")
                
                if len(p) > 0:
                    # Update ranks
                    ri = pep_data['pep2prot'][i, p[0]]
                    jj_up = pep_data['pep2prot'][:, p[0]] > ri
                    pep_data['pep2prot'][jj_up, p[0]] -= 1
                    
                    # Remove peptide
                    del pep_data['unique_pepPrec'][i]
                    pep_data['pep2prot'] = sp.vstack([
                        pep_data['pep2prot'][:i, :],
                        pep_data['pep2prot'][i+1:, :]
                    ]).tocsc()
                    pep_data['pepInt'] = np.delete(pep_data['pepInt'], i)
                    pep_data['pepN'] = np.delete(pep_data['pepN'], i)
                    pep_data['pepMax'] = np.delete(pep_data['pepMax'], i)
                    
                    cnt_removed += 1
            except ValueError:
                if verbose > 1:
                    print(f"  Peptide {pep} not found")
        
        if verbose:
            print(f"*   {cnt_removed} out of {len(pep_list)} removed.")
    
    return pep_data

def export_counts_from_pep(pep_data: Dict, source_file: str, output_base: str|None = None) -> Dict[str, pd.DataFrame]:
    """Export protein counts from peptide data."""
    global top_pep_settings, x_top_settings, verbose

    output_dict = {}

    if isinstance(source_file, pd.DataFrame):
        df = source_file
    else:
    # Read input file with proper format detection
        try:
            # Detect file format by extension
            file_ext = source_file.split('.')[-1].lower()
            if file_ext == 'tsv':
                df = pd.read_csv(source_file, sep='\t')
            elif file_ext == 'txt':
                df = pd.read_csv(source_file, sep='\t')
            else:
                df = pd.read_csv(source_file)
        except Exception as e:
            raise ValueError(f"Error reading file {source_file}: {e}")
    
    # Validate file structure
    if df.shape[1] < 3:
        raise ValueError(f"File {source_file} must have at least 3 columns (peptide, protein, samples), but has {df.shape[1]} columns")
    
    data_samples = df.columns[2:].tolist()
    data_pep_list = df.iloc[:, 0].values
    data_prot_list = df.iloc[:, 1].values
    data_values = df.iloc[:, 2:].values.astype(float) # Ensure data is float
    
    # Get unique proteins
    unique_prot = np.unique(data_prot_list)
    n_prot = len(unique_prot)
    n_samples = len(data_samples)
    
    if verbose:
        if isinstance(source_file, pd.DataFrame):
            print(f"\n* * * Protein intensities\n* Output:   None (DataFrame input)")
        else:
            print(f"\n* * * Protein intensities\n* Output:   {output_base}")
    
    # Prepare peptide and rank lookups
    pep_name_map = {}
    pep_rank_map = {}
    for pi, prot_name in enumerate(unique_prot):
        pj = np.where(np.array(pep_data['unique_protein']) == prot_name)[0]
        if len(pj) == 0:
            continue
        pj = pj[0]
        
        qj = pep_data['pep2prot'][:, pj].nonzero()[0]
        pep_names = [pep_data['unique_pepPrec'][i] for i in qj]
        pep_ranks = pep_data['pep2prot'][qj, pj].toarray().flatten()
        
        pep_name_map[prot_name] = pep_names
        pep_rank_map[prot_name] = {name: rank for name, rank in zip(pep_names, pep_ranks)}

    # Process TopPepN quantities
    for top_pep_val in top_pep_settings['TopPepVal']:
        str_x = f"TopPep{int(top_pep_val)}" if not np.isinf(top_pep_val) else "TopPepAll"
        if verbose:
            print(f"*   {str_x} intensity ... ", end='')
        
        m_top_pep_n = np.zeros((n_prot, n_samples))
        m_n_pep = np.zeros((n_prot, n_samples))
        
        for pi, prot_name in enumerate(unique_prot):
            if prot_name not in pep_name_map:
                continue
            
            pep_names = pep_name_map[prot_name]
            rank_map = pep_rank_map[prot_name]
            
            # Find peptides in the current dataset
            in_dataset_mask = np.isin(data_pep_list, pep_names)
            current_peps = data_pep_list[in_dataset_mask]
            current_values = data_values[in_dataset_mask, :]
            
            # Get ranks for these peptides
            current_ranks = np.array([rank_map.get(p, 0) for p in current_peps])
            
            # Select peptides by rank
            if np.isinf(top_pep_val):
                selected = current_ranks > 0
            else:
                selected = (current_ranks > 0) & (current_ranks <= top_pep_val)
            
            if np.any(selected):
                m_top_pep_n[pi, :] = np.sum(current_values[selected, :], axis=0)
                m_n_pep[pi, :] = np.sum(current_values[selected, :] > 0, axis=0)
        
        # Export intensities
        f_out_x = f"{output_base} Intensity {str_x}.csv"
        df_out = export_dlm_data(f_out_x if output_base else None, data=[unique_prot] + [m_top_pep_n[:, s] for s in range(n_samples)], header=['Protein'] + data_samples)
        output_dict[f'Intensity_{str_x}'] = df_out

        if top_pep_settings['export_Npep']:
            if verbose:
                print("Npep ...", end='')
            f_out_x = f"{output_base} Npep {str_x}.csv"
            df_out = export_dlm_data(f_out_x if output_base else None, data=[unique_prot] + [m_n_pep[:, s] for s in range(n_samples)], header=['Protein'] + data_samples)
            output_dict[f'Npep_{str_x}'] = df_out

        if verbose:
            print(" ok")

    # Process xTop quantities
    if x_top_settings['xTopPepVal']:
        if verbose == 1:
            print("*   xTop ... ", end='')
        elif verbose > 1:
            print("* * * xTop protein intensities * * *")
            print("Detailed info on the calculation of xTop intensities will be printed out.\n")

        # Parallel computation setup using joblib
        if x_top_settings['parallelize_flag'] and JOBLIB_AVAILABLE:
            n_workers = os.cpu_count() if x_top_settings['max_parallel_workers'] == -1 else x_top_settings['max_parallel_workers']
            backend = x_top_settings.get('backend', 'threading')
            
            # Prepare tasks for parallel execution
            tasks = []
            task_indices = []
            for pi, prot_name in enumerate(unique_prot):
                if prot_name not in pep_name_map:
                    continue
                
                pep_names = pep_name_map[prot_name]
                in_dataset_mask = np.isin(data_pep_list, pep_names)
                qi = np.where(in_dataset_mask)[0]
                
                tasks.append(delayed(get_xTop_estimator_wrapper)(data_values[qi, :], x_top_settings, verbose))
                task_indices.append(pi)
            
            # Execute tasks in parallel with joblib
            if n_workers > 0 and len(tasks) > 0:
                parallel_results = Parallel(n_jobs=n_workers, backend=backend)(tasks)
            else:
                parallel_results = []
            
            # Map results back to original protein order
            results = [(None, None)] * len(unique_prot)
            for i, result in enumerate(parallel_results):
                results[task_indices[i]] = result
                
        else:
            # Sequential processing (fallback)
            results = []
            for pi, prot_name in enumerate(unique_prot):
                if prot_name not in pep_name_map:
                    results.append((None, None))
                    continue
                
                pep_names = pep_name_map[prot_name]
                in_dataset_mask = np.isin(data_pep_list, pep_names)
                qi = np.where(in_dataset_mask)[0]
                
                result = get_xTop_estimator_wrapper(data_values[qi, :], x_top_settings, verbose)
                results.append(result)

        # Process results
        m_i_xtop = np.zeros((n_prot, n_samples))
        m_log_i_xtop_var = np.zeros((n_prot, n_samples))
        
        all_pep_stats = []

        for pi, (xTop_output, xTop_wrapper_output) in enumerate(results):
            if xTop_output is None:
                continue
            
            m_i_xtop[pi, :] = xTop_output['I_xTop']
            m_log_i_xtop_var[pi, :] = xTop_output['log_I_xTop_var']
            
            # Collect peptide stats for 'full' export
            if x_top_settings['export_level'] == 'full':
                pep_names = pep_name_map[unique_prot[pi]]
                in_dataset_mask = np.isin(data_pep_list, pep_names)
                
                # Get the filtered data
                filtered_peptides = data_pep_list[in_dataset_mask]
                filtered_values = data_values[in_dataset_mask, :]
                
                # Make sure all arrays have the same length as the filtered peptides
                n_filtered = len(filtered_peptides)
                n_xtop = len(xTop_output['eff'])
                
                # The xTop output arrays should have the same length as filtered peptides
                # If not, use only as much data as available
                n_data = min(n_filtered, n_xtop)
                
                if n_data > 0:
                    # Create peptide ranks for available data
                    eff_values = xTop_output['eff'][:n_data]
                    pep_ranks = np.argsort(-eff_values) + 1
                    
                    pep_stats = pd.DataFrame({
                        'Peptide': filtered_peptides[:n_data],
                        'protein': [unique_prot[pi]] * n_data,
                        'pep_rank': pep_ranks,
                        'pep_detections': np.sum(filtered_values[:n_data] > 0, axis=1),
                        'relQuant_isValid': xTop_output['relQuant_valid_flag'][:n_data],
                        'absQuant_isValid': xTop_output['absQuant_valid_flag'][:n_data],
                        'pep_weight': xTop_output['pep_weight'][:n_data],
                        'eff': xTop_output['eff'][:n_data],
                        'log_eff_var': xTop_output['log_eff_var'][:n_data]
                    })
                    all_pep_stats.append(pep_stats)
                else:
                    # No data available, create empty DataFrame
                    pep_stats = pd.DataFrame({
                        'Peptide': [],
                        'protein': [],
                        'pep_rank': [],
                        'pep_detections': [],
                        'relQuant_isValid': [],
                        'absQuant_isValid': [],
                        'pep_weight': [],
                        'eff': [],
                        'log_eff_var': []
                    })
                    all_pep_stats.append(pep_stats)

        # Export xTop results
        """
            xTop_output = {
        'Nc': I_peptides.shape[1],
        'N_valid_conditions': np.sum(valid_condition_flag),
        'N_valid_peptides': N_valid_peptides,
        'valid_peptide_flag': relQuant_valid_flag,
        'valid_condition_flag': valid_condition_flag,
        'N_detected': N_detected,
        'I_xTop': I_xTop,
        'log_I_xTop_var': log_I_xTop_var,
        'eff': eff,
        'log_eff_var': log_eff_var,
        'pep_weight': pep_weight,
        'relQuant_valid_flag': relQuant_valid_flag,
        'absQuant_valid_flag': absQuant_valid_flag,
        'optimization_status': optimization_status,
        's2_init': s2_init,
        'N_step': N_step,
        'L': L_iter,
        'L_optimal': L_optimal,
        'diff_L': diff_L,
        'time_elapsed': time_elapsed
                }
        """
        for val in x_top_settings['xTopPepVal']:
            str_x = str(val) if not np.isinf(val) else "All"
            
            # Calculate Veff (effective volume) for each protein based on xTopPepVal
            # This matches the MATLAB implementation
            veff = np.zeros((n_prot, n_samples))
            
            for pi, prot_name in enumerate(unique_prot):
                if prot_name not in pep_name_map:
                    continue
                
                # Get the xTop result for this protein
                xTop_output = results[pi][0]
                if xTop_output is None:
                    continue
                
                # Get peptide efficiencies and validity flags
                eff = xTop_output['eff']
                absQuant_valid = xTop_output['absQuant_valid_flag']
                
                # Create adjusted ranking based on valid peptides only
                # This matches MATLAB's par_pep_rank_adjusted_sorted = cumsum(absQuant_valid{pi}')
                adjusted_ranks = np.cumsum(absQuant_valid)
                
                # Select peptides based on xTopPepVal cutoff
                if np.isinf(val):
                    # For "All", use all valid peptides
                    selected = absQuant_valid
                else:
                    # For specific values (1, 3, etc.), use top N valid peptides
                    selected = absQuant_valid & (adjusted_ranks <= val)
                
                # Calculate Veff as sum of selected efficiencies
                if np.any(selected):
                    veff_value = np.sum(eff[selected])
                    veff[pi, :] = veff_value
            
            # Final intensities = Veff * base xTop intensities (matching MATLAB: Veff(:,xi) .* M_I_xTop(:,s))
            final_intensities = veff * m_i_xtop
            
            # Export as integers (matching MATLAB: floor(Veff(:,xi) .* M_I_xTop(:,s)))
            
            final_intensities = np.floor(final_intensities)
            
            f_out_x1 = f"{output_base} Intensity xTop{str_x}.csv"
            df_out = export_dlm_data(f_out_x1 if output_base else None, [unique_prot] + [final_intensities[:, s] for s in range(n_samples)],
                            header=['Protein'] + data_samples)
            output_dict[f'Intensity_xTop{str_x}'] = df_out
            
            # Error export
            if x_top_settings['export_level'] == 'full':
                error_type = x_top_settings['error_type']
                if error_type == 'relative (std-log)':
                    errors = np.sqrt(m_log_i_xtop_var)
                    f_out_x2 = f"{output_base} Error (std-log) xTop{str_x}.csv"
                elif error_type == 'relative (CV)':
                    errors = np.sqrt(np.exp(m_log_i_xtop_var) - 1)
                    f_out_x2 = f"{output_base} Error (CV) xTop{str_x}.csv"
                elif error_type == 'absolute':
                    cv = np.sqrt(np.exp(m_log_i_xtop_var) - 1)
                    errors = (final_intensities * cv).astype(int)
                    f_out_x2 = f"{output_base} Error xTop{str_x}.csv"
                else:
                    errors = None
                
                if errors is not None:
                    df_out = export_dlm_data(f_out_x2 if output_base else None, [unique_prot] + [errors[:, s] for s in range(n_samples)],
                                              header=['Protein'] + data_samples)
                    output_dict[f'Error_{error_type}_xTop{str_x}'] = df_out

        if x_top_settings['export_level'] == 'full' and all_pep_stats:
            all_pep_stats_df = pd.concat(all_pep_stats).sort_values(by=['protein', 'pep_rank'])
            f_out_pep = f"{output_base} xTop peptide statistics.csv"
            all_pep_stats_df.to_csv(f_out_pep, index=False)

        if verbose:
            print("ok")
        return output_dict

def export_dlm_data(filename: str| None, data: Union[np.ndarray, List[list]], header: List[str] = None,
                      delimiter: str = ',', fmt: Union[str, List[str]] = '%.6f'):
    """
    Flexible function to export tabular data to a delimited text file.
    
    Args:
        filename: Output filename.
        data: Data to be printed. Can be a NumPy array or a list of lists/arrays.
        header: List of strings for the header row.
        delimiter: Delimiter for the output file.
        fmt: Format string for the data.
    """
    df_data = {}
    if isinstance(data, list):
        # Assumes data is a list of columns
        max_len = max(len(col) for col in data) if data else 0
        for i, col in enumerate(data):
            col_name = header[i] if header and i < len(header) else f'col_{i}'
            # Pad shorter columns with empty strings or NaNs
            if len(col) < max_len:
                padding = [''] * (max_len - len(col))
                if np.issubdtype(np.array(col).dtype, np.number):
                    padding = [np.nan] * (max_len - len(col))
                col = np.concatenate((col, padding))
            df_data[col_name] = col
        df = pd.DataFrame(df_data)
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=header)
    else:
        raise TypeError("Unsupported data type for export")
    if header is not None:
        df.to_csv(filename, sep=delimiter, index=False, header=True)

    return df


def get_xTop_estimator_wrapper(I_peptides: np.ndarray, xTopSettings: Dict, verbose: int) -> Tuple[Dict, Dict]:
    """
    Calls get_xTop_estimator several times and picks the best solution.
    """
    n_optimizations = xTopSettings.get('N_optimizations', 1)

    if n_optimizations == 1:
        xTop_output = get_xTop_estimator(I_peptides, xTopSettings, verbose)
        result_status = xTop_output['optimization_status']
        result_L = xTop_output['L_optimal']
        L_optimal = xTop_output['L_optimal']
        result_N_step = xTop_output['N_step']
        result_opt_idx = 1
        Delta_L = 0
    else:
        if verbose > 1:
            print(f" * * * N_optimizations = {n_optimizations}:")
        
        results = []
        for _ in range(n_optimizations):
            results.append(get_xTop_estimator(I_peptides, xTopSettings, verbose))
        
        result_status = np.array([r['optimization_status'] for r in results])
        result_L = np.array([r['L_optimal'] for r in results])
        result_N_step = np.array([r['N_step'] for r in results])
        
        L_optimal = np.min(result_L)
        Delta_L = result_L - L_optimal
        
        result_opt_idx = np.argmin(Delta_L)
        xTop_output = results[result_opt_idx]

    xTop_wrapper_output = {
        'status': result_status,
        'opt_idx': result_opt_idx,
        'N_detected': xTop_output['N_detected'],
        'N_valid_peptides': xTop_output['N_valid_peptides'],
        'N_step': result_N_step,
        'L': result_L,
        'L_optimal': L_optimal,
        'L_range': np.max(Delta_L) if n_optimizations > 1 else 0,
        'N_local_minima': np.sum((result_L - np.min(result_L)) > 1e2 * xTopSettings['accuracy_max'] * abs(L_optimal))
    }
    
    if verbose > 1 and n_optimizations > 1:
        print(f"---> max(L)-min(L) = {xTop_wrapper_output['L_range']}\n")
        
    return xTop_output, xTop_wrapper_output


@jit(nopython=True)
def _get_xTop_estimator_numba(I_peptides, thres_xTop, N_c_thres, s2_min, 
                              accuracy_max, N_step_max, absolute_filter_ratio):
    """
    Numba-accelerated core of the xTop estimator.
    """
    Npep, Nc = I_peptides.shape
    
    # Identify valid peptides
    valid_peptide_flag = np.zeros(Npep, dtype=np.bool_)
    for k in range(Npep):
        if np.sum(I_peptides[k, :] > thres_xTop) >= N_c_thres:
            valid_peptide_flag[k] = True
    
    # Extract valid peptides manually (Numba-compatible)
    valid_indices = []
    for k in range(Npep):
        if valid_peptide_flag[k]:
            valid_indices.append(k)
    
    if len(valid_indices) == 0:
        I_peptides_valid = np.zeros((0, Nc))
    else:
        I_peptides_valid = np.zeros((len(valid_indices), Nc))
        for i, idx in enumerate(valid_indices):
            I_peptides_valid[i, :] = I_peptides[idx, :]
    
    # Check for isolated peptides
    isDetected = I_peptides_valid > 0
    peptidesInSample = np.sum(isDetected, axis=0)
    isolatedPeptidesPresent = False
    if np.max(peptidesInSample) > 1:
        otherPeptidesDetected = np.zeros(np.sum(valid_peptide_flag))
        for k in range(np.sum(valid_peptide_flag)):
            s_idx = isDetected[k, :]
            if np.sum(s_idx) > 0:
                otherPeptidesDetected[k] = np.max(peptidesInSample[s_idx]) - 1
        
        isolatedPeptide = (otherPeptidesDetected == 0)
        if np.sum(isolatedPeptide) < np.max(peptidesInSample):
            # Remove isolated peptides from valid_peptide_flag
            valid_peptide_counter = 0
            for k in range(Npep):
                if valid_peptide_flag[k]:
                    if isolatedPeptide[valid_peptide_counter]:
                        valid_peptide_flag[k] = False
                    valid_peptide_counter += 1
        else:
            isolatedPeptidesPresent = True

    # Extract valid peptides again after isolation check (Numba-compatible)
    valid_indices = []
    for k in range(Npep):
        if valid_peptide_flag[k]:
            valid_indices.append(k)
    
    if len(valid_indices) == 0:
        I_peptides_valid = np.zeros((0, Nc))
    else:
        I_peptides_valid = np.zeros((len(valid_indices), Nc))
        for i, idx in enumerate(valid_indices):
            I_peptides_valid[i, :] = I_peptides[idx, :]
    Npep_nz = I_peptides_valid.shape[0]

    # Identify valid conditions
    valid_condition_flag = np.sum(I_peptides_valid > thres_xTop, axis=0) > 0
    
    # Extract valid conditions manually (Numba-compatible)
    valid_condition_indices = []
    for j in range(Nc):
        if valid_condition_flag[j]:
            valid_condition_indices.append(j)
    
    if len(valid_condition_indices) == 0:
        I_peptides_final = np.zeros((Npep_nz, 0))
    else:
        I_peptides_final = np.zeros((Npep_nz, len(valid_condition_indices)))
        for j_idx, j in enumerate(valid_condition_indices):
            I_peptides_final[:, j_idx] = I_peptides_valid[:, j]
    
    Nc_nz = I_peptides_final.shape[1]

    N_detected = np.sum(I_peptides_final > 0)
    N_param = 2 * Npep_nz + Nc_nz

    # Handle edge cases
    if Npep_nz == 0: return 0, np.zeros(Nc), np.zeros(Nc), np.zeros(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=np.bool_), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1)
    if Npep_nz == 1:
        I_xTop = np.zeros(Nc)
        # Manually assign values to valid conditions (Numba-compatible)
        for j_idx, j in enumerate(valid_condition_indices):
            I_xTop[j] = I_peptides_final[0, j_idx]
        return -1, I_xTop, np.zeros(Nc), np.zeros(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=np.bool_), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1)
    if isolatedPeptidesPresent:
        I_xTop = np.zeros(Nc)
        # Manually assign values to valid conditions (Numba-compatible)
        sum_vals = np.sum(I_peptides_final, axis=0)
        for j_idx, j in enumerate(valid_condition_indices):
            I_xTop[j] = sum_vals[j_idx]
        return -2, I_xTop, np.zeros(Nc), np.zeros(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=np.bool_), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1)
    if N_param > N_detected:
        I_xTop = np.zeros(Nc)
        # Numba-compatible way to compute max along axis 0
        max_vals = np.zeros(Nc_nz)
        for j in range(Nc_nz):
            max_vals[j] = np.max(I_peptides_final[:, j])
        # Manually assign values to valid conditions (Numba-compatible)
        for j_idx, j in enumerate(valid_condition_indices):
            I_xTop[j] = max_vals[j_idx]
        return -3, I_xTop, np.zeros(Nc), np.zeros(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=np.bool_), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1)

    # Setup for iterative procedure
    # Extract non-zero values manually (Numba-compatible way)
    y_list = []
    rows = []
    cols = []
    for i in range(Npep_nz):
        for j in range(Nc_nz):
            if I_peptides_final[i, j] > 0:
                y_list.append(np.log(I_peptides_final[i, j]))
                rows.append(i)
                cols.append(j)
    
    y = np.array(y_list)
    rows = np.array(rows)
    cols = np.array(cols)
    Nr_nz = len(y)
    
    # Build A matrix components
    vAr = np.arange(Nr_nz)
    vAq_eff = rows
    vAq_int = cols + Npep_nz
    
    # Initialize variances
    s2 = s2_min * (100 + 400 * np.random.rand(Npep_nz))
    s2_init = s2.copy()

    # Iterative procedure
    diff_L = 1.0
    step = 0
    L_iter = np.zeros(N_step_max)
    
    while diff_L > accuracy_max and step < N_step_max:
        step += 1
        
        # Construct S-related terms
        s2_inv = 1.0 / s2
        S_diag = s2_inv[rows]
        
        # Construct M1 matrix (A'SA)
        M1 = np.zeros((Npep_nz + Nc_nz, Npep_nz + Nc_nz))
        
        # Diagonal elements
        for p in range(Npep_nz):
            M1[p, p] = np.sum(S_diag[rows == p])
        for c in range(Nc_nz):
            M1[Npep_nz + c, Npep_nz + c] = np.sum(S_diag[cols == c])
            
        # Off-diagonal elements
        for r in range(Nr_nz):
            p, c = rows[r], cols[r]
            M1[p, Npep_nz + c] = S_diag[r]
            M1[Npep_nz + c, p] = S_diag[r]
            
        # Add extra constraint - manually construct the constrained matrix (Numba-compatible)
        constraint_row = np.append(np.ones(Npep_nz), np.zeros(Nc_nz))
        constraint_col = np.append(constraint_row, 0)
        
        # Create M1_constrained manually
        M1_constrained = np.zeros((Npep_nz + Nc_nz + 1, Npep_nz + Nc_nz + 1))
        # Copy original M1
        for i in range(Npep_nz + Nc_nz):
            for j in range(Npep_nz + Nc_nz):
                M1_constrained[i, j] = M1[i, j]
        # Add constraint row
        for j in range(Npep_nz + Nc_nz):
            M1_constrained[Npep_nz + Nc_nz, j] = constraint_row[j]
        # Add constraint column  
        for i in range(Npep_nz + Nc_nz + 1):
            M1_constrained[i, Npep_nz + Nc_nz] = constraint_col[i]
        
        # Construct M2 vector (A'Sy)
        M2 = np.zeros(Npep_nz + Nc_nz)
        for r in range(Nr_nz):
            p, c = rows[r], cols[r]
            M2[p] += S_diag[r] * y[r]
            M2[Npep_nz + c] += S_diag[r] * y[r]
        
        M2_constrained = np.append(M2, 0)
        
        # Solve linear system
        try:
            x_full = np.linalg.solve(M1_constrained, M2_constrained)
            x = x_full[:-1]  # Remove the last element (Lagrange multiplier)
        except Exception:
            return -4, np.zeros(Nc), np.zeros(Nc), np.zeros(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=np.bool_), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1)

        # Update variances
        residuals = np.zeros(Nr_nz)
        for r in range(Nr_nz):
            p, c = rows[r], cols[r]
            residuals[r] = x[p] + x[Npep_nz + c] - y[r]
            
        for p in range(Npep_nz):
            iy = (rows == p)
            s2[p] = s2_min + np.sum(residuals[iy]**2) / np.sum(iy)
            
        # Calculate log-posterior
        L_val = 0.5 * np.sum(residuals**2 * S_diag) + 0.5 * np.sum(np.sum(rows == p) * (s2_min / s2 + np.log(s2)))
        L_iter[step-1] = L_val
        
        if step > 1:
            diff_L = abs((L_iter[step-1] / L_iter[step-2]) - 1) if L_iter[step-2] != 0 else 1.0
    
    # Finalize results
    optimization_status = 1 if diff_L <= accuracy_max else 2
    
    # Rescale efficiencies and intensities
    s2_pep_valid = s2
    s2_absThres = s2_min / absolute_filter_ratio
    idxAbs = s2_pep_valid < s2_absThres
    
    reliable_absQuant = True
    if np.sum(idxAbs) == 0:
        reliable_absQuant = False
        max_eff_val = np.max(x[:Npep_nz])
    else:
        v_x = x[:Npep_nz]
        v_x[~idxAbs] = -np.inf
        max_eff_val = np.max(v_x)
        
    x[:Npep_nz] -= max_eff_val
    x[Npep_nz:] += max_eff_val
    
    eff_valid = np.exp(x[:Npep_nz])
    I_xTop_valid = np.exp(x[Npep_nz:])
    
    # Map back to original dimensions
    eff = np.zeros(Npep)
    # Manually assign values (Numba-compatible)
    valid_idx = 0
    for k in range(Npep):
        if valid_peptide_flag[k]:
            eff[k] = eff_valid[valid_idx]
            valid_idx += 1
    
    I_xTop = np.zeros(Nc)
    # Manually assign values to valid conditions (Numba-compatible)
    for j_idx, j in enumerate(valid_condition_indices):
        I_xTop[j] = I_xTop_valid[j_idx]
    # Apply threshold
    for j in range(Nc):
        if I_xTop[j] < thres_xTop:
            I_xTop[j] = 0
    
    # Error estimation (simplified)
    log_eff_var = np.zeros(Npep)
    log_I_xTop_var = np.zeros(Nc)
    try:
        C_vec = np.diag(np.linalg.pinv(M1_constrained))
        log_eff_var_valid = C_vec[:Npep_nz]
        log_I_xTop_var_valid = C_vec[Npep_nz:Npep_nz+Nc_nz]
        # Manually assign values (Numba-compatible)
        valid_idx = 0
        for k in range(Npep):
            if valid_peptide_flag[k]:
                log_eff_var[k] = log_eff_var_valid[valid_idx]
                valid_idx += 1
        # Manually assign values to valid conditions (Numba-compatible)
        for j_idx, j in enumerate(valid_condition_indices):
            log_I_xTop_var[j] = log_I_xTop_var_valid[j_idx]
    except Exception:
        pass # Keep as zeros if pinv fails

    pep_weight = np.zeros(Npep)
    pep_weight_valid = 1.0 / s2
    # Manually assign values (Numba-compatible)
    valid_idx = 0
    for k in range(Npep):
        if valid_peptide_flag[k]:
            pep_weight[k] = pep_weight_valid[valid_idx]
            valid_idx += 1
    
    absQuant_valid_flag = np.zeros(Npep, dtype=np.bool_)
    if reliable_absQuant:
        absQuant_valid_flag_valid = s2 < s2_absThres
        # Manually assign values (Numba-compatible)
        valid_idx = 0
        for k in range(Npep):
            if valid_peptide_flag[k]:
                absQuant_valid_flag[k] = absQuant_valid_flag_valid[valid_idx]
                valid_idx += 1

    return optimization_status, I_xTop, log_I_xTop_var, eff, log_eff_var, pep_weight, valid_peptide_flag, absQuant_valid_flag, N_detected, Npep_nz, valid_condition_flag, s2_init, step, L_iter[step-1], diff_L, L_iter

def get_xTop_estimator(I_peptides: np.ndarray, xTopSettings: Dict, verbose: int) -> Dict:
    """
    Main function for calculating the xTop estimator.
    """
    # Set random seed
    if xTopSettings['rng_seed'] != 'shuffle':
        np.random.seed(xTopSettings['rng_seed'])

    # Call the Numba-accelerated function
    (optimization_status, I_xTop, log_I_xTop_var, eff, log_eff_var, pep_weight, 
     relQuant_valid_flag, absQuant_valid_flag, N_detected, N_valid_peptides, 
     valid_condition_flag, s2_init, N_step, L_optimal, diff_L, L_iter) = \
        _get_xTop_estimator_numba(I_peptides, xTopSettings['thres_xTop'], xTopSettings['N_c_thres'],
                                  xTopSettings['s2_min'], xTopSettings['accuracy_max'], 
                                  xTopSettings['N_step_max'], xTopSettings['absolute_filter_ratio'])

    xTop_output = {
        'Nc': I_peptides.shape[1],
        'N_valid_conditions': np.sum(valid_condition_flag),
        'N_valid_peptides': N_valid_peptides,
        'valid_peptide_flag': relQuant_valid_flag,
        'valid_condition_flag': valid_condition_flag,
        'N_detected': N_detected,
        'I_xTop': I_xTop,
        'log_I_xTop_var': log_I_xTop_var,
        'eff': eff,
        'log_eff_var': log_eff_var,
        'pep_weight': pep_weight,
        'relQuant_valid_flag': relQuant_valid_flag,
        'absQuant_valid_flag': absQuant_valid_flag,
        'optimization_status': optimization_status,
        's2_init': s2_init,
        'N_step': N_step,
        'L': L_iter,
        'L_optimal': L_optimal,
        'diff_L': diff_L,
    }

    if verbose > 1:
        print(f"Optimization status: {optimization_status:2d}     Steps: {N_step:3d}     L : {L_optimal:10.5f}   diff_L : {diff_L:8.5e}")

    if xTopSettings['plot_xTop_convergence'] and optimization_status > 0:
        # Plotting logic would go here, but it's complex to do with Numba output
        # For simplicity, this part is omitted in the conversion.
        pass
        
    return xTop_output

def export_pep_ranking(pep_data: Dict, filename: str|None)-> pd.DataFrame:
    """
    Export peptide ranking to CSV file.
    
    Args:
        pep_data: Peptide data dictionary
        filename: Output CSV filename
    """
    n_prot = len(pep_data['unique_protein'])

    prot_name_col = []
    pep_name_col = []
    rank_col = []   

    for a in range(n_prot):
        prot_name = pep_data['unique_protein'][a]
        # Find peptides for this protein
        pep_indices, _, pep_ranks = sp.find(pep_data['pep2prot'][:, a])
        
        # Filter out unranked peptides (rank 0)
        ranked_peptides = pep_indices[pep_ranks > 0]
        ranks = pep_ranks[pep_ranks > 0]
        
        # Sort by rank
        sorted_indices = np.argsort(ranks)
        
        for k, idx in enumerate(sorted_indices):
            pep_index = ranked_peptides[idx]
            pep_name = pep_data['unique_pepPrec'][pep_index]
            rank = ranks[idx]
            prot_name_col.append(prot_name)
            pep_name_col.append(pep_name)
            rank_col.append(rank)

    peptide_rank_df = pd.DataFrame({
        'Protein': prot_name_col,
        'Peptide': pep_name_col,
        'Rank': rank_col
    })
    if filename is None:
        # if is none, reture the dataframe
        pass
    else:
        # If a filename is provided, save the DataFrame to a CSV file
        peptide_rank_df.to_csv(filename, index=False)
    return peptide_rank_df

    

    

if __name__ == '__main__':
    # This block allows the script to be run from the command line.
    # Example usage: python xTop.py test_file1.csv test_file2.csv
    import sys
    
    # --- Default Settings ---
    # You can modify these settings directly or via command-line arguments in a more advanced setup.
    
    # I/O Settings
    # source_filename = ['test_file1.csv', 'test_file2.csv'] # Default if no args
    
    # Ranking Settings
    rank_settings['rankFiles'] = 'all' # 'individual' or 'all'
    rank_settings['filterThres'] = 0.5
    rank_settings['export_ranking'] = True
    
    # TopPep-N Settings
    top_pep_settings['TopPepVal'] = [1, 3, np.inf] # TopPep1, TopPep3, TopPepAll
    top_pep_settings['export_Npep'] = True
    
    # xTop Settings
    x_top_settings['xTopPepVal'] = [1]
    x_top_settings['export_level'] = 'full'
    x_top_settings['parallelize_flag'] = True # Set to False if you encounter issues with parallel processing
    x_top_settings['backend'] = 'threading' # 'threading' for I/O bound tasks, 'loky' for CPU-intensive tasks
    
    # Verbosity
    verbose = 2 # 0=silent, 1=basic, 2=detailed
    
    # --- End of Settings ---
    
    # Get filenames from command line arguments if provided
    if len(sys.argv) > 1:
        source_files = sys.argv[1:]
    else:
        # Fallback to default if no arguments are given
        source_files = ['test_file1.csv', 'test_file2.csv']
        print(f"No input files provided. Using default files: {source_files}")

    try:
        pep_data_result = xTop(source_files, file_output=True)
        print("\nDone! xTop analysis finished successfully.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the input files are in the correct directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

