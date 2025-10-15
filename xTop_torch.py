# filepath: g:\python_code\xtop\xTop.py
"""
xTop Algorithm for Proteome Data Analysis
Converted from MATLAB to Python, with PyTorch optimization.

This function loads one or more files with peptide precursor intensities,
and generates one or more files with protein data, including the protein
intensity (TopPepN, TopPepAll and/or xTop), peptide ranking and detection
efficiencies, as well as other useful quantities.

Original MATLAB implementation: https://gitlab.com/mm87/xtop
Python conversion: 20250920
PyTorch implementation: 20250921
@author: Pan Chu
@data: 2025-09-21
@mail: pan_chu@outlook.com
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datetime import datetime
import warnings
from typing import List, Union, Dict, Any, Optional, Tuple

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torch')


# Global settings (similar to MATLAB's global variables)
rank_settings = {
    'rankFiles': 'individual',  # 'individual' or 'all'
    'filterThres': 0.5,         # Filter strength (0-1)
    'export_ranking': False     # Export peptide ranking files
}

top_pep_settings = {
    'TopPepVal': [],           # TopPepN values to compute (e.g., [1, 2, 3] or [1, np.inf])
    'export_Npep': False       # Export number of peptides detected
}

x_top_settings = {
    'xTopPepVal': [1],                    # xTop normalization values
    'thres_xTop': 300,                    # Intensity threshold
    'N_c_thres': 3,                       # Minimum samples for peptide
    's2_min': 0.01,                       # Variance regularization
    'absolute_filter_ratio': 0.1,         # Outlier filter ratio
    'error_type': 'relative (std-log)',   # Error type for output
    'plot_xTop_convergence': False,       # Debug plots
    'accuracy_max': 1e-10,                # Convergence criterion
    'N_step_max': 1000,                   # Maximum optimization steps
    'N_optimizations': 5,                # Number of optimization runs
    'rng_seed': 'shuffle',                # Random seed
    'export_level': 'full',               # 'basic' or 'full' output
    'parallelize_flag': True,             # Use parallel processing
    'max_parallel_workers': -1,            # Max workers (-1 = all cores)
    'gpu_batch_size': 100,                 # Number of proteins to process in parallel on GPU
    'max_gpu_elements': 50000              # Max elements in GPU memory
}

verbose = 1  # Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)

def xTop(source_filename_arg: Union[str, List[str]] = None,
         verbose_level=verbose, file_output: bool = False) -> Union[Dict, List[Dict]]:
    """
    This function loads one or more files with peptide precursor intensities,
    and generates one or more files with protein data.
    
    Args:
        source_filename_arg: Filenames (relative or absolute paths) to the sources with peptide data.
                           If None, uses default test files.
        verbose_level: Controls console output.
        file_output: If True, saves results to files.
    
    Returns:
        pepData: A dictionary (or list of dictionaries) containing peptide ranking information.
        results: A dictionary (or list of dictionaries) containing protein intensity data.
    """
    global rank_settings, top_pep_settings, x_top_settings, verbose
    verbose = verbose_level

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
        output_folder_name = f"xTop_torch_output_{datetime.now().strftime('%y-%m-%d-%H%M')}"
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
        pep_data = remove_problematic_peptides(pep_data, [])
        
        results = []
        for k in range(len(source_filename)):
            result_temp = export_counts_from_pep(pep_data, source_filename[k], 
                                   fname_output[k] if fname_output else None)
            results.append(result_temp)
        
        if rank_settings['export_ranking']:
            fname_pep_ranking = os.path.join(output_folder_name, "[GLOBAL] PepRank.csv") if file_output else None
            pep_rank_df = export_pep_ranking(pep_data, fname_pep_ranking)
            # This part of adding to results needs clarification on structure
            # For now, let's just export the file.
    
    elif rank_settings['rankFiles'] == 'individual':
        pep_data_list = []
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
            pep_data_list.append(current_pep_data)

            result_temp = export_counts_from_pep(current_pep_data, source_filename[k], fname_output[k] if fname_output else None)
            
            if rank_settings['export_ranking']:
                fname_pep_ranking = f"{fname_output[k]} PepRank.csv" if file_output else None
                pep_rank_df = export_pep_ranking(current_pep_data, fname_pep_ranking)
                result_temp['Peptide_Ranking'] = pep_rank_df
            results.append(result_temp)
        pep_data = pep_data_list
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
        if x_top_settings['max_parallel_workers'] == -1:
            print("* The xTop calculation will be parallelized using all available workers.")
        elif x_top_settings['max_parallel_workers'] > 0:
            print(f"* The xTop calculation will be parallelized using at most {x_top_settings['max_parallel_workers']} workers.")
        else:
            print("* The xTop calculation will not be parallelized.")
    else:
        print("* The xTop calculation will not be parallelized.")
    
    print("* * * * * * * * * * * * * * * * * * * *")

def rank_peptide_levels(file_list: List[Union[str, pd.DataFrame]]) -> Dict:
    """
    Rank peptides based on their levels across samples.
    """
    global rank_settings, verbose
    
    if not file_list:
        raise ValueError("Input file list is empty.")
    
    pep_data = {'input_file': file_list, 'X': []}
    
    all_peptides, all_proteins, all_conditions = [], [], []
    
    for f in file_list:
        if isinstance(f, pd.DataFrame):
            df = f
        else:
            file_ext = f.split('.')[-1].lower()
            sep = '\t' if file_ext in ['tsv', 'txt'] else ','
            try:
                df = pd.read_csv(f, sep=sep)
            except Exception as e:
                raise ValueError(f"Error reading file {f}: {e}")
        
        if df.shape[1] < 3:
            raise ValueError(f"File {f if isinstance(f, str) else 'DataFrame'} must have at least 3 columns.")
        
        samples = df.columns[2:].tolist()
        peptides = df.iloc[:, 0].astype(str).tolist()
        proteins = df.iloc[:, 1].astype(str).tolist()
        intensities = df.iloc[:, 2:].values.astype(float)
        
        n_pep, n_samples = intensities.shape
        unique_proteins_in_file = list(np.unique(proteins))
        
        prot_to_idx = {prot: i for i, prot in enumerate(unique_proteins_in_file)}
        pvec_all2unique = [prot_to_idx[prot] for prot in proteins]
        
        pep_data['X'].append({
            'filename': f if isinstance(f, str) else 'DataFrame',
            'sample_name': samples, 'pep_info': peptides, 'prot_info': proteins,
            'unique_protein': unique_proteins_in_file, 'pvec_all2unique': pvec_all2unique,
            'pepSig': intensities
        })
        
        all_peptides.extend(peptides)
        all_proteins.extend(proteins)
        all_conditions.extend(samples)
        
        if verbose:
            if len(file_list) > 1: print("+")
            source_name = f if isinstance(f, str) else f"DataFrame (shape: {f.shape})"
            print(f"+ + + Source file: {source_name} + + +")
            print(f"+  Number of samples: {n_samples}")
            print(f"+  Input number of peptides: {n_pep}")
            print(f"+  Input number of protein assignations: {len(unique_proteins_in_file)}")
    
    if verbose and len(file_list) > 1:
        print("+\n+ + + Merging datasets + + +")
    
    unique_peptides, u2apep, a2upep = np.unique(all_peptides, return_index=True, return_inverse=True)
    pep_data['unique_pepPrec'] = unique_peptides.tolist()
    n_upep = len(unique_peptides)
    
    pep_off = [0] + [len(pep_data['X'][k]['pep_info']) for k in range(len(file_list))]
    pep_off = np.cumsum(pep_off)
    
    pep_int_sum = np.zeros(n_upep)
    pep_n = np.zeros(n_upep)
    pep_max = np.zeros(n_upep)
    
    for k in range(len(file_list)):
        for pi in range(len(pep_data['X'][k]['pep_info'])):
            pu = a2upep[pep_off[k] + pi]
            intensities = pep_data['X'][k]['pepSig'][pi, :]
            pep_int_sum[pu] += np.sum(intensities)
            pep_n[pu] += np.sum(intensities > 0)
            pep_max[pu] = max(pep_max[pu], np.max(intensities) if intensities.size > 0 else 0)
    
    all_prot_2_subset = [all_proteins[u2apep[i]] for i in range(n_upep)]
    unique_proteins, _, pvec_all2unique = np.unique(all_prot_2_subset, return_index=True, return_inverse=True)
    pep_data['unique_protein'] = unique_proteins.tolist()
    n_prot = len(unique_proteins)
    
    if verbose and len(file_list) > 1:
        print(f"+  Total number of samples: {len(np.unique(all_conditions))}")
        print(f"+  Unique peptides: {n_upep}")
        print(f"+  Unique proteins: {n_prot}")
    
    pep_data['pep2prot'] = sp.csc_matrix((np.ones(n_upep), (np.arange(n_upep), pvec_all2unique)), shape=(n_upep, n_prot))
    pep_data.update({'pepInt': pep_int_sum, 'pepN': pep_n, 'pepMax': pep_max})
    
    if verbose:
        if rank_settings['filterThres'] == 0:
            print("+  Ranking peptides according to total intensity... ", end='')
        else:
            print(f"+  Filtering and ranking (filterThres = {rank_settings['filterThres']:.2f})... ", end='')
    
    cnt_prot_cut, cnt_pep_cut = 0, 0
    
    for pi in range(n_prot):
        v_pep = pep_data['pep2prot'][:, pi].nonzero()[0]
        if len(v_pep) == 0: continue
        
        top_pep_int = pep_int_sum[v_pep]
        top_pep_n = pep_n[v_pep]
        
        if rank_settings['filterThres'] == 0:
            sorted_idx = np.argsort(-top_pep_int)
            for rank, idx in enumerate(sorted_idx, 1):
                pep_data['pep2prot'][v_pep[idx], pi] = rank
        else:
            sorted_n_idx = np.argsort(-top_pep_n)
            n_max = top_pep_n[sorted_n_idx[0]] if len(sorted_n_idx) > 0 else 0
            
            sorted_int_idx = np.argsort(-top_pep_int)
            
            avg_pep_int = top_pep_int / (top_pep_n + 1e-8)
            avg_sign_ref = avg_pep_int[sorted_n_idx[0]] if len(sorted_n_idx) > 0 else 0
            
            use_pep = ~((top_pep_n < rank_settings['filterThres'] * n_max) & (avg_pep_int > avg_sign_ref))
            
            rank_counter = 0
            for idx in sorted_int_idx:
                if use_pep[np.where(v_pep[sorted_int_idx] == v_pep[idx])[0][0]]:
                    rank_counter += 1
                    pep_data['pep2prot'][v_pep[idx], pi] = rank_counter
                else:
                    pep_data['pep2prot'][v_pep[idx], pi] = 0
            
            if np.sum(use_pep) < len(use_pep):
                cnt_prot_cut += 1
                cnt_pep_cut += len(use_pep) - np.sum(use_pep)
    
    del pep_data['X']
    
    if verbose:
        print("ok")
        if rank_settings['filterThres'] > 0:
            print(f"+  -> Peptides filtered out: {cnt_pep_cut}")
            print(f"+  -> Proteins affected: {cnt_prot_cut}")
    
    return pep_data

def remove_problematic_peptides(pep_data: Dict, pep_list: List[str]) -> Dict:
    """Remove specified problematic peptides from the analysis."""
    global verbose
    if not pep_list: return pep_data
    
    cnt_removed = 0
    if verbose: print("\n* Removing problematic peptide(s)...")
    
    pep_indices_to_remove = [i for i, pep in enumerate(pep_data['unique_pepPrec']) if pep in pep_list]
    
    # Create a mask for peptides to keep
    keep_mask = np.ones(len(pep_data['unique_pepPrec']), dtype=bool)
    keep_mask[pep_indices_to_remove] = False
    
    # Update ranks for affected proteins before removing peptides
    for i in sorted(pep_indices_to_remove, reverse=True):
        p_cols = pep_data['pep2prot'][i, :].nonzero()[1]
        if p_cols.size > 0:
            p = p_cols[0]
            ri = pep_data['pep2prot'][i, p]
            if ri > 0:
                jj_up = (pep_data['pep2prot'][:, p].toarray().flatten() > ri)
                pep_data['pep2prot'][jj_up, p] -= 1
    
    # Filter all relevant data structures
    pep_data['unique_pepPrec'] = [pep for i, pep in enumerate(pep_data['unique_pepPrec']) if keep_mask[i]]
    pep_data['pep2prot'] = pep_data['pep2prot'][keep_mask, :]
    pep_data['pepInt'] = pep_data['pepInt'][keep_mask]
    pep_data['pepN'] = pep_data['pepN'][keep_mask]
    pep_data['pepMax'] = pep_data['pepMax'][keep_mask]
    
    cnt_removed = len(pep_indices_to_remove)
    if verbose: print(f"*   {cnt_removed} out of {len(pep_list)} removed.")
    
    return pep_data

def export_counts_from_pep(pep_data: Dict, source_file: Union[str, pd.DataFrame], output_base: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Export protein counts from peptide data."""
    global top_pep_settings, x_top_settings, verbose

    output_dict = {}

    if isinstance(source_file, pd.DataFrame):
        df = source_file
    else:
        file_ext = source_file.split('.')[-1].lower()
        sep = '\t' if file_ext in ['tsv', 'txt'] else ','
        try:
            df = pd.read_csv(source_file, sep=sep)
        except Exception as e:
            raise ValueError(f"Error reading file {source_file}: {e}")
    
    if df.shape[1] < 3:
        raise ValueError(f"File must have at least 3 columns, but has {df.shape[1]}.")
    
    data_samples = df.columns[2:].tolist()
    data_pep_list = df.iloc[:, 0].values
    data_prot_list = df.iloc[:, 1].values
    data_values = df.iloc[:, 2:].values.astype(float)
    
    unique_prot = np.unique(data_prot_list)
    n_prot, n_samples = len(unique_prot), len(data_samples)
    
    if verbose:
        source_name = "DataFrame input" if isinstance(source_file, pd.DataFrame) else output_base
        print(f"\n* * * Protein intensities\n* Output:   {source_name}")
    
    pep_name_map, pep_rank_map = {}, {}
    for pi, prot_name in enumerate(unique_prot):
        pj = np.where(np.array(pep_data['unique_protein']) == prot_name)[0]
        if len(pj) == 0: continue
        
        qj = pep_data['pep2prot'][:, pj[0]].nonzero()[0]
        pep_names = [pep_data['unique_pepPrec'][i] for i in qj]
        pep_ranks = pep_data['pep2prot'][qj, pj[0]].toarray().flatten()
        
        pep_name_map[prot_name] = pep_names
        pep_rank_map[prot_name] = {name: rank for name, rank in zip(pep_names, pep_ranks) if rank > 0}

    for top_pep_val in top_pep_settings['TopPepVal']:
        str_x = f"TopPep{int(top_pep_val)}" if not np.isinf(top_pep_val) else "TopPepAll"
        if verbose: print(f"*   {str_x} intensity ... ", end='')
        
        m_top_pep_n = np.zeros((n_prot, n_samples))
        m_n_pep = np.zeros((n_prot, n_samples))
        
        for pi, prot_name in enumerate(unique_prot):
            if prot_name not in pep_rank_map: continue
            
            rank_map = pep_rank_map[prot_name]
            
            # Find peptides in the current dataset that have a rank
            pep_indices = [i for i, p in enumerate(data_pep_list) if p in rank_map]
            if not pep_indices: continue

            current_peps = data_pep_list[pep_indices]
            current_values = data_values[pep_indices, :]
            current_ranks = np.array([rank_map.get(p, 0) for p in current_peps])
            
            if np.isinf(top_pep_val):
                selected = current_ranks > 0
            else:
                selected = (current_ranks > 0) & (current_ranks <= top_pep_val)
            
            if np.any(selected):
                m_top_pep_n[pi, :] = np.sum(current_values[selected, :], axis=0)
                m_n_pep[pi, :] = np.sum(current_values[selected, :] > 0, axis=0)
        
        df_out = export_dlm_data(f"{output_base} Intensity {str_x}.csv" if output_base else None, data=[unique_prot] + list(m_top_pep_n.T), header=['Protein'] + data_samples)
        output_dict[f'Intensity_{str_x}'] = df_out

        if top_pep_settings['export_Npep']:
            if verbose: print("Npep ...", end='')
            df_out_npep = export_dlm_data(f"{output_base} Npep {str_x}.csv" if output_base else None, data=[unique_prot] + list(m_n_pep.T), header=['Protein'] + data_samples)
            output_dict[f'Npep_{str_x}'] = df_out_npep

        if verbose: print(" ok")

    if x_top_settings['xTopPepVal']:
        if verbose == 1: print("*   xTop ... ", end='')
        elif verbose > 1: print("* * * xTop protein intensities * * *")

        # Prepare data for GPU batch processing
        protein_data = []
        protein_indices = []
        
        for pi, prot_name in enumerate(unique_prot):
            if prot_name not in pep_name_map: continue
            
            pep_names = pep_name_map[prot_name]
            in_dataset_mask = np.isin(data_pep_list, pep_names)
            qi = np.where(in_dataset_mask)[0]
            if qi.size > 0:
                protein_data.append(data_values[qi, :])
                protein_indices.append(pi)

        # Use GPU batch processing instead of CPU threading
        results = {}
        if protein_data:
            if x_top_settings['parallelize_flag']:
                # GPU batch processing
                results = process_proteins_gpu_batch(protein_data, protein_indices, x_top_settings, verbose)
            else:
                # Sequential processing
                for idx, (pi, I_pep) in enumerate(zip(protein_indices, protein_data)):
                    results[pi] = get_xTop_estimator_wrapper(I_pep, x_top_settings, verbose)

        m_i_xtop = np.zeros((n_prot, n_samples))
        m_log_i_xtop_var = np.zeros((n_prot, n_samples))
        all_pep_stats = []

        for pi, (xTop_output, _) in results.items():
            if xTop_output is None: continue
            
            m_i_xtop[pi, :] = xTop_output['I_xTop']
            m_log_i_xtop_var[pi, :] = xTop_output['log_I_xTop_var']
            
            if x_top_settings['export_level'] == 'full':
                pep_names = pep_name_map[unique_prot[pi]]
                in_dataset_mask = np.isin(data_pep_list, pep_names)
                
                filtered_peptides = data_pep_list[in_dataset_mask]
                filtered_values = data_values[in_dataset_mask, :]
                
                n_pep_in_prot = len(filtered_peptides)
                if n_pep_in_prot > 0:
                    eff_values = xTop_output['eff']
                    pep_ranks = np.argsort(-eff_values) + 1
                    
                    pep_stats = pd.DataFrame({
                        'Peptide': filtered_peptides,
                        'protein': unique_prot[pi],
                        'pep_rank': pep_ranks,
                        'pep_detections': np.sum(filtered_values > 0, axis=1),
                        'relQuant_isValid': xTop_output['relQuant_valid_flag'],
                        'absQuant_isValid': xTop_output['absQuant_valid_flag'],
                        'pep_weight': xTop_output['pep_weight'],
                        'eff': eff_values,
                        'log_eff_var': xTop_output['log_eff_var']
                    })
                    all_pep_stats.append(pep_stats)

        for val in x_top_settings['xTopPepVal']:
            str_x = str(val) if not np.isinf(val) else "All"
            final_intensities = m_i_xtop * val
            
            df_out_xtop = export_dlm_data(f"{output_base} Intensity xTop{str_x}.csv" if output_base else None, [unique_prot] + list(final_intensities.T), header=['Protein'] + data_samples)
            output_dict[f'Intensity_xTop{str_x}'] = df_out_xtop
            
            if x_top_settings['export_level'] == 'full':
                error_type = x_top_settings['error_type']
                if error_type == 'relative (std-log)':
                    errors = np.sqrt(m_log_i_xtop_var)
                    fname = f"{output_base} Error (std-log) xTop{str_x}.csv"
                elif error_type == 'relative (CV)':
                    errors = np.sqrt(np.exp(m_log_i_xtop_var) - 1)
                    fname = f"{output_base} Error (CV) xTop{str_x}.csv"
                else: # 'absolute'
                    cv = np.sqrt(np.exp(m_log_i_xtop_var) - 1)
                    errors = (final_intensities * cv)
                    fname = f"{output_base} Error xTop{str_x}.csv"
                
                df_out_err = export_dlm_data(fname if output_base else None, [unique_prot] + list(errors.T), header=['Protein'] + data_samples)
                output_dict[f'Error_{error_type}_xTop{str_x}'] = df_out_err

        if x_top_settings['export_level'] == 'full' and all_pep_stats:
            all_pep_stats_df = pd.concat(all_pep_stats).sort_values(by=['protein', 'pep_rank'])
            if output_base:
                all_pep_stats_df.to_csv(f"{output_base} xTop peptide statistics.csv", index=False)
            output_dict['xTop_peptide_statistics'] = all_pep_stats_df

        if verbose: print("ok")
    return output_dict

def export_dlm_data(filename: Optional[str], data: List, header: List[str], delimiter: str = ',') -> pd.DataFrame:
    """Exports data to a file and returns a DataFrame."""
    df = pd.DataFrame(dict(zip(header, data)))
    if filename:
        df.to_csv(filename, sep=delimiter, index=False, header=True)
    return df

def process_proteins_gpu_batch(protein_data: List[np.ndarray], protein_indices: List[int], 
                              xTopSettings: Dict, verbose: int) -> Dict:
    """
    Process multiple proteins in batches on GPU for better performance.
    
    Args:
        protein_data: List of intensity matrices for each protein
        protein_indices: List of protein indices corresponding to protein_data
        xTopSettings: xTop algorithm settings
        verbose: Verbosity level
    
    Returns:
        Dictionary mapping protein indices to optimization results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # Process proteins with similar sizes together for better GPU utilization
    protein_sizes = [(len(data), len(data[0]) if len(data) > 0 else 0, idx, pi) 
                     for idx, (data, pi) in enumerate(zip(protein_data, protein_indices))]
    protein_sizes.sort(key=lambda x: (x[0], x[1]))  # Sort by number of peptides and samples
    
    # Define batch processing parameters
    max_batch_size = xTopSettings.get('gpu_batch_size', 100)  # Process up to 8 proteins simultaneously
    max_total_elements = xTopSettings.get('max_gpu_elements', 50000)  # Memory constraint
    
    batch_start = 0
    while batch_start < len(protein_sizes):
        # Determine batch size based on memory constraints
        batch_end = batch_start + 1
        total_elements = protein_sizes[batch_start][0] * protein_sizes[batch_start][1]
        
        while (batch_end < len(protein_sizes) and 
               batch_end - batch_start < max_batch_size and
               total_elements + protein_sizes[batch_end][0] * protein_sizes[batch_end][1] < max_total_elements):
            total_elements += protein_sizes[batch_end][0] * protein_sizes[batch_end][1]
            batch_end += 1
        
        # Process current batch
        batch_results = process_protein_batch_gpu(
            [protein_data[protein_sizes[i][2]] for i in range(batch_start, batch_end)],
            [protein_indices[protein_sizes[i][2]] for i in range(batch_start, batch_end)],
            xTopSettings, verbose, device
        )
        
        # Merge results
        results.update(batch_results)
        batch_start = batch_end
    
    return results

def process_protein_batch_gpu(batch_data: List[np.ndarray], batch_indices: List[int],
                             xTopSettings: Dict, verbose: int, device: torch.device) -> Dict:
    """
    Process a batch of proteins simultaneously on GPU.
    """
    results = {}
    
    # For proteins that can be processed together (similar structure)
    if len(batch_data) == 1:
        # Single protein - use existing method
        results[batch_indices[0]] = get_xTop_estimator_wrapper(batch_data[0], xTopSettings, verbose)
    else:
        # Multiple proteins - try to process together if possible
        try:
            # Check if all proteins have similar dimensions for batch processing
            shapes = [data.shape for data in batch_data]
            if len(set(shapes)) == 1:
                # All proteins have same shape - can do true batch processing
                results.update(process_identical_proteins_gpu(batch_data, batch_indices, xTopSettings, verbose, device))
            else:
                # Different shapes - process individually but on GPU
                for data, pi in zip(batch_data, batch_indices):
                    results[pi] = get_xTop_estimator_wrapper(data, xTopSettings, verbose)
        except Exception as e:
            # Fallback to individual processing
            if verbose > 1:
                print(f"Batch processing failed, falling back to individual processing: {e}")
            for data, pi in zip(batch_data, batch_indices):
                results[pi] = get_xTop_estimator_wrapper(data, xTopSettings, verbose)
    
    return results

def process_identical_proteins_gpu(batch_data: List[np.ndarray], batch_indices: List[int],
                                  xTopSettings: Dict, verbose: int, device: torch.device) -> Dict:
    """
    Process proteins with identical dimensions simultaneously on GPU.
    """
    if not batch_data:
        return {}
    
    results = {}
    batch_size = len(batch_data)
    n_optimizations = xTopSettings.get('N_optimizations', 1)
    
    # Stack all protein data into a single tensor for batch processing
    stacked_data = np.stack(batch_data, axis=0)  # Shape: (batch_size, n_peptides, n_samples)
    
    # Process each optimization run
    best_results = {}
    for opt_run in range(n_optimizations):
        if verbose > 1 and n_optimizations > 1:
            print(f" * * * Batch optimization run {opt_run + 1}/{n_optimizations}")
        
        batch_results = run_batch_optimization_gpu(stacked_data, xTopSettings, device)
        
        # Select best results for each protein
        for i, pi in enumerate(batch_indices):
            if batch_results and i < len(batch_results):
                if pi not in best_results or (batch_results[i] and 
                    batch_results[i][0] and 'L_optimal' in batch_results[i][0] and
                    (not best_results[pi][0] or batch_results[i][0]['L_optimal'] < best_results[pi][0]['L_optimal'])):
                    best_results[pi] = batch_results[i]
    
    # Convert to expected format
    for pi in batch_indices:
        if pi in best_results:
            results[pi] = best_results[pi]
        else:
            # Fallback for failed optimizations
            results[pi] = get_xTop_estimator_wrapper(batch_data[batch_indices.index(pi)], xTopSettings, verbose)
    
    return results

def run_batch_optimization_gpu(stacked_data: np.ndarray, xTopSettings: Dict, device: torch.device) -> List:
    """
    Run optimization for a batch of proteins simultaneously on GPU.
    """
    batch_size, n_peptides, n_samples = stacked_data.shape
    results = []
    
    try:
        # Convert to torch tensors
        batch_tensor = torch.from_numpy(stacked_data).float().to(device)
        
        # Apply filtering and preprocessing for each protein in the batch
        thres_xTop = xTopSettings['thres_xTop']
        N_c_thres = xTopSettings['N_c_thres']
        
        # Vectorized filtering
        valid_peptide_flags = torch.sum(batch_tensor > thres_xTop, dim=2) >= N_c_thres
        
        # Process each protein individually (since they may have different valid peptides)
        for b in range(batch_size):
            try:
                protein_data = stacked_data[b]
                optimization_status, result = _get_xTop_estimator_torch(protein_data, xTopSettings, device)
                
                if result is not None:
                    # Convert to expected output format
                    (I_xTop, log_I_xTop_var, eff, log_eff_var, pep_weight, 
                     relQuant_valid_flag, absQuant_valid_flag, N_detected, N_valid_peptides, 
                     valid_condition_flag, s2_init, N_step, L_optimal, diff_L, L_iter) = result
                    
                    xTop_output = {
                        'Nc': protein_data.shape[1], 'N_valid_conditions': np.sum(valid_condition_flag),
                        'N_valid_peptides': N_valid_peptides, 'valid_peptide_flag': relQuant_valid_flag,
                        'valid_condition_flag': valid_condition_flag, 'N_detected': N_detected,
                        'I_xTop': I_xTop, 'log_I_xTop_var': log_I_xTop_var, 'eff': eff,
                        'log_eff_var': log_eff_var, 'pep_weight': pep_weight,
                        'relQuant_valid_flag': relQuant_valid_flag, 'absQuant_valid_flag': absQuant_valid_flag,
                        'optimization_status': optimization_status, 's2_init': s2_init, 'N_step': N_step,
                        'L': L_iter, 'L_optimal': L_optimal, 'diff_L': diff_L,
                    }
                    
                    # Create wrapper output format
                    xTop_wrapper_output = {
                        'status': [optimization_status], 'opt_idx': 0,
                        'N_detected': N_detected, 'N_valid_peptides': N_valid_peptides,
                        'N_step': [N_step], 'L': [L_optimal], 'L_optimal': L_optimal,
                        'L_range': 0, 'N_local_minima': 0
                    }
                    
                    results.append((xTop_output, xTop_wrapper_output))
                else:
                    results.append((None, {}))
                    
            except Exception as e:
                results.append((None, {}))
                
    except Exception as e:
        # Fallback: return empty results for the entire batch
        results = [(None, {}) for _ in range(batch_size)]
    
    return results

def get_xTop_estimator_wrapper(I_peptides: np.ndarray, xTopSettings: Dict, verbose: int) -> Tuple[Optional[Dict], Dict]:
    """Calls get_xTop_estimator multiple times and picks the best solution."""
    n_optimizations = xTopSettings.get('N_optimizations', 1)
    
    if n_optimizations == 1:
        xTop_output = get_xTop_estimator(I_peptides, xTopSettings, verbose)
        if xTop_output is None: return None, {}
        L_optimal = xTop_output['L_optimal']
        result_status = [xTop_output['optimization_status']]
        result_L = [L_optimal]
        result_N_step = [xTop_output['N_step']]
        result_opt_idx = 0
    else:
        if verbose > 1: print(f" * * * N_optimizations = {n_optimizations}:")
        
        results = [get_xTop_estimator(I_peptides, xTopSettings, verbose) for _ in range(n_optimizations)]
        results = [r for r in results if r is not None and r['L_optimal'] is not None]
        
        if not results: return None, {}

        result_status = np.array([r['optimization_status'] for r in results])
        result_L = np.array([r['L_optimal'] for r in results])
        result_N_step = np.array([r['N_step'] for r in results])
        
        result_opt_idx = np.argmin(result_L)
        xTop_output = results[result_opt_idx]
        L_optimal = result_L[result_opt_idx]

    xTop_wrapper_output = {
        'status': result_status, 'opt_idx': result_opt_idx,
        'N_detected': xTop_output['N_detected'], 'N_valid_peptides': xTop_output['N_valid_peptides'],
        'N_step': result_N_step, 'L': result_L, 'L_optimal': L_optimal,
        'L_range': np.max(result_L) - L_optimal if n_optimizations > 1 else 0,
        'N_local_minima': np.sum((result_L - L_optimal) > 1e2 * xTopSettings['accuracy_max'] * abs(L_optimal))
    }
    
    if verbose > 1 and n_optimizations > 1:
        print(f"---> max(L)-min(L) = {xTop_wrapper_output['L_range']}\n")
        
    return xTop_output, xTop_wrapper_output

def _get_xTop_estimator_torch(I_peptides: np.ndarray, settings: Dict, device: torch.device) -> Tuple:
    """PyTorch-based core of the xTop estimator."""
    Npep, Nc = I_peptides.shape
    s2_min, thres_xTop, N_c_thres = settings['s2_min'], settings['thres_xTop'], settings['N_c_thres']
    accuracy_max, N_step_max = settings['accuracy_max'], settings['N_step_max']

    # --- Data Filtering ---
    valid_peptide_flag = np.sum(I_peptides > thres_xTop, axis=1) >= N_c_thres
    I_peptides_valid = I_peptides[valid_peptide_flag, :]
    
    if I_peptides_valid.shape[0] == 0: return -1, None
    
    isDetected = I_peptides_valid > 0
    peptidesInSample = np.sum(isDetected, axis=0)
    if np.max(peptidesInSample) > 1:
        otherPeptidesDetected = np.array([np.max(peptidesInSample[isDetected[k, :]]) - 1 if np.any(isDetected[k, :]) else 0 for k in range(I_peptides_valid.shape[0])])
        isolatedPeptide = (otherPeptidesDetected == 0)
        if np.sum(isolatedPeptide) < np.max(peptidesInSample):
            valid_peptide_flag[np.where(valid_peptide_flag)[0][isolatedPeptide]] = False
            I_peptides_valid = I_peptides[valid_peptide_flag, :]

    valid_condition_flag = np.sum(I_peptides_valid > thres_xTop, axis=0) > 0
    I_peptides_final = I_peptides_valid[:, valid_condition_flag]
    
    Npep_nz, Nc_nz = I_peptides_final.shape
    N_detected = np.sum(I_peptides_final > 0)
    
    # --- Handle Edge Cases ---
    if Npep_nz == 0: return -1, None
    if Npep_nz == 1:
        I_xTop = np.zeros(Nc)
        I_xTop[valid_condition_flag] = I_peptides_final[0, :]
        return -2, (I_xTop, np.zeros(Nc), np.ones(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=bool), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1))
    if 2 * Npep_nz + Nc_nz > N_detected:
        I_xTop = np.zeros(Nc)
        I_xTop[valid_condition_flag] = np.max(I_peptides_final, axis=0)
        return -3, (I_xTop, np.zeros(Nc), np.ones(Npep), np.zeros(Npep), np.zeros(Npep), valid_peptide_flag, np.zeros(Npep, dtype=bool), N_detected, Npep_nz, valid_condition_flag, np.zeros(1), 0, 0.0, 0.0, np.zeros(1))

    # --- PyTorch Optimization Setup ---
    y_nz = torch.from_numpy(np.log(I_peptides_final[I_peptides_final > 0])).float().to(device)
    rows, cols = np.where(I_peptides_final > 0)
    rows, cols = torch.from_numpy(rows).long().to(device), torch.from_numpy(cols).long().to(device)
    
    log_eff = torch.zeros(Npep_nz, device=device, requires_grad=True)
    log_int = torch.zeros(Nc_nz, device=device, requires_grad=True)
    s2 = torch.from_numpy(s2_min * (100 + 400 * np.random.rand(Npep_nz))).float().to(device)
    s2_init = s2.clone().cpu().numpy()
    
    optimizer = torch.optim.LBFGS([log_eff, log_int], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")
    
    L_iter = np.zeros(N_step_max)
    diff_L = 1.0
    step = 0

    while diff_L > accuracy_max and step < N_step_max:
        def closure():
            optimizer.zero_grad()
            residuals = log_eff[rows] + log_int[cols] - y_nz
            loss = 0.5 * torch.sum(residuals**2 / s2[rows])
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        with torch.no_grad():
            residuals = log_eff[rows] + log_int[cols] - y_nz
            for p in range(Npep_nz):
                iy = (rows == p)
                if torch.any(iy):
                    s2[p] = s2_min + torch.sum(residuals[iy]**2) / torch.sum(iy)
            
            L_val = 0.5 * torch.sum(residuals**2 / s2[rows]) + 0.5 * torch.sum(torch.bincount(rows, minlength=Npep_nz) * (s2_min / s2 + torch.log(s2)))
            L_iter[step] = L_val.item()
            
            if step > 0:
                diff_L = abs(L_iter[step] / L_iter[step-1] - 1) if L_iter[step-1] != 0 else 1.0
        step += 1

    # --- Finalize Results ---
    optimization_status = 1 if diff_L <= accuracy_max else 2
    with torch.no_grad():
        s2_pep_valid = s2.cpu().numpy()
        s2_absThres = s2_min / settings['absolute_filter_ratio']
        idxAbs = s2_pep_valid < s2_absThres
        
        reliable_absQuant = np.sum(idxAbs) > 0
        if reliable_absQuant:
            max_eff_val = torch.max(log_eff[torch.from_numpy(idxAbs).to(device)])
        else:
            max_eff_val = torch.max(log_eff)
        
        log_eff -= max_eff_val
        log_int += max_eff_val
        
        eff_valid = torch.exp(log_eff).cpu().numpy()
        I_xTop_valid = torch.exp(log_int).cpu().numpy()

    eff = np.zeros(Npep)
    eff[valid_peptide_flag] = eff_valid
    
    I_xTop = np.zeros(Nc)
    I_xTop[valid_condition_flag] = I_xTop_valid
    I_xTop[I_xTop < thres_xTop] = 0
    
    # Simplified error estimation (Hessian inverse is complex here)
    log_eff_var = np.zeros(Npep)
    log_I_xTop_var = np.zeros(Nc)
    
    pep_weight = np.zeros(Npep)
    pep_weight[valid_peptide_flag] = 1.0 / s2_pep_valid
    
    absQuant_valid_flag = np.zeros(Npep, dtype=bool)
    if reliable_absQuant:
        absQuant_valid_flag[np.where(valid_peptide_flag)[0][idxAbs]] = True

    return optimization_status, (I_xTop, log_I_xTop_var, eff, log_eff_var, pep_weight, valid_peptide_flag, absQuant_valid_flag, N_detected, Npep_nz, valid_condition_flag, s2_init, step, L_iter[step-1], diff_L, L_iter)

def get_xTop_estimator(I_peptides: np.ndarray, xTopSettings: Dict, verbose: int) -> Optional[Dict]:
    """Main function for calculating the xTop estimator using PyTorch."""
    if xTopSettings['rng_seed'] != 'shuffle':
        np.random.seed(xTopSettings['rng_seed'])
        torch.manual_seed(xTopSettings['rng_seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    optimization_status, results = _get_xTop_estimator_torch(I_peptides, xTopSettings, device)

    if results is None:
        return None

    (I_xTop, log_I_xTop_var, eff, log_eff_var, pep_weight, 
     relQuant_valid_flag, absQuant_valid_flag, N_detected, N_valid_peptides, 
     valid_condition_flag, s2_init, N_step, L_optimal, diff_L, L_iter) = results

    xTop_output = {
        'Nc': I_peptides.shape[1], 'N_valid_conditions': np.sum(valid_condition_flag),
        'N_valid_peptides': N_valid_peptides, 'valid_peptide_flag': relQuant_valid_flag,
        'valid_condition_flag': valid_condition_flag, 'N_detected': N_detected,
        'I_xTop': I_xTop, 'log_I_xTop_var': log_I_xTop_var, 'eff': eff,
        'log_eff_var': log_eff_var, 'pep_weight': pep_weight,
        'relQuant_valid_flag': relQuant_valid_flag, 'absQuant_valid_flag': absQuant_valid_flag,
        'optimization_status': optimization_status, 's2_init': s2_init, 'N_step': N_step,
        'L': L_iter, 'L_optimal': L_optimal, 'diff_L': diff_L,
    }

    if verbose > 1:
        print(f"Optimization status: {optimization_status:2d}     Steps: {N_step:3d}     L : {L_optimal:10.5f}   diff_L : {diff_L:8.5e}")
        
    return xTop_output

def export_pep_ranking(pep_data: Dict, filename: Optional[str]) -> pd.DataFrame:
    """Export peptide ranking to CSV file."""
    prot_name_col, pep_name_col, rank_col = [], [], []

    for a in range(len(pep_data['unique_protein'])):
        prot_name = pep_data['unique_protein'][a]
        pep_indices, _, pep_ranks = sp.find(pep_data['pep2prot'][:, a])
        
        ranked_peptides = pep_indices[pep_ranks > 0]
        ranks = pep_ranks[pep_ranks > 0]
        
        for idx in np.argsort(ranks):
            pep_index = ranked_peptides[idx]
            prot_name_col.append(prot_name)
            pep_name_col.append(pep_data['unique_pepPrec'][pep_index])
            rank_col.append(ranks[idx])

    peptide_rank_df = pd.DataFrame({'Protein': prot_name_col, 'Peptide': pep_name_col, 'Rank': rank_col})
    if filename:
        peptide_rank_df.to_csv(filename, index=False)
    return peptide_rank_df

if __name__ == '__main__':
    import sys
    
    # --- Default Settings ---
    rank_settings['rankFiles'] = 'individual'
    rank_settings['filterThres'] = 0.5
    rank_settings['export_ranking'] = True
    
    top_pep_settings['TopPepVal'] = [1, 3, np.inf]
    top_pep_settings['export_Npep'] = True
    
    x_top_settings['xTopPepVal'] = [1]
    x_top_settings['export_level'] = 'full'
    x_top_settings['parallelize_flag'] = True
    
    verbose_level = 2 # 0=silent, 1=basic, 2=detailed
    
    # --- End of Settings ---
    
    if len(sys.argv) > 1:
        source_files = sys.argv[1:]
    else:
        source_files = ['test_file1.csv', 'test_file2.csv']
        print(f"No input files provided. Using default files: {source_files}")
        # Create dummy files for testing if they don't exist
        for fname in source_files:
            if not os.path.exists(fname):
                print(f"Creating dummy file: {fname}")
                n_pep, n_prot, n_cond = 100, 10, 8
                peptides = [f"PEPTIDE{i}" for i in range(n_pep)]
                proteins = [f"PROT{i % n_prot}" for i in range(n_pep)]
                np.random.shuffle(proteins)
                data = {'Peptide': peptides, 'Protein': proteins}
                for c in range(n_cond):
                    intensities = np.random.lognormal(mean=10, sigma=2, size=n_pep)
                    intensities[np.random.rand(n_pep) < 0.3] = 0 # 30% missing values
                    data[f'Sample_{c+1}'] = intensities.astype(int)
                pd.DataFrame(data).to_csv(fname, index=False)

    try:
        pep_data_result, results = xTop(source_files, verbose_level=verbose_level, file_output=True)
        print("\nDone! xTop analysis finished successfully.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the input files are in the correct directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


        