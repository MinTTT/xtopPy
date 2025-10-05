"""
Simple example: xTop with DataFrame input and file output

This example shows the basic usage of xTop with:
1. pandas DataFrame as input
2. file_output=True to save results to disk
3. Access to both saved files and in-memory results

@author: Simple example for xTop DataFrame + file output
@date: 2025-09-21
"""

import pandas as pd
import numpy as np
from xTop import xTop, rank_settings, top_pep_settings, x_top_settings

def simple_dataframe_file_output_example():
    """
    Simple example of using DataFrame input with file output
    """
    print("Simple xTop Example: DataFrame Input + File Output")
    print("=" * 50)
    
    # Configure basic settings
    rank_settings['rankFiles'] = 'individual'
    top_pep_settings['TopPepVal'] = [1, 3, np.inf]
    x_top_settings['xTopPepVal'] = [1]
    
    # Step 1: Load your data as a DataFrame
    df = pd.read_csv(r"G:\python_code\xtop\test_file1.csv")
    print(f"Data loaded: {df.shape[0]} peptides, {df.shape[1]-2} samples")
    
    # Step 2: Run xTop with DataFrame input and file output
    print("\nRunning xTop analysis...")
    pep_data, results = xTop(
        source_filename_arg=df,          # DataFrame input
        verbose=1,                       # Show progress
        file_output=True                 # Save files to disk
    )
    
    # Step 3: Access results (both in-memory and files are available)
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        
        # In-memory results
        topPep1 = result['Intensity_TopPep1']
        xTop1 = result['Intensity_xTop1']
        
        print(f"\nResults available:")
        print(f"- TopPep1 intensities: {topPep1.shape}")
        print(f"- xTop1 intensities: {xTop1.shape}")
        print(f"- Files saved in output directory")
        
        # Example: Get top proteins by intensity
        protein_sums = xTop1.iloc[:, 1:].sum(axis=1)
        top_proteins = protein_sums.nlargest(3)
        
        print(f"\nTop 3 proteins by total xTop intensity:")
        for i, (idx, total) in enumerate(top_proteins.items(), 1):
            protein_name = xTop1.iloc[idx, 0]
            print(f"  {i}. {protein_name}: {total:,.0f}")
    
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print("Both files and in-memory results are now available.")
    return results

if __name__ == "__main__":
    results = simple_dataframe_file_output_example()
