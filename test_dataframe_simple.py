"""
Simple test demonstrating xTop DataFrame mode with no file output

This script demonstrates how to:
1. Read a CSV file as a pandas DataFrame
2. Pass it directly to xTop function 
3. Get analysis results without saving any output files
4. Access and display the results

@author: Test script for xTop DataFrame mode
@date: 2025-09-21
"""

import pandas as pd
import numpy as np
import os
from xTop import xTop, rank_settings, top_pep_settings, x_top_settings

def main():
    """
    Simple demonstration of xTop DataFrame mode
    """
    print("xTop DataFrame Mode - Simple Test")
    print("=" * 50)
    
    # Configure settings
    rank_settings['rankFiles'] = 'individual'
    rank_settings['filterThres'] = 0.5
    rank_settings['export_ranking'] = True
    
    top_pep_settings['TopPepVal'] = [1, 3, np.inf]
    top_pep_settings['export_Npep'] = True
    
    x_top_settings['xTopPepVal'] = [1]
    x_top_settings['export_level'] = 'full'
    
    try:
        # Step 1: Read CSV file as DataFrame
        print("\n1. Reading test_file1.csv as DataFrame...")
        csv_file_path = r"G:\python_code\xtop\test_file1.csv"
        df = pd.read_csv(csv_file_path)
        
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Peptides: {len(df)}")
        print(f"   Proteins: {df.iloc[:, 1].nunique()}")
        print(f"   Samples: {len(df.columns) - 2}")
        
        # Step 2: Run xTop with DataFrame input, no file output
        print("\n2. Running xTop analysis...")
        pep_data, results = xTop(source_filename_arg=df, verbose=1, file_output=False)
        
        # Step 3: Display results
        print("\n3. Analysis Results:")
        print("-" * 30)
        
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            
            # Show available data types
            print(f"Available result types: {list(result.keys())}")
            
            # Display TopPep1 intensities
            if 'Intensity_TopPep1' in result:
                topPep1 = result['Intensity_TopPep1']
                print(f"\nTopPep1 Intensities:")
                print(f"  Shape: {topPep1.shape}")
                print(f"  Non-zero proteins: {(topPep1.iloc[:, 1:].sum(axis=1) > 0).sum()}")
                print(f"  Sample data (first 3 proteins):")
                print(topPep1.head(3).to_string(index=False))
            
            # Display xTop intensities
            if 'Intensity_xTop1' in result:
                xTop1 = result['Intensity_xTop1']
                print(f"\nxTop1 Intensities:")
                print(f"  Shape: {xTop1.shape}")
                print(f"  Non-zero proteins: {(xTop1.iloc[:, 1:].sum(axis=1) > 0).sum()}")
                print(f"  Sample data (first 3 proteins):")
                print(xTop1.head(3).to_string(index=False))
            
            # Display peptide ranking
            if 'Peptide_Ranking' in result and 'Peptide_Ranking' in result['Peptide_Ranking']:
                ranking = result['Peptide_Ranking']['Peptide_Ranking']
                print(f"\nPeptide Rankings:")
                print(f"  Total peptides ranked: {len(ranking)}")
                print(f"  Unique proteins: {ranking['Protein'].nunique()}")
                print(f"  Sample rankings (first 5):")
                print(ranking.head(5).to_string(index=False))
        
        print("\n" + "=" * 50)
        print("SUCCESS: DataFrame mode works correctly!")
        print("All analysis results retrieved without file output.")
        print("=" * 50)
        
        return pep_data, results
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    pep_data, results = main()
