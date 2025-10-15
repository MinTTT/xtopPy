#%%
import xTop
# import numpy as np
import pandas as pd

import sys
#%%
pr_file = r'./test_file3.tsv'

#read python arguement
if len(sys.argv) > 1:
    pr_file = sys.argv[1]
elif len(sys.argv) == 1:
    print("No input file provided, using default test file.")

xTop.x_top_settings['xTopPepVal'] = [1]  # This array sets the overall normalization of the xTop intensities, exp features after xTop v2.0

pr_data = pd.read_csv(pr_file, sep='\t')

col_not_sample = ['Protein.Group',
 'Protein.Ids',
 'Protein.Names',
 'Genes',
 'First.Protein.Description',
 'Proteotypic',
 'Stripped.Sequence',
 'Modified.Sequence',
 'Precursor.Charge',
 'Precursor.Id']
Precursor_col_name = 'Precursor.Id'
Proteind_col_name = 'Genes'

input_data = pr_data[[Precursor_col_name, Proteind_col_name] + [col for col in pr_data.columns if col not in col_not_sample]]

xTop.xTop(input_data, file_output=True)

# results = xTop.xTop(input_data, file_output=False)
# # %%
# xTop_intensities = results[1]['Intensity_xTop1']
# # %%
