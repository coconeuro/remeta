# %%
import pandas as pd
import numpy as np
import remeta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('error')


d = pd.read_csv(f'data_Maniscalco_2017_expt2.csv')
d = d[~d.Response.isna()].reset_index(drop=True)

choices = d.Response.astype(int).values
stimulus_ids = np.sign(d.Stimulus.values - 0.5).astype(int)
difficulty_levels = d.Contrast.values
# remeta.check_linearity(stimulus_ids, choices, difficulty_levels)
# plt.savefig('../content/img/linearization_example.png', bbox_inches='tight', pad_inches=0.02)
# # %%
# stimuli_linear = remeta.linearize_stimulus_evidence(stimulus_ids, choices, difficulty_levels)
# # %%
# remeta.check_linearity(stimuli_linear, choices)
# plt.savefig('../content/img/linearization_example_after.png', bbox_inches='tight', pad_inches=0.02)
# # %%
# stimuli_linear2 = remeta.linearize_stimulus_evidence(stimulus_ids, choices, difficulty_levels, method='discretize_linear')
# remeta.check_linearity(stimuli_linear2, choices)
# %%
stimuli_linear3 = remeta.linearize_stimulus_evidence(stimulus_ids, choices, difficulty_levels, method='discretize_linear', discretize_nlevels=15)
remeta.check_linearity(stimuli_linear3, choices)
